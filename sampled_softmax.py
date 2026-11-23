"""Shared-negative sampled softmax adapted from modded-nanoGPT PR #360.

The training-only approximation replaces the full vocabulary normalizer with a
candidate set shared by every token in a microbatch.  The set contains every
next-token target observed in that microbatch plus deterministic negatives from
a stride permutation of the vocabulary.  Validation always uses full softmax.

The candidate counts and coarse schedule mirror record #360:
  P=10240 -> P=14336 -> P=24576 -> full vocabulary.
Because this assignment has a smaller 32,768-token vocabulary and supports
arbitrary microbatch sizes, ``CandidateBuilder`` automatically escalates to the
next supported P (or full softmax) if a microbatch contains too many distinct
targets for the requested candidate count.

This is intentionally a biased training objective, just as in PR #360: omitted
classes contribute no mass to the sampled normalizer.  The final training tail
uses the exact full-vocabulary objective to remove the approximation before the
final checkpoint.

Provenance:
https://github.com/KellerJordan/modded-nanogpt/pull/360
https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/sampled_softmax.py
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# Same deterministic stride as record #360.  It is coprime with 32,768, so
# k * NEGATIVE_STRIDE mod V visits every vocabulary row exactly once per cycle.
NEGATIVE_STRIDE = 20_011

PR360_CANDIDATE_COUNTS = (10_240, 14_336, 24_576)

# The upstream PR #360 benchmark changed widths at approximately these fractions of its main
# training schedule (681/1194, 965/1194, 1107/1194).  We scale those transition
# points to the configured training budget; full_start_step can end sampling earlier.
PR360_FRACTIONS = (0.57, 0.81, 0.93)


def pr360_candidate_count(step: int, total_steps: int, vocab_size: int,
                          *, full_start_step: int | None = None) -> int:
    """Return the requested candidate width for a training optimizer step.

    The schedule mirrors PR #360 proportionally.  A count >= vocab size is
    treated as full softmax by callers.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    if full_start_step is not None:
        if not 0 <= full_start_step <= total_steps:
            raise ValueError("full_start_step must be between 0 and total_steps")
        if step >= full_start_step:
            return 0

    progress = min(1.0, step / total_steps)
    if progress < PR360_FRACTIONS[0]:
        p = PR360_CANDIDATE_COUNTS[0]
    elif progress < PR360_FRACTIONS[1]:
        p = PR360_CANDIDATE_COUNTS[1]
    elif progress < PR360_FRACTIONS[2]:
        p = PR360_CANDIDATE_COUNTS[2]
    else:
        return 0
    return p if p < vocab_size else 0


def schedule_transitions(total_steps: int, vocab_size: int,
                         *, full_start_step: int | None = None) -> list[tuple[int, int]]:
    """Return ``[(start_step, P), ...]``; P=0 denotes full softmax."""
    # Validate the override and keep reporting identical to runtime selection.
    pr360_candidate_count(0, total_steps, vocab_size, full_start_step=full_start_step)
    full_start = int(math.ceil(total_steps * PR360_FRACTIONS[-1]))
    if full_start_step is not None:
        full_start = min(full_start, full_start_step)
    starts = sorted({0, full_start} | {
        int(math.ceil(total_steps * f)) for f in PR360_FRACTIONS
        if int(math.ceil(total_steps * f)) < full_start
    })
    out: list[tuple[int, int]] = []
    for s in starts:
        p = 0 if s == full_start else pr360_candidate_count(
            s, total_steps, vocab_size, full_start_step=full_start_step
        )
        if not out or out[-1][1] != p:
            out.append((s, p))
    return out


class CandidateBuilder:
    """CPU candidate builder for shared-negative sampled softmax.

    Candidate construction follows PR #360:
      1. include every class appearing as a target in the microbatch;
      2. fill to P with distinct negatives from a deterministic stride sweep;
      3. sort candidate ids and remap each target to its position in that set.

    The mutable stride offset is checkpointed so a resumed run consumes exactly
    the same negative sequence as an uninterrupted run.
    """

    def __init__(
        self,
        vocab_size: int,
        candidate_counts: tuple[int, ...] = PR360_CANDIDATE_COUNTS,
        negative_stride: int = NEGATIVE_STRIDE,
    ):
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if math.gcd(negative_stride, vocab_size) != 1:
            raise ValueError(
                f"negative stride {negative_stride} must be coprime with vocab_size={vocab_size}"
            )

        self.vocab_size = int(vocab_size)
        self.negative_stride = int(negative_stride)
        self.candidate_counts = tuple(
            sorted({int(p) for p in candidate_counts if 0 < int(p) < vocab_size})
        )

        self.mark = np.zeros(self.vocab_size, dtype=bool)
        self.pos = np.empty(self.vocab_size, dtype=np.int64)
        sweep = (
            self.negative_stride * np.arange(self.vocab_size, dtype=np.int64)
        ) % self.vocab_size
        # Doubling means a wrapped window is always one contiguous slice.
        self.sweep = np.concatenate((sweep, sweep))
        self.sweep_offset = 0

    def state_dict(self) -> dict[str, int]:
        return {"sweep_offset": int(self.sweep_offset)}

    def load_state_dict(self, state: dict) -> None:
        self.sweep_offset = int(state.get("sweep_offset", 0)) % self.vocab_size

    def _resolve_count(self, requested: int, distinct_targets: int) -> int:
        if requested <= 0 or requested >= self.vocab_size:
            return self.vocab_size
        minimum = max(int(requested), int(distinct_targets))
        for p in self.candidate_counts:
            if p >= minimum:
                return p
        return self.vocab_size

    def _negatives(self, need: int) -> np.ndarray:
        """Return ``need`` distinct, currently unmarked vocabulary ids."""
        if need <= 0:
            return np.empty(0, dtype=np.int64)

        v = self.vocab_size
        # Same overdraw heuristic as PR #360; collisions with targets are filtered.
        draw = min(v, int(need * 1.7) + 256)
        idx = self.sweep[self.sweep_offset : self.sweep_offset + draw]
        self.sweep_offset = (self.sweep_offset + draw) % v
        idx = idx[~self.mark[idx]]
        if idx.size >= need:
            return idx[:need]

        # Very target-dense microbatches can exhaust the first stride window.
        # Deterministically fill from the remaining vocabulary.
        free = np.flatnonzero(~self.mark)
        if free.size < need:
            raise RuntimeError(f"only {free.size} free classes for {need} negatives")
        if free.size == need:
            return free
        # Spread fallback picks over the remaining ids rather than taking one block.
        take = np.linspace(0, free.size - 1, num=need, dtype=np.int64)
        return free[take]

    def build(
        self,
        requested_count: int,
        targets: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Build candidates and remapped labels for one microbatch.

        Returns ``(None, None, vocab_size)`` when full softmax should be used.
        Otherwise returns ``candidate_ids[P]``, ``target_positions[N]``, and P.
        """
        flat = np.asarray(targets, dtype=np.int64).reshape(-1)
        if flat.size == 0:
            raise ValueError("targets must be non-empty")
        if flat.min() < 0 or flat.max() >= self.vocab_size:
            raise ValueError("target id outside vocabulary")

        self.mark.fill(False)
        self.mark[flat] = True
        distinct = int(np.count_nonzero(self.mark))
        p = self._resolve_count(requested_count, distinct)
        if p >= self.vocab_size:
            return None, None, self.vocab_size

        if distinct < p:
            self.mark[self._negatives(p - distinct)] = True

        candidates = np.flatnonzero(self.mark)
        if candidates.size != p:
            raise RuntimeError(f"candidate builder produced {candidates.size} ids, expected {p}")

        self.pos.fill(-1)
        self.pos[candidates] = np.arange(p, dtype=np.int64)
        target_pos = self.pos[flat]
        if np.any(target_pos < 0):
            raise RuntimeError("a target was omitted from the sampled-softmax candidate set")

        # Own the arrays: callers can start constructing the next microbatch as soon
        # as these are handed to PyTorch without aliasing builder scratch state.
        return candidates.astype(np.int64, copy=False), target_pos.copy(), p
