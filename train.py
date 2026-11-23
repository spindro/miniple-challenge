from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model import MiniPLE, MiniPLEConfig
from muon import MuonAdamW
from sampled_softmax import CandidateBuilder, pr360_candidate_count, schedule_transitions
from data_progress import read_progress, wait_for_tokens
from validation_reference import (
    VALIDATION_TOKENS, verify_tokenizer_reference, verify_validation_reference,
)


class MemmapTokenLoader:
    """Sequential low-overhead loader for packed uint16 token streams."""

    def __init__(
        self,
        path: Path,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seed: int,
        progressive: bool = False,
    ):
        self.path = path
        self.progressive = progressive
        ready = int(read_progress(path)["tokens"]) if progressive else path.stat().st_size // 2
        self.tokens = (np.memmap(path, dtype=np.uint16, mode="r", shape=(ready,))
                       if ready else np.empty(0, dtype=np.uint16))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.need = batch_size * seq_len + 1
        if not progressive and len(self.tokens) < self.need:
            raise ValueError(
                f"{path} has only {len(self.tokens):,} tokens; need >= {self.need:,}"
            )
        self.pos = 0

    def state_dict(self) -> dict[str, Any]:
        return {"pos": self.pos, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.pos = int(state.get("pos", 0))
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]

    def reset(self, pos: int = 0) -> None:
        self.pos = int(pos)

    def _wrap_if_needed(self) -> None:
        if self.pos + self.need > len(self.tokens):
            max_start = len(self.tokens) - self.need + 1
            self.pos = int(self.rng.integers(0, max_start))

    def next_batch(self, return_cpu_targets: bool = False, batch_size: int | None = None):
        batch_size = self.batch_size if batch_size is None else batch_size
        if not 0 < batch_size <= self.batch_size:
            raise ValueError("batch_size must be between 1 and the loader's configured batch size")
        need = batch_size * self.seq_len + 1
        if self.pos + need > len(self.tokens):
            if self.progressive:
                ready = wait_for_tokens(self.path, self.pos + need)
                self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r", shape=(ready,))
            else:
                self._wrap_if_needed()

        # One contiguous read. Rows overlap by one boundary token so every target
        # is the true next token in the packed stream.
        chunk = np.array(
            self.tokens[self.pos : self.pos + need],
            dtype=np.int64,
            copy=True,
        )
        self.pos += batch_size * self.seq_len

        chunk_cpu = torch.from_numpy(chunk)
        if self.device.type == "cuda":
            chunk_cpu = chunk_cpu.pin_memory()
        chunk_device = chunk_cpu.to(self.device, non_blocking=True)
        x = chunk_device[:-1].view(batch_size, self.seq_len)
        y = chunk_device[1:].view(batch_size, self.seq_len)
        if return_cpu_targets:
            return x, y, chunk[1:]
        return x, y


class DecayMixtureLoader:
    """
    Deterministic two-stage L2 / QA / Multi-Style microbatch mixture.
    """

    stable_pattern = ("l2", "l2", "l2", "qa", "l2", "l2", "l2", "style")
    cooldown_pattern = ("l2", "l2", "qa", "style")

    def __init__(self, l2: MemmapTokenLoader, qa: MemmapTokenLoader, style: MemmapTokenLoader):
        self.loaders = {"l2": l2, "qa": qa, "style": style}
        self.pattern = self.stable_pattern
        self.index = 0

    def set_pattern(self, pattern: tuple[str, ...]) -> None:
        if self.pattern != pattern:
            self.pattern = pattern
            self.index = 0

    def next_batch(self, return_cpu_targets: bool = False):
        name = self.pattern[self.index % len(self.pattern)]
        self.index += 1
        return self.loaders[name].next_batch(return_cpu_targets=return_cpu_targets)

    def state_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "pattern": self.pattern,
            "loaders": {name: loader.state_dict() for name, loader in self.loaders.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.index = int(state.get("index", 0))
        self.pattern = tuple(state.get("pattern", self.stable_pattern))
        if self.pattern not in (self.stable_pattern, self.cooldown_pattern):
            raise ValueError("Checkpoint has an unsupported data-mixture pattern")
        for name, loader_state in state.get("loaders", {}).items():
            if name in self.loaders:
                self.loaders[name].load_state_dict(loader_state)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train MiniPLE PLE64 (146.5M with learned sinks) with Ultra-FineWeb L2 -> L3 decay curriculum"
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/baseline_3b"))
    p.add_argument("--validate-data-only", action="store_true",
                   help="Validate prepared data on CPU and exit without initializing training")

    p.add_argument("--stage1-tokens", type=int, default=2_100_000_000)
    p.add_argument("--decay-tokens", type=int, default=900_000_000)
    p.add_argument("--mix-start-tokens", type=int, default=1_500_000_000,
                   help="Switch from L2-only to the 75/12.5/12.5 mixture after this many tokens")
    p.add_argument("--micro-batch", type=int, default=20)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)

    # Muon hidden matrices and auxiliary AdamW use independent peak rates.
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--muon-lr", type=float, default=8e-4,
                   help="Muon peak LR with Moonlight scaling; --lr controls auxiliary AdamW")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument(
        "--warmup-tokens",
        type=int,
        default=10_000_000,
        help="Warm up linearly for this many predicted tokens before holding the peak LR stable",
    )
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument(
        "--eval-sequences",
        type=int,
        default=128,
        help="Number of fixed-length validation sequences per evaluation; independent of training micro-batch",
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Overwrite latest.pt every N optimizer steps (0 disables periodic recovery checkpoints)",
    )
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    p.add_argument(
        "--loss-backend",
        choices=["auto", "torch", "liger", "chunked"],
        default="auto",
    )
    p.add_argument(
        "--sampled-softmax",
        choices=["pr360", "off"],
        default="pr360",
        help="Training-only shared-negative sampled softmax; validation always uses full vocabulary",
    )
    p.add_argument(
        "--full-softmax-at-decay", action=argparse.BooleanOptionalAction, default=True,
        help="End sampled softmax when WSD cooldown starts",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        choices=["default", "max-autotune", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
    )
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--ce-chunk-size", type=int, default=2048)
    p.add_argument("--device", default="auto", help="auto, cuda, mps, cpu, cuda:0, ...")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def choose_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(spec: str, device: torch.device):
    if spec == "bf16":
        return torch.bfloat16
    if spec == "fp16":
        return torch.float16
    if spec == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "mps"} and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def make_optimizer(model: nn.Module, args, device: torch.device):
    """Muon for hidden matrices; fused auxiliary AdamW for embeddings/norms."""
    opt = MuonAdamW(
        model, lr=args.lr, muon_lr=args.muon_lr,
        weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        momentum=args.muon_momentum, ns_steps=args.muon_ns_steps,
        compile_ns=args.compile and device.type == "cuda",
    )
    for group in opt.param_groups:
        print(f"optimizer {group['kind']}: {sum(p.numel() for p in group['params']):,} "
              f"parameters; peak_lr={group['initial_lr']}; decay={group['weight_decay']}")
    return opt


def wsd_lr(
    step: int,
    total_steps: int,
    stage1_steps: int,
    base_lr: float,
    warmup_steps: int,
) -> float:
    """Warmup -> Stable -> linear Decay to zero."""
    warmup_steps = max(1, int(warmup_steps))

    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    if step < stage1_steps:
        return base_lr

    decay_steps = max(1, total_steps - stage1_steps)
    decay_index = step - stage1_steps
    # First decay step is close to base_lr; final step reaches zero.
    if decay_steps == 1:
        return 0.0
    frac = min(1.0, decay_index / (decay_steps - 1))
    return base_lr * (1.0 - frac)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    step: int,
    loader_states: dict[str, Any],
    args,
    schedule: dict[str, Any],
    scaler,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loader_states": loader_states,
        "args": vars(args),
        "schedule": schedule,
        "model_config": asdict(model.cfg),
        "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    if next(model.parameters()).device.type == "mps":
        payload["mps_rng"] = torch.mps.get_rng_state()

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def compile_training_model(raw_model: MiniPLE, mode: str) -> MiniPLE:
    return torch.compile(raw_model, mode=mode, dynamic=False)


@torch.no_grad()
def evaluate(model, loader: MemmapTokenLoader, sequences: int, device, dtype):
    """Evaluate a fixed number of T-token sequences independent of micro-batch size."""
    if sequences <= 0:
        raise ValueError("--eval-sequences must be positive")
    if sequences * loader.seq_len + 1 > len(loader.tokens):
        raise ValueError("Validation file is too short for --eval-sequences without repeating data")
    was_training = model.training
    model.eval()
    old_state = loader.state_dict()
    loader.reset(0)
    weighted_loss = 0.0
    seen = 0
    remaining = sequences
    try:
        while remaining > 0:
            take = min(remaining, loader.batch_size)
            x, y = loader.next_batch(batch_size=take)
            with autocast_context(device, dtype):
                loss = model(x, y)
            weighted_loss += float(loss.detach()) * take
            seen += take
            remaining -= take
    finally:
        loader.load_state_dict(old_state)
        model.train(was_training)
    return weighted_loss / seen


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_data.py first.")
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_prepared_data(data_dir: Path, meta: dict[str, Any]) -> None:
    if int(meta.get("format_version", -1)) != 2:
        raise ValueError(
            "Prepared data format is outdated or unknown. Re-run this repo's "
            "prepare_data.py so validation/train/decay slices use the leakage-safe packer."
        )
    packing = meta.get("packing", {})
    if packing.get("split_documents_across_output_files") is not False:
        raise ValueError(
            "Prepared data does not declare document-disjoint output boundaries. "
            "Re-run prepare_data.py with the current repo."
        )
    if int(meta.get("vocab_size", -1)) != 32_768:
        raise ValueError(
            f"This assignment is fixed to vocab_size=32,768; metadata has {meta.get('vocab_size')}"
        )
    tokenizer_meta = meta.get("tokenizer", {})
    tok_name = tokenizer_meta.get("local_file", "tokenizer.json")
    tok_path = data_dir / tok_name
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing frozen tokenizer file: {tok_path}")
    verify_tokenizer_reference(tok_path)
    expected_hash = tokenizer_meta.get("sha256")
    if expected_hash:
        actual_hash = sha256_file(tok_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Tokenizer SHA256 mismatch for {tok_path}: expected {expected_hash}, got {actual_hash}"
            )

    files = meta.get("files", {})
    if files.get("val_l2.bin") != VALIDATION_TOKENS:
        raise ValueError(
            f"Prepared data must declare the frozen validation stream with {VALIDATION_TOKENS:,} tokens"
        )
    for name, expected_tokens in files.items():
        path = data_dir / name
        if meta.get("progressive_preparation") and name == "val_l2.bin":
            wait_for_tokens(path, int(expected_tokens))
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared token file: {path}")
        if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
            raise ValueError(f"Token file has invalid uint16 byte length: {path}")
        actual_tokens = path.stat().st_size // np.dtype(np.uint16).itemsize
        if meta.get("progressive_preparation"):
            progress = read_progress(path)
            actual_tokens = path.stat().st_size // np.dtype(np.uint16).itemsize
            if (progress.get("error") or int(progress["target_tokens"]) != int(expected_tokens)
                    or not 0 <= int(progress["tokens"]) <= actual_tokens <= int(expected_tokens)):
                raise ValueError(f"Invalid data preparation state for {path}: {progress}")
        elif actual_tokens != int(expected_tokens):
            raise ValueError(
                f"Token count mismatch for {path.name}: metadata={int(expected_tokens):,}, "
                f"file={actual_tokens:,}"
            )
    verify_validation_reference(data_dir / "val_l2.bin", expected_tokens=VALIDATION_TOKENS)


def all_loader_states(stage1, decay_l3, decay_l2, val, sampled_builder=None):
    state = {
        "stage1": stage1.state_dict(),
        "decay_l3": decay_l3.state_dict(),
        "decay_l2": decay_l2.state_dict(),
        "val": val.state_dict(),
    }
    if sampled_builder is not None:
        state["sampled_softmax"] = sampled_builder.state_dict()
    return state


def restore_loader_states(state, stage1, decay_l3, decay_l2, val, sampled_builder=None):
    if not state:
        return
    if "stage1" in state:
        stage1.load_state_dict(state["stage1"])
    if "decay_l3" in state:
        decay_l3.load_state_dict(state["decay_l3"])
    if "decay_l2" in state:
        decay_l2.load_state_dict(state["decay_l2"])
    if "val" in state:
        val.load_state_dict(state["val"])
    if sampled_builder is not None and "sampled_softmax" in state:
        sampled_builder.load_state_dict(state["sampled_softmax"])


def log_metrics(out_dir: Path, event: str, **values) -> None:
    """Append one durable, machine-readable record at existing logging points."""
    record = {"time_unix": time.time(), "event": event, **values}
    with (out_dir / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    args = parse_args()
    meta = load_metadata(args.data_dir)
    validate_prepared_data(args.data_dir, meta)
    if args.validate_data_only:
        print(f"Prepared data and frozen validation/tokenizer verified: {args.data_dir}")
        return
    for name in ("micro_batch", "grad_accum", "seq_len", "eval_sequences",
                 "log_interval", "ce_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if args.seq_len > MiniPLEConfig.max_seq_len:
        raise ValueError(f"MiniPLE PLE64 supports context <= {MiniPLEConfig.max_seq_len}")

    data_fingerprint = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    vocab_size = int(meta["vocab_size"])
    stage1_tokens = int(meta["stage1_tokens"] if args.stage1_tokens is None else args.stage1_tokens)
    decay_tokens = int(meta["decay_tokens"] if args.decay_tokens is None else args.decay_tokens)

    prepared_files = meta.get("files", {})
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("optimizer", {}).get("muon_version") != 1:
            raise ValueError("Resume requires a MuonAdamW v1 checkpoint")
        cfg = MiniPLEConfig.from_checkpoint(ckpt["model_config"])
        if cfg.vocab_size != vocab_size or cfg.max_seq_len != args.seq_len:
            raise ValueError("Resume vocabulary/context mismatch; keep data and --seq-len unchanged")
    else:
        cfg = MiniPLEConfig(vocab_size=vocab_size, max_seq_len=args.seq_len)
    cfg.attn_backend = args.attn
    cfg.loss_backend = args.loss_backend
    cfg.gradient_checkpointing = args.grad_checkpointing
    cfg.ce_chunk_size = args.ce_chunk_size
    raw_model = MiniPLE(cfg).to(device)
    print(f"device: {device}; autocast dtype: {dtype}")
    print(
        f"trainable parameters: {raw_model.num_parameters():,} "
        f"({raw_model.num_parameters()/1e6:.3f}M)"
    )
    print(f"position encoding: local=RoPE; global={'NoPE' if cfg.global_nope else 'RoPE'}")
    print(f"attention layout: {cfg.attention_pattern}")
    print(f"attention windows: local={cfg.local_window}; global/context={cfg.max_seq_len}")
    print(f"learned attention sinks: {cfg.n_layers * cfg.n_heads} (one logit per head)")

    optimizer = make_optimizer(raw_model, args, device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=(device.type in {"cuda", "mps"} and dtype == torch.float16)
    )

    def loader(filename: str, seed_offset: int):
        return MemmapTokenLoader(
            args.data_dir / filename,
            args.micro_batch,
            args.seq_len,
            device,
            args.seed + seed_offset,
            progressive=bool(meta.get("progressive_preparation")),
        )

    stage1_loader = loader("stage1_l2.bin", 10)
    l2_decay_source = loader("decay_l2.bin", 20)
    qa_decay_source = loader("decay_l3_qa.bin", 30)
    style_decay_source = loader("decay_l3_style.bin", 40)
    val_loader = loader("val_l2.bin", 50)
    if args.eval_sequences * args.seq_len + 1 > len(val_loader.tokens):
        raise ValueError("Validation file is too short for --eval-sequences without repeating data")

    decay_l3_loader = DecayMixtureLoader(
        l2_decay_source,
        qa_decay_source,
        style_decay_source,
    )
    decay_l2_loader = l2_decay_source
    sampled_builder = CandidateBuilder(vocab_size) if args.sampled_softmax == "pr360" else None

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    if stage1_tokens < tokens_per_step or decay_tokens < tokens_per_step:
        raise ValueError(
            "Each stage token budget must be at least one optimizer update "
            f"({tokens_per_step:,} tokens with the current batch geometry)."
        )
    # Floor each stage independently so no source file is ever over-consumed.
    # The short remainder (< one optimizer update) is intentionally unused.
    stage1_steps = stage1_tokens // tokens_per_step
    decay_steps = decay_tokens // tokens_per_step
    total_steps = stage1_steps + decay_steps
    mix_start_steps = args.mix_start_tokens // tokens_per_step
    if not 0 < mix_start_steps <= stage1_steps:
        raise ValueError("--mix-start-tokens must fall at or before the learning-rate cooldown")
    stable_mix_steps = stage1_steps - mix_start_steps
    if (
        stable_mix_steps * args.grad_accum % len(DecayMixtureLoader.stable_pattern)
        or decay_steps * args.grad_accum % len(DecayMixtureLoader.cooldown_pattern)
    ):
        raise ValueError("Token budgets and accumulation must complete both data-mixture cycles")
    stage1_scheduled_tokens = stage1_steps * tokens_per_step
    decay_scheduled_tokens = decay_steps * tokens_per_step
    total_scheduled_tokens = stage1_scheduled_tokens + decay_scheduled_tokens
    mix_start_scheduled_tokens = mix_start_steps * tokens_per_step
    if ("mix_start_tokens" in meta
            and int(meta["mix_start_tokens"]) // tokens_per_step != mix_start_steps):
        raise ValueError("Prepared data mix boundary does not match --mix-start-tokens")
    required_tokens = {"stage1_l2.bin": mix_start_scheduled_tokens}
    curriculum = meta.get("curriculum", {})
    expected_stable = {"Ultra-FineWeb L2": 0.75,
                       "Ultra-FineWeb-L3 English QA Synthetic": 0.125,
                       "Ultra-FineWeb-L3 English Multi-Style Synthetic": 0.125}
    expected_cooldown = {"Ultra-FineWeb L2": 0.5,
                         "Ultra-FineWeb-L3 English QA Synthetic": 0.25,
                         "Ultra-FineWeb-L3 English Multi-Style Synthetic": 0.25}
    if (curriculum.get("stable_l3_mix") != expected_stable
            or curriculum.get("decay_l3_mix") != expected_cooldown):
        raise ValueError("Prepared data metadata must declare the 75/12.5/12.5 stable and 50/25/25 cooldown mixtures")
    source_microbatches = {"l2": 0, "qa": 0, "style": 0}
    for steps, pattern in ((stable_mix_steps, DecayMixtureLoader.stable_pattern),
                           (decay_steps, DecayMixtureLoader.cooldown_pattern)):
        cycles, remainder = divmod(steps * args.grad_accum, len(pattern))
        for source in source_microbatches:
            source_microbatches[source] += cycles * pattern.count(source)
            source_microbatches[source] += pattern[:remainder].count(source)
    for source, filename in (("l2", "decay_l2.bin"),
                             ("qa", "decay_l3_qa.bin"),
                             ("style", "decay_l3_style.bin")):
        required_tokens[filename] = source_microbatches[source] * args.micro_batch * args.seq_len
    for name, count in required_tokens.items():
        if int(prepared_files.get(name, 0)) < count + 1:
            raise ValueError(f"{name} needs {count + 1:,} tokens including the final next-token label; re-prepare data")
    warmup_steps = max(1, math.ceil(args.warmup_tokens / tokens_per_step))
    warmup_scheduled_tokens = warmup_steps * tokens_per_step
    if warmup_steps >= stage1_steps:
        raise ValueError(
            f"Warmup ({warmup_steps:,} steps / {warmup_scheduled_tokens:,} tokens) must finish before "
            f"the stable stage ends ({stage1_steps:,} steps). Reduce --warmup-tokens."
        )

    schedule = {
        "data_fingerprint": data_fingerprint,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "tokens_per_step": tokens_per_step,
        "stage1_steps": stage1_steps,
        "decay_steps": decay_steps,
        "mix_start_steps": mix_start_steps,
        "mix_start_scheduled_tokens": mix_start_scheduled_tokens,
        "total_steps": total_steps,
        "stage1_scheduled_tokens": stage1_scheduled_tokens,
        "decay_scheduled_tokens": decay_scheduled_tokens,
        "total_scheduled_tokens": total_scheduled_tokens,
        "warmup_tokens_requested": args.warmup_tokens,
        "warmup_steps": warmup_steps,
        "warmup_scheduled_tokens": warmup_scheduled_tokens,
        "lr": args.lr,
        "sampled_softmax": args.sampled_softmax,
        "full_softmax_at_decay": args.full_softmax_at_decay,
        "data_mix_patterns": {
            "stable": DecayMixtureLoader.stable_pattern,
            "cooldown": DecayMixtureLoader.cooldown_pattern,
        },
        "decay_data": "l3_mix",
        "decay_required_tokens": required_tokens,
        "dtype": str(dtype),
    }

    print(f"effective tokens/update: {tokens_per_step:,}")
    print(
        f"stable LR: {stage1_steps:,} steps ({stage1_scheduled_tokens:,} predicted tokens; "
        f"requested {stage1_tokens:,})"
    )
    print(f"data mix starts: step {mix_start_steps:,} after {mix_start_scheduled_tokens:,} tokens")
    print(
        f"decay:   {decay_steps:,} steps ({decay_scheduled_tokens:,} predicted tokens; "
        f"requested {decay_tokens:,})"
    )
    print(f"total:   {total_steps:,} steps ({total_scheduled_tokens:,} predicted tokens)")
    print(
        f"warmup:  {warmup_steps:,} steps ({warmup_scheduled_tokens:,} predicted tokens; "
        f"requested {args.warmup_tokens:,}) -> peak LR {args.lr:.2e}"
    )
    print(f"uniform-prediction NLL baseline: log(V)={math.log(vocab_size):.4f}")
    if sampled_builder is not None:
        transitions = schedule_transitions(
            total_steps, vocab_size,
            full_start_step=stage1_steps if args.full_softmax_at_decay else None,
        )
        summary = ", ".join(
            f"step {start:,}: {'full' if p == 0 else f'P={p:,}'}"
            for start, p in transitions
        )
        print(f"sampled softmax (PR #360 schedule): {summary}")
        print("validation loss remains full-vocabulary cross entropy")
    else:
        print("sampled softmax: off (full vocabulary for every training step)")

    start_step = 0
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for key in ("weight_decay", "beta1", "beta2", "grad_clip"):
            previous = ckpt.get("args", {}).get(key, getattr(args, key))
            if previous != getattr(args, key):
                raise ValueError(f"Resume optimizer mismatch for {key}: {previous} vs {getattr(args, key)}")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        ckpt_schedule = ckpt.get("schedule")
        if ckpt_schedule is not None:
            if ckpt_schedule.get("dtype", str(dtype)) != str(dtype):
                raise ValueError("Resume dtype mismatch; keep --dtype unchanged")
            if ckpt_schedule.get("data_fingerprint") not in {None, data_fingerprint}:
                raise ValueError(
                    "Resume data fingerprint mismatch. Use the exact same prepared "
                    "data/tokenizer metadata when resuming or branching."
                )
            if ckpt_schedule.get("sampled_softmax", "off") != args.sampled_softmax:
                raise ValueError("Resume sampled-softmax mode mismatch; keep the loss schedule unchanged")
            if (sampled_builder is not None
                    and ckpt_schedule.get("full_softmax_at_decay", True) != args.full_softmax_at_decay):
                raise ValueError("Resume full-softmax-at-decay mismatch; keep the loss schedule unchanged")
            if (ckpt_schedule.get("data_mix_patterns") != schedule["data_mix_patterns"]
                    and int(ckpt["step"]) + 1 > mix_start_steps):
                raise ValueError("Cannot change the data-mixture schedule after the mix has started")
            for key in (
                "micro_batch",
                "grad_accum",
                "seq_len",
                "tokens_per_step",
                "stage1_steps",
                "decay_steps",
                "mix_start_steps",
                "total_steps",
                "warmup_tokens_requested",
                "warmup_steps",
                "warmup_scheduled_tokens",
            ):
                previous = ckpt_schedule.get(
                    key, ckpt_schedule.get("stage1_steps", -1) if key == "mix_start_steps" else -1
                )
                if int(previous) != int(schedule[key]):
                    raise ValueError(
                        f"Resume schedule mismatch for {key}: checkpoint={ckpt_schedule.get(key)}, "
                        f"current={schedule[key]}. Keep micro-batch, grad-accum, sequence length, "
                        "token budgets, and warmup unchanged when resuming/branching."
                    )
            ckpt_lr = float(ckpt_schedule.get("lr", float("nan")))
            if not math.isclose(ckpt_lr, float(schedule["lr"]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Resume peak-LR mismatch: checkpoint={ckpt_lr}, current={schedule['lr']}. "
                    "Keep --lr unchanged when resuming/branching."
                )
        start_step = int(ckpt["step"]) + 1
        restore_loader_states(
            ckpt.get("loader_states", {}),
            stage1_loader,
            decay_l3_loader,
            decay_l2_loader,
            val_loader,
            sampled_builder,
        )
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        if "python_rng" in ckpt:
            random.setstate(ckpt["python_rng"])
        if device.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if device.type == "mps" and "mps_rng" in ckpt:
            torch.mps.set_rng_state(ckpt["mps_rng"])
        del ckpt
        print(f"resumed from {args.resume} at optimizer step {start_step}")

    model = raw_model
    if args.compile:
        print(f"compiling model with torch.compile(mode={args.compile_mode!r}) ...")
        model = compile_training_model(raw_model, args.compile_mode)

    if start_step > total_steps:
        raise ValueError("Checkpoint is already beyond the final update")

    raw_model.train()
    optimizer.zero_grad(set_to_none=True)
    # Keep loss accumulation on-device; calling .item() per microbatch would
    # force a CUDA synchronization and materially reduce single-GPU throughput.
    recent_loss = torch.zeros((), device=device, dtype=torch.float32)
    recent_steps = 0
    window_tokens = 0
    window_start = time.perf_counter()
    log_metrics(args.out_dir, "start", step=start_step, total_steps=total_steps,
                resume=str(args.resume) if args.resume else None,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                model_config=asdict(cfg), schedule=schedule)

    for step in range(start_step, total_steps):
        lr_stage = "stable" if step < stage1_steps else "decay"
        if step < mix_start_steps:
            data_stage = "L2"
        elif step < stage1_steps:
            data_stage = "l3_mix_75"
        else:
            data_stage = "l3_mix_50"
        stage = f"{lr_stage}-{data_stage}"
        if step >= mix_start_steps:
            pattern = (DecayMixtureLoader.stable_pattern if step < stage1_steps
                       else DecayMixtureLoader.cooldown_pattern)
            decay_l3_loader.set_pattern(pattern)
        active_loader = stage1_loader if step < mix_start_steps else decay_l3_loader

        lr_factor = wsd_lr(
            step,
            total_steps,
            stage1_steps,
            1.0,
            warmup_steps,
        )
        lr = args.lr * lr_factor
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", args.lr) * lr_factor
        optimizer.record_diagnostics = (step + 1) % args.log_interval == 0 or step == start_step

        loss_accum = torch.zeros((), device=device, dtype=torch.float32)
        requested_p = (
            pr360_candidate_count(
                step, total_steps, vocab_size,
                full_start_step=stage1_steps if args.full_softmax_at_decay else None,
            )
            if sampled_builder is not None else 0
        )
        actual_candidate_counts = []
        for _ in range(args.grad_accum):
            if requested_p:
                x, y, cpu_targets = active_loader.next_batch(return_cpu_targets=True)
                candidates_np, target_pos_np, actual_p = sampled_builder.build(
                    requested_p, cpu_targets
                )
                actual_candidate_counts.append(actual_p)
                if candidates_np is not None:
                    candidates = torch.from_numpy(candidates_np)
                    target_pos = torch.from_numpy(target_pos_np)
                    if device.type == "cuda":
                        candidates = candidates.pin_memory()
                        target_pos = target_pos.pin_memory()
                    candidates = candidates.to(device, non_blocking=(device.type == "cuda"))
                    target_pos = target_pos.to(device, non_blocking=(device.type == "cuda"))
                    with autocast_context(device, dtype):
                        loss = model(
                            x, y, sampled_candidates=candidates, sampled_targets=target_pos
                        ) / args.grad_accum
                else:
                    with autocast_context(device, dtype):
                        loss = model(x, y) / args.grad_accum
            else:
                x, y = active_loader.next_batch()
                actual_candidate_counts.append(vocab_size)
                with autocast_context(device, dtype):
                    loss = model(x, y) / args.grad_accum
            loss_accum.add_(loss.detach().float())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), args.grad_clip, foreach=(device.type == "cuda"),
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        recent_loss.add_(loss_accum)
        recent_steps += 1
        window_tokens += tokens_per_step

        if (step + 1) % args.log_interval == 0 or step == start_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            tok_s = window_tokens / max(elapsed, 1e-9)
            avg_loss = float((recent_loss / max(1, recent_steps)).item())
            mem = ""
            if device.type == "cuda":
                mem = f" | peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
                torch.cuda.reset_peak_memory_stats()
            processed = (step + 1) * tokens_per_step
            p_min = min(actual_candidate_counts)
            p_max = max(actual_candidate_counts)
            softmax_status = (
                "full" if p_min >= vocab_size else
                f"P={p_min:,}" if p_min == p_max else f"P={p_min:,}-{p_max:,}"
            )
            print(
                f"step {step+1:6d}/{total_steps} | {stage:18s} | "
                f"tokens {processed/1e6:7.1f}M | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad {float(grad_norm):.2f} | "
                f"softmax {softmax_status} | {tok_s:,.0f} tok/s{mem}"
            )
            log_metrics(args.out_dir, "train", step=step + 1, tokens=processed,
                        stage=stage, loss=avg_loss, averaged_steps=recent_steps,
                        lr=lr, grad_norm=float(grad_norm), tokens_per_second=tok_s,
                        candidate_min=p_min, candidate_max=p_max,
                        **{"muon_lr": optimizer.param_groups[0]["lr"],
                           **optimizer.diagnostics()})
            recent_loss.zero_()
            recent_steps = 0
            window_tokens = 0
            window_start = time.perf_counter()

        if (args.eval_interval > 0 and (step + 1) % args.eval_interval == 0) or step + 1 == total_steps:
            # Avoid compiling a second train/eval graph and a partial-batch graph.
            val_loss = evaluate(raw_model, val_loader, args.eval_sequences, device, dtype)
            print(f"  validation L2 loss: {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}")
            log_metrics(args.out_dir, "validation", step=step + 1,
                        tokens=(step + 1) * tokens_per_step, loss=val_loss,
                        perplexity=math.exp(min(val_loss, 20)),
                        sequences=args.eval_sequences, full_vocabulary=True)

        if step + 1 == mix_start_steps and mix_start_steps < stage1_steps:
            branch_path = args.out_dir / "mix_start.pt"
            save_checkpoint(
                branch_path, raw_model, optimizer, step,
                all_loader_states(stage1_loader, decay_l3_loader, decay_l2_loader, val_loader, sampled_builder),
                args, schedule, scaler,
            )
            print(f"saved data-mixture branch checkpoint: {branch_path}")

        if step + 1 == stage1_steps:
            branch_path = args.out_dir / "stage1_end.pt"
            save_checkpoint(
                branch_path,
                raw_model,
                optimizer,
                step,
                all_loader_states(stage1_loader, decay_l3_loader, decay_l2_loader, val_loader, sampled_builder),
                args,
                schedule,
                scaler,
            )
            print(f"saved common stage-1 branch checkpoint: {branch_path}")

        if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
            latest = args.out_dir / "latest.pt"
            states = all_loader_states(
                stage1_loader, decay_l3_loader, decay_l2_loader, val_loader, sampled_builder
            )
            save_checkpoint(latest, raw_model, optimizer, step, states, args, schedule, scaler)

    states = all_loader_states(stage1_loader, decay_l3_loader, decay_l2_loader, val_loader, sampled_builder)
    final_path = args.out_dir / "final_l3_mix.pt"
    save_checkpoint(final_path, raw_model, optimizer, total_steps - 1, states, args, schedule, scaler)
    print(f"saved {final_path}")
    log_metrics(args.out_dir, "complete", step=total_steps,
                tokens=total_steps * tokens_per_step, checkpoint=str(final_path))


if __name__ == "__main__":
    main()
