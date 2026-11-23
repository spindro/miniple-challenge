from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from tqdm import tqdm
from data_progress import publish_tokens, read_progress
from validation_reference import verify_tokenizer_reference, verify_validation_reference

from tokenizer_utils import (
    BOS_TOKEN,
    EXPECTED_VOCAB_SIZE,
    NANOCHAT_TOKENIZER_REPO,
    NANOCHAT_TOKENIZER_REVISION,
    load_fixed_nanochat_tokenizer,
)

DATASETS_VERSION = "5.0.1"
PARQUET_BATCH_SIZE = 2_000
PARQUET_SCAN_OPTIONS = {"pre_buffer": False, "use_buffered_stream": True, "buffer_size": 65_536}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Prepare the MiniPLE training corpus: Ultra-FineWeb L2 for the stable "
            "phase and Ultra-FineWeb-L3 QA/Multi-Style for the decay phase."
        )
    )
    p.add_argument("--out-dir", type=Path, required=True)

    # Learning-rate cooldown starts at 70%; the data mixture starts at 50%.
    p.add_argument("--stage1-tokens", type=int, default=2_100_000_000)
    p.add_argument("--decay-tokens", type=int, default=900_000_000)
    p.add_argument("--mix-start-tokens", type=int, default=1_500_000_000)
    p.add_argument("--val-tokens", type=int, default=5_000_000)

    p.add_argument(
        "--tokenizer-json",
        type=Path,
        default=Path(__file__).resolve().parent / "baselines/validation/tokenizer.json",
        help=(
            "Frozen course tokenizer.json (defaults to the bundled reference)."
        ),
    )
    p.add_argument("--tokenizer-repo", default=NANOCHAT_TOKENIZER_REPO)
    p.add_argument("--tokenizer-revision", default=NANOCHAT_TOKENIZER_REVISION)
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--shuffle-buffer", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--progressive", action="store_true",
                   help="Publish durable token counts so training can start during preparation")
    p.add_argument("--validation-reference", type=Path,
                   default=Path(__file__).resolve().parent / "baselines/validation/val_l2.bin",
                   help="Frozen course validation; always checked, including standalone preparation")
    p.add_argument("--check-validation-only", action="store_true",
                   help="Reconstruct and check only the 5M-token validation stream; no training corpus")

    p.add_argument("--l2-dataset", default="openbmb/Ultra-FineWeb")
    p.add_argument("--l2-revision", default="02c85641e3d19a854be2e09139c25adaa9518063",
                   help="Pinned L2 dataset revision used by the course baseline")
    p.add_argument("--l2-split", default="en")
    p.add_argument("--l2-text-field", default="content")

    p.add_argument("--l3-dataset", default="openbmb/Ultra-FineWeb-L3")
    p.add_argument("--l3-revision", default="bc3b1ba986fcaef6871b9790a413b16267c2de0f",
                   help="Pinned L3 dataset revision used by the course baseline")
    p.add_argument(
        "--l3-qa-config",
        default="Ultra-FineWeb-L3-en-QA-Synthetic",
    )
    p.add_argument(
        "--l3-style-config",
        default="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
    )
    p.add_argument("--l3-text-field", default="content")

    return p.parse_args()


def _shuffle(ds, seed: int, buffer_size: int):
    if buffer_size <= 0:
        return ds
    return ds.shuffle(seed=seed, buffer_size=buffer_size, max_buffer_input_shards=10)


def _reader_kwargs(text_field: str):
    from datasets import Features, Value
    from pyarrow.dataset import ParquetFragmentScanOptions

    # Read text in bounded batches; fetching unused columns evicts the HTTP cache.
    return {
        "columns": [text_field],
        "features": Features({text_field: Value("string")}),
        "batch_size": PARQUET_BATCH_SIZE,
        "fragment_scan_options": ParquetFragmentScanOptions(**PARQUET_SCAN_OPTIONS),
    }


def l2_stream(args, seed: int, revision: str):
    from datasets import load_dataset

    ds = load_dataset(
        args.l2_dataset,
        split=args.l2_split,
        revision=revision,
        streaming=True,
        **_reader_kwargs(args.l2_text_field),
    )
    return _shuffle(ds, seed, args.shuffle_buffer)


def l3_stream(args, config: str, seed: int, revision: str):
    from datasets import load_dataset

    ds = load_dataset(
        args.l3_dataset,
        config,
        split="train",
        revision=revision,
        streaming=True,
        **_reader_kwargs(args.l3_text_field),
    )
    return _shuffle(ds, seed, args.shuffle_buffer)


def extract_text(row: dict, preferred: str) -> str:
    value = row.get(preferred)
    if isinstance(value, str) and value.strip():
        return value

    for key in ("content", "text", "synthetic", "output", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise KeyError(
        f"Could not find a text field. Available fields: {sorted(row.keys())}"
    )



def _batched_texts(
    ds,
    text_field: str,
    batch_size: int,
) -> Iterator[list[str]]:
    batch: list[str] = []
    for row in ds:
        try:
            text = extract_text(row, text_field)
        except KeyError:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def pack_stream_across_targets(
    ds,
    tok: Tokenizer,
    text_field: str,
    targets: Sequence[tuple[Path, int]],
    encode_batch_size: int,
    description: str,
    progressive: bool = False,
    validation_sha256: str | None = None,
    validation_reference: Path | None = None,
) -> dict[str, int]:
    """
    Consume one stream once and fill multiple token files sequentially.

    This is used for L2 so validation, stage-1, and decay-control data are
    disjoint slices of the same deterministic shuffled stream.
    """
    bos = tok.token_to_id(BOS_TOKEN)
    if bos is None:
        raise RuntimeError(f"Tokenizer has no required {BOS_TOKEN} token")

    total_target = sum(n for _, n in targets)
    reference = (np.fromfile(validation_reference, dtype=np.uint16)
                 if validation_reference is not None else None)
    pbar = tqdm(total=targets[0][1], unit="tok", unit_scale=True,
                desc=f"{description}: {targets[0][0].name}")

    files = [open(path, "wb") for path, _ in targets]
    written = [0 for _ in targets]
    published = [0 for _ in targets]
    current = 0

    def publish(index: int, complete: bool):
        files[index].flush()
        os.fsync(files[index].fileno())
        path, target = targets[index]
        if complete and path.name == "val_l2.bin" and validation_sha256:
            actual = sha256_file(path)
            if actual != validation_sha256:
                raise ValueError(
                    f"Generated validation SHA256 mismatch: expected {validation_sha256}, got {actual}; "
                    "training must not start. Use a fresh output directory after correcting preparation."
                )
            print(f"\nPASS: frozen validation SHA256 {actual}", flush=True)
        if progressive:
            publish_tokens(path, written[index], target, complete=complete)
        published[index] = written[index]

    try:
        for texts in _batched_texts(ds, text_field, encode_batch_size):
            encodings = tok.encode_batch(texts, add_special_tokens=False)
            for enc in encodings:
                # nanochat pretraining convention: every document begins with BOS.
                ids = [bos] + enc.ids
                if current >= len(targets):
                    break

                # Never carry the tail of one document into the next target file.
                # This guarantees that validation, stable training, and decay slices
                # cannot contain different fragments of the same source document.
                remaining = targets[current][1] - written[current]
                if remaining <= 0:
                    current += 1
                    continue
                n = min(remaining, len(ids))
                chunk = np.asarray(ids[:n], dtype=np.uint16)
                if reference is not None and targets[current][0].name == "val_l2.bin":
                    start = written[current]
                    expected = reference[start:start + n]
                    if len(expected) != n:
                        raise ValueError("Validation reference token count differs from the preparation target")
                    different = np.flatnonzero(chunk != expected)
                    if different.size:
                        offset = int(different[0])
                        raise ValueError(
                            f"Generated validation differs from --validation-reference at token "
                            f"{start + offset:,}: expected {int(expected[offset])}, got {int(chunk[offset])}. "
                            "Stopped at the first differing document, before preparing training data. "
                            "Check the printed package versions, seed and shuffle settings; "
                            "use a fresh output directory after correcting preparation. "
                            "Do not replace the frozen validation or disable this check."
                        )
                chunk.tofile(files[current])
                written[current] += n
                pbar.update(n)
                if written[current] >= targets[current][1]:
                    publish(current, True)
                    current += 1
                    # Any unconsumed tail of this document is intentionally dropped.
                    # The next file starts with the BOS of the next document.
                elif progressive and written[current] - published[current] >= 10_000_000:
                    publish(current, False)
                if current >= len(targets):
                    return {
                        str(targets[i][0].name): written[i]
                        for i in range(len(targets))
                    }
                if written[current] == 0:
                    pbar.reset(total=targets[current][1])
                    pbar.set_description(f"{description}: {targets[current][0].name}")
    finally:
        pbar.close()
        for f in files:
            f.close()

    raise RuntimeError(
        f"Dataset stream ended early. Wrote {sum(written):,}/{total_target:,} tokens."
    )


def pack_single(
    ds,
    tok: Tokenizer,
    text_field: str,
    path: Path,
    target_tokens: int,
    encode_batch_size: int,
    description: str,
    progressive: bool = False,
) -> int:
    result = pack_stream_across_targets(
        ds,
        tok,
        text_field,
        [(path, target_tokens)],
        encode_batch_size,
        description,
        progressive=progressive,
    )
    return result[path.name]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dataset_revision(repo_id: str, requested: str | None) -> str:
    info = HfApi().dataset_info(repo_id, revision=requested)
    if not info.sha:
        raise RuntimeError(f"Could not resolve a revision SHA for {repo_id}")
    return info.sha


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "metadata.json").exists():
        raise FileExistsError("Prepared metadata already exists; use a fresh output directory")

    for name in ("stage1_tokens", "decay_tokens", "mix_start_tokens", "val_tokens", "encode_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.mix_start_tokens >= args.stage1_tokens:
        raise ValueError("--mix-start-tokens must be before the learning-rate cooldown")

    # Reject missing/corrupt references locally, before any Hub calls or streaming.
    validation_sha256 = verify_validation_reference(args.validation_reference, args.val_tokens + 1)
    verify_tokenizer_reference(args.tokenizer_json)
    packages = {}
    for name in ("datasets", "numpy", "tokenizers", "huggingface_hub", "pyarrow", "fsspec"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    print("Preparation packages: " + json.dumps(packages, sort_keys=True), flush=True)
    if packages["datasets"] != DATASETS_VERSION:
        raise RuntimeError(
            f"Preparation requires datasets=={DATASETS_VERSION}; found {packages['datasets']}. "
            "Run python -m pip install -r requirements.txt in the same environment used for preparation. "
            "Streaming shuffle algorithms differ between versions; use the pinned preparation version."
        )
    print(f"Seed: {args.seed}; L2 shuffle seed: {args.seed + 1}; "
          f"buffer: {args.shuffle_buffer:,}; input shards: 10; "
          f"encode batch: {args.encode_batch_size}", flush=True)

    tok, tokenizer_path = load_fixed_nanochat_tokenizer(
        args.out_dir,
        repo_id=args.tokenizer_repo,
        revision=args.tokenizer_revision,
        local_json=args.tokenizer_json,
    )
    print(f"Using fixed nanochat tokenizer: {tokenizer_path}")
    print(f"Vocabulary size: {tok.get_vocab_size():,}")
    print(f"BOS/document delimiter: {BOS_TOKEN} -> {tok.token_to_id(BOS_TOKEN)}")
    tokenizer_sha256 = sha256_file(tokenizer_path)
    print(f"Tokenizer SHA256: {tokenizer_sha256}")

    # Resolve moving Hub branches to immutable SHAs before streaming. The resolved
    # revisions are written to metadata.json so the exact source snapshot is known.
    l2_revision = resolve_dataset_revision(args.l2_dataset, args.l2_revision)
    print(f"L2 revision: {l2_revision}")
    print("Preparing validation first; initial streaming fills the shuffle buffer before tokens appear.",
          flush=True)
    if args.check_validation_only:
        pack_stream_across_targets(
            l2_stream(args, args.seed + 1, l2_revision), tok, args.l2_text_field,
            [(args.out_dir / "val_l2.bin", args.val_tokens + 1)], args.encode_batch_size,
            "Ultra-FineWeb L2", validation_sha256=validation_sha256,
            validation_reference=args.validation_reference,
        )
        print("Validation-only check complete. No training corpus or metadata.json was created.", flush=True)
        return
    l3_revision = resolve_dataset_revision(args.l3_dataset, args.l3_revision)
    print(f"L3 revision: {l3_revision}")

    # Budget only the active mixture, with margin for update/cycle rounding.
    l2_mixed = math.ceil((args.stage1_tokens - args.mix_start_tokens) * 0.75
                         + args.decay_tokens * 0.5) + 1_000_000
    l3_each = math.ceil((args.stage1_tokens - args.mix_start_tokens) * 0.125
                        + args.decay_tokens * 0.25) + 1_000_000
    l2_targets = [
        (args.out_dir / "val_l2.bin", args.val_tokens + 1),
        (args.out_dir / "stage1_l2.bin", args.mix_start_tokens + 1),
        (args.out_dir / "decay_l2.bin", l2_mixed + 1),
    ]
    targets = l2_targets + [(args.out_dir / name, l3_each + 1)
                           for name in ("decay_l3_qa.bin", "decay_l3_style.bin")]
    metadata = {
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "bos_token_id": tok.token_to_id(BOS_TOKEN),
        "document_delimiter": BOS_TOKEN,
        "format_version": 2,
        "tokenizer": {
            "family": "nanochat",
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "source_kind": "explicit_local_json" if args.tokenizer_json is not None else "pinned_hf_or_cached",
            "source_repo": args.tokenizer_repo if args.tokenizer_json is None else None,
            "source_revision": args.tokenizer_revision if args.tokenizer_json is None else None,
            "source_path": str(args.tokenizer_json) if args.tokenizer_json is not None else None,
            "local_file": tokenizer_path.name,
            "sha256": tokenizer_sha256,
            "training_corpus": "~2B characters of NVIDIA ClimbMix pretraining data",
        },
        "packing": {
            "document_delimiter": BOS_TOKEN,
            "split_documents_across_output_files": False,
            "extra_next_token_label": True,
            "note": "Document tails are dropped at output-file boundaries to prevent train/validation leakage.",
        },
        "stage1_tokens": args.stage1_tokens,
        "decay_tokens": args.decay_tokens,
        "mix_start_tokens": args.mix_start_tokens,
        "val_tokens": args.val_tokens,
        "curriculum": {
            "stage1": "100% Ultra-FineWeb L2 English for first half",
            "stable_l3_mix": {
                "Ultra-FineWeb L2": 0.75,
                "Ultra-FineWeb-L3 English QA Synthetic": 0.125,
                "Ultra-FineWeb-L3 English Multi-Style Synthetic": 0.125,
            },
            "decay_l3_mix": {
                "Ultra-FineWeb L2": 0.50,
                "Ultra-FineWeb-L3 English QA Synthetic": 0.25,
                "Ultra-FineWeb-L3 English Multi-Style Synthetic": 0.25,
            },
        },
        "datasets": {
            "l2": {
                "dataset": args.l2_dataset,
                "revision": l2_revision,
                "split": args.l2_split,
                "text_field": args.l2_text_field,
            },
            "l3": {
                "dataset": args.l3_dataset,
                "revision": l3_revision,
                "qa_config": args.l3_qa_config,
                "style_config": args.l3_style_config,
                "text_field": args.l3_text_field,
            },
        },
        "files": {path.name: count for path, count in targets},
        "seed": args.seed,
        "preparation": {
            "packages": packages,
            "shuffle_buffer": args.shuffle_buffer,
            "max_buffer_input_shards": 10,
            "encode_batch_size": args.encode_batch_size,
            "parquet_reader": {
                "batch_size": PARQUET_BATCH_SIZE,
                "columns": "configured text field only",
                **PARQUET_SCAN_OPTIONS,
            },
        },
    }
    if validation_sha256:
        metadata["validation_reference_sha256"] = validation_sha256
    if args.progressive:
        metadata["progressive_preparation"] = True
        for path, target in targets:
            with path.open("xb"):
                pass
            publish_tokens(path, 0, target)
        # Publish an immutable plan; changing ready counts live in separate files.
        temporary = args.out_dir / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, args.out_dir / "metadata.json")
    try:
        pack_stream_across_targets(
            l2_stream(args, args.seed + 1, l2_revision), tok, args.l2_text_field,
            l2_targets, args.encode_batch_size, "Ultra-FineWeb L2",
            progressive=args.progressive, validation_sha256=validation_sha256,
            validation_reference=args.validation_reference,
        )
        for config, name, label, offset in (
            (args.l3_qa_config, "decay_l3_qa.bin", "QA", 2),
            (args.l3_style_config, "decay_l3_style.bin", "Multi-Style", 3),
        ):
            pack_single(
                l3_stream(args, config, args.seed + offset, l3_revision),
                tok, args.l3_text_field, args.out_dir / name, l3_each + 1,
                args.encode_batch_size, f"Ultra-FineWeb-L3 {label}",
                progressive=args.progressive,
            )
    except BaseException as exc:
        if args.progressive:
            for path, target in targets:
                previous = read_progress(path)
                if not previous["complete"]:
                    publish_tokens(path, previous["tokens"], target, error=str(exc))
        raise
    if not args.progressive:
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
