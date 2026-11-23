# Handout verification — 7 November 2026

This record describes packaging checks, not a new training result. The revised handout excludes training data; the frozen validation set/tokenizer and both checkpoints remain.

## Passed

- Starter syntax and shell checks; explicit 3B defaults, seed and pinned source revisions; inert fresh-run protection and resume-invocation checks; preparation with and without bundled reference files; existing-data handling and trainer error propagation.
- Three small tests: frozen validation/tokenizer rejection checks; full-vocabulary loss invariance across microbatch sizes with loader/model state restoration; artifact-verifier success/corruption/missing/unsafe-path checks.
- All three files listed in `baselines/validation/manifest.json` match their byte counts and SHA256 hashes. Both full baseline checkpoints match their historical recorded byte counts and SHA256 hashes.
- The retained evaluation-only metadata, frozen validation hash, and tokenizer hash are accepted by the loss evaluator. Training binaries have been removed; preparation remains a student step.
- Both checkpoints load strictly into the student model implementation on CPU and contain 146,496,016 unique parameters. Each completes a finite 32-target CPU forward/loss probe. These tiny probes are compatibility checks, not benchmark scores; outputs are in `baselines/historical_3b_20261022/checkpoint_load_check_original.json`.
- Model, attention sinks, Muon, sampled-softmax, tokenizer, and four-task evaluator source hashes match the historical 3B manifest. The original source archive's members match that manifest. Current generation code includes later sampling defaults/top-p support; current training/data code includes later preparation safeguards and the documented student defaults.
- Baseline final losses, token budgets, schedules, benchmark figures, and checkpoint hashes were independently checked against the supplied raw records. The 3B plot was visually inspected.
- Relative Markdown links resolve and all packaged JSON documents parse. No access password is stored in student-facing code or documentation.

## Commands

```bash
python3 scripts/check_starter.py
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
```

The small Python tests and artifact checks were run with the existing local Python 3.10 environment; the training installer targets Linux/Python 3.12 as documented. The artifact verifier uses portable standard-library hashing.

## Repository distribution update — 9 November 2026

- The source repository includes the frozen validation/tokenizer reference. Full checkpoints, training data, generated runs, caches, and Office lock files are excluded from Git.
- A clean export of the Git index passed the starter check and default artifact verification with no checkpoints present.
- All ten unit tests passed on local Python 3.12/PyTorch 2.11.0. Downloader tests cover successful selected downloads, skipping already verified files, rejected corrupt/truncated/oversized downloads, network failure cleanup, and preservation of existing files, including a concurrent write at download completion. Verifier tests cover explicit selection, missing unselected checkpoints, checksum failures, and manifest/path protections.
- The existing 3B and 50B checkpoints passed the downloader's verification without network access. Both checkpoints and all frozen validation assets retain their original recorded hashes.
- `source_manifest.json` records byte counts and SHA256 values for the versioned files, excluding itself. Historical manifests and archives remain unchanged.

The downloader uses public GitHub Release URLs from `distribution.json`; private-release access is documented separately. These local packaging checks preceded publication and did not test live release downloads. No training or GPU benchmark was run for this packaging update.

## Data preparation verification — 14 November 2026

- Reconstructed all **5,000,001 validation tokens** from the pinned live L2 source in a clean data-preparation environment. The 5,809 contributing documents produced SHA256 `cd9b0e5cbf20e45922905d249a1dfe397d66abe886aa622d2632eaeb604e81a8`, exactly matching the immutable reference.
- Ran the updated `prepare_data.py --check-validation-only` itself with bounded, cached HTTP-range transport. Its output was independently hashed: validation and tokenizer match the bundled files; no training files or `metadata.json` were created. This was a correctness check with a warm range cache, not a cold-download timing measurement.
- Verified environment: Python 3.12.2, datasets 5.0.1, NumPy 2.5.4, tokenizers 0.23.3, PyArrow 26.0.1, huggingface-hub 1.33.0, fsspec 2026.6.0. `pip check` passed in that clean environment. The full Linux/CUDA training environment was not installed for this check.
- Both pinned L3 configurations yielded a nonempty text row through the new projected, buffered reader. Those smoke checks disabled shuffle and did not reconstruct the full L3 streams.
- The initial L2 shuffle-buffer fill involved roughly 1.61 GB of ranged source reads in the diagnostic. Validation-only preparation can therefore take minutes before emitting tokens, despite its final output being only 10 MB.
- Datasets 5.0.1 is pinned and checked before network access; the ten-input-shard setting is explicit. The verification above applies to this supplied preparation recipe.
- All 29 CPU tests pass, covering early token mismatch, full SHA verification, document boundaries, progressive publication, rejection of incompatible library versions before downloads, canonical references for standalone preparation, and rejection of mismatched cached data before GPU access. Starter/launcher checks and bundled artifact hashes also pass.
- The assignment retains the original validation bytes, tokenizer, checkpoint assets, historical scores, and v1.0 checkpoint download locations. Prepared metadata remains fixed for reproducible experiments and compatible checkpoint resumes.

No full 3B corpus reconstruction, GPU training, benchmark rerun, or Colab/CUDA installation was performed for this fix. The checks above establish the data-preparation contract, not end-to-end training performance.

## Original packaging limits

No full data reconstruction, new 3B/50B GPU training, full-checkpoint validation, four-task benchmark rerun, or fresh Linux/CUDA installation was performed while assembling this revision. Data-preparation arguments and launcher behavior were checked without downloads; complete source reconstruction was not rerun. The supplied benchmark scores and curves are the completed historical runs, with their raw evidence and environment metadata. SFT, Italian training/evaluation, and proposed architecture improvements are assignment tasks, not implemented solutions.

Course staff should run the documented setup and a matched baseline on the allocated course environment before grading speed claims. No hidden test set or course schedule/GPU entitlement was created. These release decisions are listed in `INSTRUCTOR.md`.

## Completed assignment baseline — 2026-11-23T08:23:48.387516+00:00

The unchanged default source `ab78ea34b49edc000e8dc05128857ce6b036c721` completed a seed-1337 scratch run of 2,999,910,400 tokens (18,310 updates). The local integration independently checked the checkpoint hash, supervisor completion, source snapshot, data hashes/metadata, complete validation cadence, strict-model-load report, frozen loss report, complete four-task results and artifact hashes.

The adopted final CE is 3.0082439556717873, with perplexity 20.251805613409555. The trainer/evaluator CE difference is 0 (required ≤0.001 for BF16 corroboration). Complete evidence is in `baselines/3b/`; old 3B evidence is archived, 50B and frozen evaluation files are unchanged. This supersedes the earlier statements that no default 3B run had been executed. It does not establish a universal speed baseline or multi-seed improvement.

Preparation cost totals 14413.765 seconds, including 6299.190 seconds from 1 preserved prior failures before training started. It is excluded from training timing and reported separately in `baselines/3b/reference.json`.

Post-training frozen loss and benchmarks ran on shared GPU 1 with a 25% PyTorch allocator memory limit, after the GPU 0 wait ended by authorized reallocation. Training itself ran on GPU 0; its original timing record is unchanged. Evaluator elapsed times and the later queue wait are excluded from training-speed measurements. The plan, approval, original status snapshots and GPU observations are preserved in the evaluation-continuation evidence. See `baselines/3b/evaluation_plan.json` and `evaluation_continuation_manifest.json`.

## Prepared 3B download packaging — 23 November 2026

- Streamed the seven canonical prepared files from the completed assignment run into a USTAR archive, split into exactly three parts: 2,010,000,000, 2,010,000,000 and 1,999,880,960 bytes. Every archived file matches `baselines/3b/data_manifest.json`, including the original metadata.
- All 44 CPU unit tests passed on local Python 3.12, including 15 downloader fixtures covering corruption, incomplete downloads and retries, archive safety, disk space, existing-data preservation, and failed-install cleanup. Starter and frozen-reference checks passed.
- Tested the real installer on all three full-size local parts, verified all seven installed SHA256s independently, repeated installation without network access, and passed the real `train.py --validate-data-only` preflight. This used no GPU and did not reconstruct or retrain the model.
