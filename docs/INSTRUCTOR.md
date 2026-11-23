# Instructor release notes

The source repository contains the completed default 3B assignment control, historical 3B/50B evidence, editable training code, and frozen evaluation references. Full checkpoints are optional release assets; the exact prepared 3B corpus has a separate three-part release workflow, with reconstruction from pinned sources as an alternative. For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change.

The source repository is public at [spindro/miniple-challenge](https://github.com/spindro/miniple-challenge). Share the current main branch and its first-install guide with the class; source tag `v1.1.0` predates the prepared-data downloader. Optional checkpoint assets are hosted in [release v1.1.0](https://github.com/spindro/miniple-challenge/releases/tag/v1.1.0). See [distribution](DISTRIBUTION.md) for asset names and selected downloads. This packaging does not assign a new overall license; existing source attributions and provenance are retained.

## Defaults chosen for this assignment

- Six tracks; one primary track per project.
- Comparable size means total parameters within ±10% of 146,496,016. All experts, learned tables, and training-only heads count; frozen base parameters count for adapters. Adapter tuning is eligible for E/F even with a small trainable fraction.
- Pretraining quality is scored using frozen full-vocabulary held-out next-token CE. Training CE is still plotted, but its sampled/composite forms cannot fairly rank different objectives.
- A uses the exact 3B budget. B/C target the 3B final CE and share a 3B cap. D declares one of those objectives and identifies the changed data condition.
- E starts from 50B; F defaults to continued pretraining from 50B, with scratch Italian reported separately.
- The rubric in `DELIVERABLES.md` is a suggested research-quality rubric. No due date, group size, mandatory number of full runs, or GPU-hour entitlement has been invented.

Adjust these choices consistently before distributing the handout. The size band, compute allowance, and target precision should be stable once projects begin.

## Before teaching begins

1. Announce deadlines, team size, GPU access/allocation, total experiment budget, submission storage, and grading weights.
2. Verify the published student source and the `miniple-3b.pt` and `miniple-50b.pt` assets hosted in release `v1.1.0`. Keep those full `.pt` files and generated training data out of Git. Confirm `distribution.json` points to the checkpoint repository and tag. A source checkout is sufficient for scratch training after environment setup and prepared-data download; students download only the optional checkpoint needed by their track. Test all three data parts and the selected checkpoint downloads from a clean checkout. Students need network access and about 12 GB free during data download/extraction, plus environment and checkpoint space. Keep the data release `data-3b-v1` independent of checkpoint release `v1.1.0`; distribute only the verified 3B corpus. Retain the [upstream attribution and notices](DATA_ATTRIBUTION.md) alongside the data parts.
3. Allocate environment checks and candidate runs. A/B students can use the provided control after training-byte and recipe parity checks. Use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. A shared RTX 3090 model name alone does not establish timing parity.
4. Freeze an additional document-disjoint test set for final claims if feasible. The shipped 128-sequence monitor is public and already used for development; the rest of the shipped file is public too. No hidden final test set has been created here.
5. Decide the permitted external-data and synthetic-data conditions for D/E/F and how preprocessing/teacher costs enter the class budget. The docs require disclosure and separate conditions; no paid API usage is required.
6. Decide whether final multi-seed confirmation fits the allocation. The handout recommends it when feasible, not unlimited compute.

## Historical artifacts and reproducibility

`baselines/3b/` contains the completed assignment control, its canonical data hashes, exact source snapshot, metrics, plots, validation, benchmark outputs, and timing. The historical 2026-10-22 3B records and original checkpoint manifest are preserved under `baselines/historical_3b_20261022/`; release v1.0 remains unchanged. The 50B model and evidence remain byte-identical.

The current `miniple-3b.pt` installs as `baselines/3b/final_l3_mix.pt`; `miniple-50b.pt` installs as `baselines/50b/final_l3_mix.pt`. Both assets must be present in the configured checkpoint release. Verify downloaded assets against `baselines/manifest.json`; never substitute historical 3B bytes for the assignment control.

For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change.

The new result is one seed on the public development holdout. Numerical behavior may differ across software/kernels. Dataset metadata is provenance, not a portable byte-for-byte file: compare training binaries directly and retain each run's resume fingerprint.

## What is and is not implemented

Included: training/data preparation, generation for the baseline architecture, fixed loss evaluator, pinned four-task base benchmark evaluator, plotting, prepared-data and optional checkpoint downloaders, selected artifact checks, CPU correctness checks, student documents, and agent guidance.

Student work: architecture/optimizer/data innovations, target-checkpoint saving/timing instrumentation when needed, SFT/chat training and generation evaluation, Italian adaptation and benchmarks, and any custom-model adapters.

The default assignment 3B run is now completed and verified; no new 50B run was performed. See `VERIFICATION.md` and the raw baseline evidence for the measured checks and limitations.
