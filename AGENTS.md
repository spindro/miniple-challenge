# Working in the MiniPLE student assignment

Read `README.md`, `docs/PROTOCOL.md`, the chosen section of `docs/TRACKS.md`, and the relevant code before editing. The student's stated track, approved resource allocation, and current experiment proposal define the task. If the track is unclear, inspect existing work and ask one focused question while doing safe setup checks.

## Help the student conduct an experiment

- Explain the hypothesis, the smallest useful change, the expected metric, and the check that could falsify the idea. Help the student understand and defend the result.
- Keep a concise experiment ledger: configuration, command, code revision/diff, seed, token budget, source/checkpoint hashes, GPU-hours, result, and decision. Use `templates/PROPOSAL.md`, `templates/REPORT.md`, and `templates/result.json`.
- Change one main factor before combining improvements. Preserve negative results and failed pilots. Never invent benchmark scores, GPU timings, paper claims, or completed runs.
- Read linked research when relying on it. Cite the paper/repository and exact revision used. A technique's success on H100s or another dataset is a hypothesis for this 3090 task.

## Preserve the scientific comparison

- A–D start from random weights. E/F adaptation loads the identified pretrained model weights and starts a deliberate new training stage. Never disguise pretrained initialization as scratch training.
- The exact reference token budget is 2,999,910,400; the common target is full-vocabulary next-token CE 3.0082439556717873 on the frozen 128×2048 validation protocol.
- Do not alter validation bytes, tokenizer, causal context, target alignment, task prompts, loss normalization, or benchmark examples to improve a score. Architecture adapters must preserve the scoring definition.
- Count total model parameters, all experts/lookup tables/auxiliary heads, and repeated data exposures honestly. Log auxiliary prediction events and adaptation lineage separately.
- Do not compare sampled-softmax loss or an auxiliary/composite loss directly with the frozen next-token CE. Report raw and normalized benchmark accuracy under their correct names.
- Keep data-mixture changes explicitly labelled. Check document/near-duplicate leakage and record data provenance before training.

## Files and implementations

- Treat `baselines/`, including its frozen validation/tokenizer reference, as immutable. Training data is not distributed: reconstruct it with the supplied preparation code in `data/ultrafineweb_3b/`, then keep that prepared corpus fixed for controlled comparisons. Use new directories for changed data and fresh `runs/<descriptive_name>/` directories for runs.
- The Git checkout contains frozen validation/tokenizer files and historical results. Full checkpoints are optional GitHub Release assets: use `scripts/download_checkpoints.py` with a required `--checkpoint` selection of `3b`, `50b`, or `all`, following `docs/DISTRIBUTION.md`. Adding a verified release checkpoint to its designated baseline path is allowed; modifying the reference bytes is not. Default scratch training needs no downloaded checkpoint.
- Preserve output locks, checkpoint validation, metadata/hash checks, and resume compatibility checks. Do not bypass a failed check by falsifying metadata.
- Make surgical changes to the relevant code. Update configuration, serialization, optimizer groups, evaluation, and inference callers when a model change requires it; avoid unrelated refactors.
- Use `rg` for code search. Reuse existing helpers and the existing environment. Do not add dependencies or frameworks without a concrete need.
- The standard launcher only forwards resume options. Do not invent CLI flags: inspect the parser and launcher, then implement and document any new option.
- Only load trusted instructor/self-produced full `.pt` checkpoints; never put access credentials in code, logs, reports, or the handout.

## Verify before spending the budget

Start with `python3 scripts/check_starter.py` and the CPU tests. Verify selected artifacts with `scripts/verify_artifacts.py`; `--checkpoints none` always checks frozen references. Follow SETUP.md to prepare data and check every canonical training-stream hash. For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change.

For first-time setup, follow the sequence in `docs/SETUP.md`: install the provided requirements, verify the bundled references, prepare `data/ultrafineweb_3b`, and validate it with `python train.py --data-dir data/ultrafineweb_3b --validate-data-only` before training. The optional `python prepare_data.py --out-dir data/validation_check --check-validation-only` checks data access and reconstructs just the frozen validation stream on CPU/network; it does not prepare training data. Both preparation and the launcher require the bundled references.

For a substantive model/optimizer/data change, add a small relevant correctness check: causality, shifted targets, gradients, accumulation, routing, masks, or resume equivalence. Verify actual parameter counts. Follow with a bounded pilot and then the planned experiment. Never launch the full default 3B run as a generic smoke test.

Check the existing GPU allocation and running jobs before using CUDA. Stay within the student's authorized budget. Once a specific run and resource budget are authorized, execute and monitor it without repeatedly asking for the same approval. Do not kill other jobs, change shared GPU power settings, or consume extra GPUs. If no compute allocation has been specified, continue code/CPU checks and ask for the missing allocation before a long GPU job.

For speed claims, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. Include compilation, validation, checkpointing and replayed work as specified in the protocol. Preserve interrupted logs and reconcile steps before plotting.

## Communicate evidence

State what changed, why, what was actually checked, the measured result, and remaining limitations. Distinguish a proposed experiment, a running job, a completed run, and a verified score. Explain error messages instead of weakening protections until a run starts. Keep the final submission reproducible and readable by a reviewer who did not see the chat.
