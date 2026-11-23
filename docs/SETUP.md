# Setup and first experiment

After cloning, run the commands below from the repository root, the directory containing `train.py` and `scripts/`. Use your allocated GPU number; examples use GPU 0. Training and generation require the course Linux/NVIDIA environment. CPU checks can run on a laptop, but CPU timing is not a 3090 result.

## 1. Get the source and inspect it before allocating GPU time

See [distribution and downloads](DISTRIBUTION.md) for the public GitHub repository, release, and download instructions. The repository includes the frozen validation file and tokenizer; it does not include the pretrained checkpoint files or training corpus.

```bash
git clone --branch v1.1.0 https://github.com/spindro/miniple-challenge.git
cd miniple-challenge
python3 scripts/check_starter.py
python3 scripts/verify_artifacts.py --checkpoints none
```

This standard-library check validates syntax and 3B defaults and exercises the launcher with inert stand-ins. It neither downloads data nor starts training. Read `AGENTS.md`, [the track briefs](TRACKS.md), and [the protocol](PROTOCOL.md).

**No pretrained checkpoint is needed for the default scratch run.** Each optional reference checkpoint is about 1.4 GB; obtain only the one needed for reference evaluation or adaptation. Allow roughly 6 GB for the prepared default training corpus, additional space for dataset/download caches and the environment, and several roughly 1.4 GB output checkpoints per experiment. The frozen validation/tokenizer reference is included in the source checkout. **Training data is not included:** students download and tokenize it with the supplied preparation code.

## 2. Install on Linux x86_64 with Python 3.12

```bash
bash scripts/install.sh
source .venv/bin/activate
python -m pip freeze > environment.txt
nvidia-smi
```

The installer uses PyTorch 2.8.0/CUDA 12.8, FlashAttention 2.8.3.post1, and Liger 0.8.4. A working NVIDIA driver is required. The FlashAttention wheel is specific to the documented Python/PyTorch platform. If it is unavailable, use a compatible CUDA 12.8 toolkit and compiler to build the same version:

```bash
python -m pip install packaging ninja psutil einops
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2 \
  python -m pip install --no-build-isolation --no-binary=flash-attn flash-attn==2.8.3.post1
```

For CPU-only development, use a compatible PyTorch installation and `python -m pip install -r requirements.txt`. CPU fallback supports small correctness checks; the production launcher still requires CUDA. Existing custom MiniPLE weights are not directly loadable by Transformers or TRL without an adapter/conversion.

The installer includes **`datasets==5.0.1`**, the verified version pinned in `requirements.txt`. Preparation checks this dependency before any dataset download to keep the validation stream reproducible. Use the same environment for installation, preparation, and the launcher.

## 3. Select and verify reference assets

For scratch training, verify the bundled validation/tokenizer without downloading model weights:

```bash
python3 scripts/verify_artifacts.py --checkpoints none
python -m unittest discover -s tests -v
```

The artifact verifier **always** checks the frozen validation/tokenizer reference. It checks only the checkpoint selection requested with `--checkpoints none|3b|50b|all`; the default is `none`. Missing unselected checkpoints are expected. These checks do not require training data or use a GPU. The unit tests exercise frozen loss evaluation on a tiny CPU model, including batch-size invariance and rejection of modified data. They do not run the full pretrained model.

For 3B reference evaluation:

```bash
python3 scripts/download_checkpoints.py --checkpoint 3b
python3 scripts/verify_artifacts.py --checkpoints 3b
```

For E/F adaptation from the 50B-token model:

```bash
python3 scripts/download_checkpoints.py --checkpoint 50b
python3 scripts/verify_artifacts.py --checkpoints 50b
```

Use `--checkpoint all` and `--checkpoints all` only when both references are needed. Downloads use `distribution.json` and write to `baselines/3b/final_l3_mix.pt` or `baselines/50b/final_l3_mix.pt`. Both are **full training checkpoints**, preserving their original optimizer/RNG state and hashes. They have not been converted into weights-only exports. See [distribution](DISTRIBUTION.md) for repository/tag overrides and private-release access.

## 4. Reconstruct training data

Run preparation separately so you can inspect the result before allocating a long training job:

```bash
python -u prepare_data.py --out-dir data/ultrafineweb_3b
python train.py --data-dir data/ultrafineweb_3b --validate-data-only
(cd data/ultrafineweb_3b && sha256sum --check ../../baselines/3b/data.sha256)
```

Preparation uses CPU, disk, and network access; it does not train or require a GPU. It streams the pinned L2/L3 source revisions, tokenizes documents, writes document-disjoint packed files and metadata, and checks that regenerated validation exactly matches the frozen reference. The second command verifies the prepared files on CPU before you start training. Preserve the resulting files and metadata for both control and candidate.

The bundled tokenizer and validation files are the defaults for both standalone preparation and the training launcher. Missing or modified references fail before dataset downloads. Preparation prints package versions and shuffle settings and shows progress for each output file. It reads only the configured text column with buffered Parquet I/O. Initial streaming needs to fetch large source chunks and fill the 20,000-document shuffle buffer before token progress appears; this can take minutes and transfer much more than the 10 MB validation output.

For an optional data-access check before full preparation, reconstruct only the validation stream:

```bash
python -u prepare_data.py --out-dir data/validation_check --check-validation-only
```

This reconstructs all 5,000,001 validation tokens and checks the full SHA256. It uses CPU/network and can take minutes, but never loads L3 data or starts training. It creates no training metadata; use a separate directory for full preparation.

The default `scripts/train.sh` also performs full preparation automatically if the selected `DATA_DIR` has no metadata. It requires the bundled references and validates prepared data on CPU before querying or waiting for a GPU. Unlike the separate preparation command, it proceeds to training afterward. It uses `.venv/bin/python` unless `PYTHON` is set; activate that environment for standalone commands too.

For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change. The hash list covers four training streams plus the frozen validation/tokenizer. Prepared metadata is retained as provenance and may contain machine-specific paths. The CPU validator alone checks lengths and frozen references; the separate SHA256 command establishes training-byte parity.

## 5. Train your candidate or reproduce the reference

The published A/B reference need not be retrained after parity checks. Make and record your experimental change before launching your candidate, and use a fresh output directory. Running unchanged source is an optional reference reproduction when needed. Check configuration and CUDA availability with `bash scripts/debug.sh` before an allocated training job.

```bash
GPU_ID=0 OUT_DIR=runs/my_candidate_3b bash scripts/train.sh
```

Defaults: random initialization, seed 1337, 20 sequences per microbatch, four accumulation steps, context 2048, Muon plus AdamW, BF16, FlashAttention, Liger, compiled execution, and the 3B schedule. See BASELINE.md for the measured assignment-run timing; other machines and compiler-cache states can differ. Use the course scheduler or a persistent session for long jobs.

The launcher refuses to overwrite an existing run and waits if the selected GPU has a compute process. It does not reserve the GPU against another simultaneous launcher; follow the allocation system. An output-directory lock only protects that one directory.

Inspect and plot a run:

```bash
tail -n 20 runs/my_candidate_3b/train.log
python loss_plots.py runs/my_candidate_3b
# Optional, ongoing refresh in another terminal:
python loss_plots.py --watch runs/my_candidate_3b
```

Outputs include `metrics.jsonl`, `status`, `train.log`, `latest.pt`, milestone checkpoints, and `final_l3_mix.pt`. The plotter writes `losses.png` and `losses.svg`. Start/validation/completion events and token counters are in the JSONL log.

To change optimizer, architecture, batch geometry, or validation behavior, edit the corresponding student copy and record the diff. `scripts/train.sh` accepts only `--resume` as a positional option; arbitrary `train.py` flags are not forwarded as a general experiment interface. Read its actual command before modifying it. `STAGE1_TOKENS`, `DECAY_TOKENS`, `MIX_START_TOKENS`, and `WARMUP_TOKENS` are explicit environment overrides, but prepared metadata and complete mixture cycles must match.

## 6. Resume your own interrupted run

```bash
GPU_ID=0 OUT_DIR=runs/my_candidate_3b bash scripts/train.sh \
  --resume runs/my_candidate_3b/latest.pt
```

Keep the original data, model, schedule, optimizer settings, output directory, and batch geometry. The loader/RNG/optimizer state is restored. The checkpoint can lag behind the log: preserve the original log, reconcile records after the saved step, and account for replayed compute before making plots or timing claims. The starter does not automatically repair duplicate log segments after a crash.

Do not use `--resume` on a completed reference checkpoint to start SFT, change data, or extend a zero-LR schedule. E/F load model weights into a deliberate new stage with a fresh optimizer and schedule. Preserve data fingerprints when resuming your own interrupted run.

## 7. Evaluate a checkpoint

Download and verify the selected reference as described in section 3, or use your own saved checkpoint. Frozen validation loss (batch 20 matches the original monitoring geometry; use a smaller batch if required and record it):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_loss.py \
  --checkpoint baselines/3b/final_l3_mix.pt \
  --data-dir baselines/validation --batch-size 20 \
  --out runs/reference_3b_validation.json
```

Use `baselines/50b/final_l3_mix.pt` for the other reference, or your own checkpoint. The evaluator defaults to `baselines/validation`, so it works before training data is prepared. It reads the model configuration and weights, validates frozen inputs, and reports the checkpoint hash and loss. For changed architectures, implement a compatible next-token evaluation path and verify it against this reference protocol. Its reported `training_tokens = completed_steps × tokens_per_step` assumes the original constant-batch format; use your explicit exposure counter for variable-batch or resumed/replayed experiments.

Full benchmark suite:

```bash
python -m pip install -r requirements-eval.txt
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint runs/my_candidate_3b/final_l3_mix.pt \
  --data-dir baselines/validation --batch-size 8 \
  --out runs/my_candidate_3b/evaluation.json
```

`--limit 2` is useful for a functional pilot only. Omit it for reported scores. This evaluator scores multiple-choice likelihoods; it does not implement IFEval, chat evaluation, or arbitrary Italian tasks. Extend it or add a separate evaluator for those tracks without changing the reference scoring.

## 8. Generate a base-model continuation

This example requires the optional 50B-token checkpoint; download and verify it using section 3 first.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python scripts/generate.py \
  --checkpoint baselines/50b/final_l3_mix.pt \
  --data-dir baselines/validation \
  --prompt "The most important discovery was" --max-new-tokens 128 \
  --temperature 1.0 --top-p 0.95 --top-k 20
```

The supplied models are base language models, not chat assistants. The generator uses a CUDA KV-cache runtime specific to the existing architecture; architecture changes may require matching inference changes. Prompt plus generation must fit the 2048-token context. Use fixed prompts, sampling settings, and seeds for comparisons.

Only load instructor-provided or self-produced `.pt` files: these full training checkpoints contain Python optimizer/RNG state and are loaded with pickle support.
