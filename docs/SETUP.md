# Setup and first experiment

After cloning, run the commands below from the repository root, the directory containing `train.py` and `scripts/`. Use your allocated GPU number; examples use GPU 0. Training and generation require the course Linux/NVIDIA environment. CPU checks can run on a laptop, but CPU timing is not a 3090 result.

## 1. Get the source and inspect it before allocating GPU time

See [distribution and downloads](DISTRIBUTION.md) for the public GitHub repository, release, and download instructions. The repository includes the frozen validation file and tokenizer; it does not include the pretrained checkpoint files or training corpus.

```bash
git clone https://github.com/spindro/miniple-challenge.git
cd miniple-challenge
python3 scripts/check_starter.py
python3 scripts/verify_artifacts.py --checkpoints none
```

This standard-library check validates syntax and 3B defaults and exercises the launcher with inert stand-ins. It neither downloads data nor starts training. Read `AGENTS.md`, [the track briefs](TRACKS.md), and [the protocol](PROTOCOL.md).

**No pretrained checkpoint is needed for the default scratch run.** Each optional reference checkpoint is about 1.4 GB; obtain only the one needed for reference evaluation or adaptation. Allow about **12 GB free during prepared-data download/extraction** and about **6.02 GB for the installed corpus**, plus the environment and several roughly 1.4 GB output checkpoints per experiment. The frozen validation/tokenizer reference is included in Git. The exact prepared **3B corpus** is a separate three-part release; no 50B training corpus is distributed. See [download details](DISTRIBUTION.md).

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

## 4. Download the exact prepared 3B data

Download and verify the exact corpus used by the supplied assignment baseline:

```bash
python scripts/download_data.py
python train.py --data-dir data/ultrafineweb_3b --validate-data-only
(cd data/ultrafineweb_3b && sha256sum --check ../../baselines/3b/data.sha256)
```

The downloader installs into `data/ultrafineweb_3b/` using the independent `data-3b-v1` release configured in `data_distribution.json`. It downloads three roughly 2.01 GB parts and verifies their hashes and the seven extracted files. The corpus contains four packed training streams, frozen validation, the tokenizer, and the original `metadata.json`; preserve all seven files unchanged. It contains only the default 3B data, including the token margins recorded in the baseline manifest. No model checkpoint is needed for scratch training.

The CPU validator checks prepared-file lengths and frozen references. The separate SHA256 command confirms the four training streams as well as validation/tokenizer against the published control. The launcher detects the downloaded metadata and proceeds without reconstructing data. It still runs its CPU checks before querying or waiting for a GPU.

For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change. The hash list covers four training streams plus the frozen validation/tokenizer. Prepared metadata is retained as provenance and may contain machine-specific paths. The CPU validator alone checks lengths and frozen references; the separate SHA256 command establishes training-byte parity.

### Alternative: reconstruct from upstream sources

Reconstruction remains available for studying preparation, independently reproducing the data, or developing a Track D experiment. It uses CPU, disk and network, and needs additional upstream dataset-cache space. The pinned `datasets==5.0.1` environment is required. Use a separate directory so the downloaded reference remains intact:

```bash
python -u prepare_data.py --out-dir data/ultrafineweb_3b_rebuilt
python train.py --data-dir data/ultrafineweb_3b_rebuilt --validate-data-only
(cd data/ultrafineweb_3b_rebuilt && sha256sum --check ../../baselines/3b/data.sha256)
```

For a run using this rebuilt copy, add `DATA_DIR=data/ultrafineweb_3b_rebuilt` to the training command below. Preparation streams pinned L2/L3 revisions, tokenizes and packs documents, preserves document boundaries between files, and verifies that regenerated validation matches the frozen reference. Source metadata paths may differ from the downloaded baseline metadata; compare training bytes, and keep each run's original resume fingerprint.

Preparation prints package versions and shuffle settings. It reads only the configured text column with buffered Parquet I/O. Initial streaming can take minutes and fetch large source chunks before the 20,000-document shuffle buffer produces token progress. The bundled tokenizer and validation are required by both standalone preparation and the launcher.

For a smaller upstream-access check, reconstruct only the frozen validation stream:

```bash
python -u prepare_data.py --out-dir data/validation_check --check-validation-only
```

This checks all 5,000,001 validation tokens and creates no training metadata. It is unnecessary when using the verified prepared-data download. If `DATA_DIR` has no metadata, `scripts/train.sh` still performs full preparation automatically and then starts training; use standalone preparation when you want to inspect the result first.

For **Track D**, prepare the intentionally changed corpus in a new directory such as `data/my_mixture`, preserve the frozen tokenizer/validation, and pass `DATA_DIR=data/my_mixture` to the launcher. Record changed sources, revisions, selection/order/mixture and costs. Intentionally changed training bytes will not match the default hash manifest; retain their own hashes and the control required by [the protocol](PROTOCOL.md).

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
