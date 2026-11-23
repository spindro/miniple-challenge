# Source repository and optional model downloads

The public source repository is [spindro/miniple-challenge](https://github.com/spindro/miniple-challenge). Download the assignment from [source release v1.1.0](https://github.com/spindro/miniple-challenge/releases/tag/v1.1.0). Optional checkpoint assets are hosted in [GitHub Release v1.1.0](https://github.com/spindro/miniple-challenge/releases/tag/v1.1.0). Downloads require no GitHub account or invitation.

## What comes from Git

The source checkout contains editable training code, student documents, agent instructions, tests, historical metrics/plots/benchmark results, checkpoint hash manifests, and the frozen files under `baselines/validation/`, including `val_l2.bin` and `tokenizer.json`.

It excludes full model checkpoints, generated training data, and student experiment outputs. **The default scratch-training workflow needs no pretrained checkpoint.** Prepare training data from the pinned sources using [SETUP.md](SETUP.md).

Clone the course release and verify the included reference files:

```bash
git clone --branch v1.1.0 https://github.com/spindro/miniple-challenge.git
cd miniple-challenge
python3 scripts/check_starter.py
python3 scripts/verify_artifacts.py --checkpoints none
```

A GitHub source archive is also usable: extract it and run commands from its root directory. Release checkpoint assets are separate from source archives.

## Choose only the weights you need

| Use | Download selection | Release asset | Installed path |
|---|---|---|---|
| Scratch pretraining, tracks A–D or scratch Italian | None | None | None |
| Evaluate or generate from the 3B-token reference | `3b` | `miniple-3b.pt` | `baselines/3b/final_l3_mix.pt` |
| Instruction tuning or Italian adaptation from 50B | `50b` | `miniple-50b.pt` | `baselines/50b/final_l3_mix.pt` |
| Work requiring both reference models | `all` | Both assets | Both paths above |

Each checkpoint is about **1.4 GB**; both together are about **2.8 GB**. These are **full training checkpoints** for the new assignment 3B control and unchanged historical 50B model, including optimizer/RNG state, rather than converted weights-only files. Their exact byte counts and SHA256 values are recorded in `baselines/manifest.json`. Load only the trusted instructor release or checkpoints you produced yourself, because the full checkpoint format uses pickle support.

Select one of these workflows:

```bash
# Only the 3B-token reference:
python3 scripts/download_checkpoints.py --checkpoint 3b
python3 scripts/verify_artifacts.py --checkpoints 3b
```

```bash
# Only the 50B-token reference:
python3 scripts/download_checkpoints.py --checkpoint 50b
python3 scripts/verify_artifacts.py --checkpoints 50b
```

```bash
# Both references, only when needed:
python3 scripts/download_checkpoints.py --checkpoint all
python3 scripts/verify_artifacts.py --checkpoints all
```

The downloader requires `--checkpoint`; it does not implicitly select both models. The verifier accepts `--checkpoints none`, `3b`, `50b`, or `all`, defaults to `none`, and **always verifies the frozen validation/tokenizer reference**. Missing checkpoints outside the selected set are expected.

## Repository and release configuration

`distribution.json` records the checkpoint repository and release tag:

```json
{
  "repository": "spindro/miniple-challenge",
  "release": "v1.1.0"
}
```

The downloader uses this checkpoint release automatically; students do not need to change `distribution.json`. If the instructor provides another location, explicitly override it:

```bash
python3 scripts/download_checkpoints.py --checkpoint 50b \
  --repository OWNER/REPO --tag TAG
```

Replace `OWNER/REPO` and `TAG` with the instructor-provided values. Changing the download location does not change the expected checkpoint hashes; use a matching source release and manifest. Do not edit a hash to make an unverified download pass.

## Private copies of the assignment

The official course repository is public. If you use a private copy, you need GitHub access to that repository and its release. The Python downloader uses public HTTPS release URLs; it does not automatically use GitHub CLI authentication or accept embedded credentials. Use an authenticated GitHub CLI download or the release page in an authenticated browser instead.

For example, after installing GitHub CLI and signing in with `gh auth login`, download only the 50B asset into a fresh staging directory:

```bash
gh release download v1.1.0 --repo OWNER/PRIVATE-REPO \
  --pattern miniple-50b.pt --dir checkpoint_download
```

Replace `OWNER/PRIVATE-REPO` with the private copy's repository. Place the downloaded file at `baselines/50b/final_l3_mix.pt`, preserving any existing local file until you have checked what it is, then verify:

```bash
python3 scripts/verify_artifacts.py --checkpoints 50b
```

For the 3B reference, select `miniple-3b.pt`, install it at `baselines/3b/final_l3_mix.pt`, and verify with `--checkpoints 3b`. Never put access tokens or private SSH keys in source files, notebooks, command examples, or submissions.

## Instructor publication checks

- Preserve existing source attributions and notices when preparing another release.
- Include the frozen validation/tokenizer files in Git and retain their hash manifest.
- Upload the unchanged full checkpoints with exactly the asset names above.
- Keep generated training data and experiment outputs out of the repository and release.
- From a clean source checkout, verify `--checkpoints none`, download each asset separately, and verify the corresponding selection. Check the actual public or authenticated access path chosen for students.
- Announce the repository, release tag, access requirements, and compute allocation.

Release `v1.1.0` carries the new `miniple-3b.pt` and the byte-identical historical `miniple-50b.pt`. The original 3B asset remains in [release v1.0](https://github.com/spindro/miniple-challenge/releases/tag/v1.0); its records and hash are under `baselines/historical_3b_20261022/`. Use that historical release's manifest to verify its checkpoint; the current `3b` selector verifies the assignment control.
