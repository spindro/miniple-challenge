# Project title

## Question and claim

State the chosen track, starting condition, hypothesis, and the strongest conclusion the evidence supports. Say whether the primary target was met.

## Method

Explain the modification and why it might work. Describe the baseline, controlled variables, parameter inventory, source/tokenizer/checkpoint lineage, training objective, token accounting, schedule, and evaluation protocol. Cite borrowed techniques and code.

## Reproduction

Give exact setup, train/resume, evaluation, and inference commands. Identify source revision/patch, environment, data/model hashes, seeds, GPU/CPU/storage, and artifact locations. Describe the correctness checks and their outputs.

## Results

| Configuration | Total / trainable params | Consumed tokens | Fixed CE | Target reached? | Time to target | GPU-hours | Peak VRAM |
|---|---:|---:|---:|---|---:|---:|---:|
| Control | | | | | | | |
| Candidate | | | | | | | |
| Ablation | | | | | | | |

Use appropriate additional columns for E/F task metrics and retained English capability. Show all seeds and distinguish historical reference numbers from your own measurements. Include the preceding and qualifying evaluation points for a target crossing. Report “not reached” where applicable, not an estimated crossing.

Attach validation-versus-tokens/time plots and full benchmark outputs. For SFT, give template/mask examples and instruction metrics. For Italian, separate scratch/continuation/SFT stages and report tokenizer statistics and English retention.

## Interpretation

What did the ablation establish? Could data, batch size, tuning budget, precision, or a measurement change explain the result? How large is the effect relative to seed or evaluation variability? Describe representative failures, including negative experiments.

## Cost and limitations

Report total GPU-hours and all attempted runs, preprocessing/teacher cost, inherited pretraining, interruptions/replay, and omitted checks. State the scope of any development-set-only or single-seed claim. Propose one next experiment that would resolve the largest uncertainty.

## Artifact index and references

List raw logs, configuration, manifests, plots, evaluations, checkpoints/adapters and their hashes, code, and the sources used.

When using the supplied assignment control for A/B, cite its checkpoint/source and canonical training hashes, document parity, and label it as supplied rather than a run you performed. Use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. D may use the supplied default-data reference arm when other conditions match; its comparison must isolate the data change.
