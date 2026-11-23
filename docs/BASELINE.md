# Assignment baseline and historical references

## Architecture and recipe

| Property | Reference |
|---|---|
| Unique parameters | 146,496,016 |
| Vocabulary / maximum context | 32,768 / 2048 |
| Layers / residual width / FFN width | 18 / 640 / 3072 |
| Query heads / KV heads / head dimension | 8 / 2 / 64 |
| Attention pattern | `GGLLLGLLLGLLLGLLLG` |
| Global layers | Full causal context, NoPE |
| Local layers | RoPE, 256-token local window |
| PLE / nonlinearity | 64 channels / ReLU² |
| Attention sinks | One learned scalar per query head; 144 total |
| Input/output embedding | Tied |
| Optimizer | Muon for selected hidden matrices, auxiliary AdamW |
| Peak learning rates | Muon `8e-4`, AdamW `6e-4` |
| Batch | 20 × 2048 × 4 = 163,840 tokens per optimizer update |
| Seed | 1337 |
| Training execution | BF16, FlashAttention, Liger, `torch.compile` |

## Exact 3B schedule

| Boundary | Completed updates | Consumed predicted tokens |
|---|---:|---:|
| End of warmup | 62 | 10,158,080 |
| End of pure L2; start 75/12.5/12.5 L2/QA/style | 9,155 | 1,499,955,200 |
| End of stable LR; start cooldown and 50/25/25 mixture | 12,817 | 2,099,937,280 |
| Final checkpoint | 18,310 | 2,999,910,400 |

Requested budgets are 2.1B stable-stage tokens plus 0.9B cooldown, mixture start 1.5B, and 10M warmup. The implementation rounds stages down and warmup up to whole updates. Cooldown linearly reaches zero LR. Sampled-softmax candidate counts are 10,240 for updates 1–10,437 and 14,336 for 10,438–12,817; updates 12,818–18,310 use all 32,768 vocabulary entries. The ordinary validation head always uses the full vocabulary.

The aggregate 3B exposure is approximately 80% L2, 10% QA, and 10% style. Exact file consumption is:

| Training file | Predicted tokens consumed |
|---|---:|
| `stage1_l2.bin` | 1,499,955,200 |
| `decay_l2.bin` | 899,973,120 |
| `decay_l3_qa.bin` | 299,991,040 |
| `decay_l3_style.bin` | 299,991,040 |

The table describes the unchanged assignment recipe. The completed assignment run used the exact freshly prepared default corpus. Compare all four complete training files with `baselines/3b/data.sha256`; validation and tokenizer bytes remain fixed. Prepared-file lengths include margins beyond consumed tokens. Metadata and full data hashes are retained alongside the run. The original training slices and source are archived under `baselines/historical_3b_20261022/`. For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change.

## Results

| Metric | Assignment 3B | Historical 50B |
|---|---:|---:|
| Predicted training tokens | 2,999,910,400 | 49,999,872,000 |
| Final fixed L2 CE | 3.0082439556717873 | 2.7997181564569473 |
| Final fixed L2 perplexity | 20.251805613409555 | 16.44001260667217 |
| HellaSwag `acc_norm` | 32.60% | 39.23% |
| PIQA `acc` | 64.85% | 67.95% |
| ARC-Easy `acc` | 54.63% | 58.38% |
| ARC-Challenge `acc_norm` | 26.37% | 27.65% |

The assignment target is the final checkpoint's frozen batch-20 BF16 evaluator result. Raw trainer validation and its numerical difference are recorded in `reference.json`. This is a single-seed development-set result. The 50B comparison is descriptive: training data, budget, and schedule differ.

All four zero-shot benchmarks use the complete pinned task definitions (15,428 expected examples), with no limit and zero truncated requests. Reported sample counts are checked when present; they are not independently counted when the harness omits them. Exact scores, standard errors, pinned dataset revisions, evaluator source and package versions accompany the raw outputs.

The assignment run took **38103.960 seconds** from launcher start through exit, including CPU preflight and any race-time GPU queue. Trainer-start to completion was **38089.253 seconds** by log timestamps. Preparation and earlier queueing are separate. Use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. See `timing.json` and the protocol.

Total preparation time was **14413.765 seconds**: 8114.576 seconds for the successful preparation plus 6299.190 seconds across 1 earlier failed preparation attempts. This preprocessing cost is reported separately and excluded from the GPU training timer.
The failed attempt records and their original preserved-file locations are documented in `preparation_history.json`. Training had not started in those attempts; their cost is not hidden in the successful run's timing.
The original failed preparation log and supervisor status are preserved under `baselines/3b/preparation_attempt1/`, with their SHA256 manifest alongside the successful-run provenance.

Post-training frozen loss and benchmarks ran on shared GPU 1 with a 25% PyTorch allocator memory limit, after the GPU 0 wait ended by authorized reallocation. Training itself ran on GPU 0; its original timing record is unchanged. Evaluator elapsed times and the later queue wait are excluded from training-speed measurements. The plan, approval, original status snapshots and GPU observations are preserved in the evaluation-continuation evidence.

## Reference checkpoints

| Checkpoint | Bytes | SHA256 |
|---|---:|---|
| `baselines/3b/final_l3_mix.pt` | 1,407,273,863 | `c548e59d86edd069cdfb2e1a7d30740590106d27a99b0970d33f93fef440905a` |
| `baselines/50b/final_l3_mix.pt` | 1,407,273,863 | `957249a427b31e4237063cdc70d789551da6971c3da106a27a8c623d043a7898` |

These are full training checkpoints, including optimizer and RNG/loader state. Completed pretraining schedules are not SFT recipes: adaptation loads model weights into a new training stage.

The assignment control ran source commit `ab78ea34b49edc000e8dc05128857ce6b036c721`. Its source archive, manifest, launch plan, environment, data metadata and hashes are included in `baselines/3b/`. The original 3B source and measurements remain in [the historical archive](../baselines/historical_3b_20261022/README.md); the original v1.0 release remains available. The classroom presentation retains historical figures; this document and PROTOCOL.md define the current assignment reference.

## How to interpret the curves

- Training and validation use different distributions: changing from L2 to QA/style affects training CE. This alone does not diagnose overfitting.
- Candidate-count changes alter the sampled-softmax objective. Do not compare their raw training CE values as if the objective were constant.
- The monitored validation subset is only 262,144 targets, not the full 5M-token file or separate QA/style holdouts.
- A longer run improving benchmarks does not isolate the effect of PLE, sinks, Muon, or any single ingredient. That requires an ablation.

[3B plot](../baselines/3b/losses.png) · [50B plot](../baselines/50b/losses.png) · [Protocol](PROTOCOL.md)
