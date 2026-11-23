# Reference model and historical results

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

The table describes the historical run; training files are not distributed. Students reconstruct training data with the supplied code. Each prepared stream needs an additional next-token label. The original full 5,000,001-token validation stream and tokenizer remain in `baselines/validation/`, with their own hash manifest. `baselines/3b/data_metadata_original.json` and `historical_data_manifest.json` preserve the historical data provenance and consumed-prefix hashes for audit, not a list of included training files. The default fresh corpus uses different training slices; compare changes with a matched fresh control.

## Results

| Metric | 3B reference | 50B reference |
|---|---:|---:|
| Predicted training tokens | 2,999,910,400 | 49,999,872,000 |
| Final fixed L2 CE | 3.0081611946225166 | 2.7997181564569473 |
| Final fixed L2 perplexity | 20.250129622081722 | 16.44001260667217 |
| HellaSwag `acc_norm` | 32.95% | 39.23% |
| PIQA `acc` | 64.15% | 67.95% |
| ARC-Easy `acc` | 54.59% | 58.38% |
| ARC-Challenge `acc_norm` | 26.88% | 27.65% |

Both final validation results were the best recorded for their respective runs. The 50B run used the same architecture, with schedule boundaries scaled to 25B for mixing and 35B for cooldown/full softmax. It resumed from a saved checkpoint after a power interruption. Its training history is supplied for study, not as a one-segment speed baseline.

Benchmark results cover all 15,428 examples: HellaSwag 10,042, PIQA 1,838, ARC-Easy 2,376, ARC-Challenge 1,172. They are zero-shot multiple-choice likelihood evaluations with no request truncation and no example limit. The pinned harness commit is `d6de81643928d653435c431bae19945d41d32520`. Exact scores, standard errors, dataset revisions, and environment details are in each `evaluation.json` and `evaluation_metadata.json`.

The original 3B run logged about **11h 00m 35s** from the training-start event to completion, including first-use training compilation, validation, and checkpoint writes, but excluding earlier setup. The metadata's `elapsed_seconds` field is the **benchmark evaluation duration**, not training time. Use `training_duration_seconds` for that historical figure. A fair speed comparison requires a matched reference rerun.

## Reference checkpoints

| Checkpoint | Bytes | SHA256 |
|---|---:|---|
| `baselines/3b/final_l3_mix.pt` | 1,407,273,735 | `3b9c1ed9daff59f26beb115bb4631aa0ec802171832a6d44493609d40d766e4a` |
| `baselines/50b/final_l3_mix.pt` | 1,407,273,863 | `957249a427b31e4237063cdc70d789551da6971c3da106a27a8c623d043a7898` |

These are full original training checkpoints, including model weights/configuration, optimizer, schedule, and RNG/loader state. They are suitable for scoring and as the model-weight source for adaptation. Their completed schedules and historical data fingerprints are not new-stage training recipes.

The student source preserves the model/optimizer/evaluator implementation and makes 3B defaults explicit. It also contains later resume/data-handling safeguards and an added fixed-loss evaluator. It is not claimed to be a byte-identical copy of the historical launch source. The original source manifest and the 3B `source_at_launch.tar.gz` are included for audit; the current handout has its own source manifest. Do not execute the historical launcher as the student default.

## How to interpret the curves

- Training and validation use different distributions: changing from L2 to QA/style affects training CE. This alone does not diagnose overfitting.
- Candidate-count changes alter the sampled-softmax objective. Do not compare their raw training CE values as if the objective were constant.
- The monitored validation subset is only 262,144 targets, not the full 5M-token file or separate QA/style holdouts.
- A longer run improving benchmarks does not isolate the effect of PLE, sinks, Muon, or any single ingredient. That requires an ablation.

[3B plot](../baselines/3b/losses.png) · [50B plot](../baselines/50b/losses.png) · [Protocol](PROTOCOL.md)
