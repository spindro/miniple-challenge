# Common experimental protocol

These are the assignment's default rules. The instructor may revise them for the whole class before experiments begin. Record any track-specific departure prominently; a different task must not silently enter the same ranking.

## Size and starting conditions

- Reference size: **146,496,016 unique parameters**. The comparable-size band is ±10%: **131,846,415–161,145,617** inclusive.
- Count all model parameter scalars present during training, including frozen model components, every MoE expert, router, embedding/n-gram table, PLE channel, sink, adapter, and auxiliary prediction head. Report trainable, total, deployed, and active-per-token counts separately. Tied/shared storage counts once. Replacing/rotating learned banks does not evade the budget: report their union as well as peak simultaneous size. Parameter-free buffers do not count as learned parameters, but report their memory and any external retrieval storage.
- Removing an auxiliary head for inference does not remove its training cost or training parameter count. A 1B-parameter MoE with 150M active parameters is outside this size band.
- Tracks A–D start from random initialization: no supplied checkpoint initialization, pretrained embeddings, imported learned n-gram tables, or undisclosed teacher weights. Teacher-generated data is a disclosed data condition, not an architecture-only control.
- E and the adaptation variant of F start from the supplied 50B weights. Report the 50B inherited budget; do not compare their total sample efficiency with scratch models.
- Adapter tuning is allowed for E/F: the size band applies to the complete base-plus-adapter model, not the small trainable subset alone.
- Use one allocated RTX 3090. Report CPU/RAM/storage and any preprocessing/teacher compute separately. Do not combine multiple GPUs and divide wall time to simulate one GPU.

## What counts as a token

The reference consumes **2,999,910,400 ordinary next-token training targets**, in 18,310 updates of 163,840 tokens. This is the exact A budget and the B/C upper cap. The label “3B” is rounded.

Count real training-token exposures, including repeats, replay after a restart, discarded-batch compute if it trained the model, and all mixture sources. In the supplied packed pretraining format, every input position has one next-token target. A boundary label stored in each file is not an additional optimizer-token exposure. Re-reading a token in another update is another exposure.

For MTP, count underlying input positions once toward the standard token budget and log extra supervised future targets separately. Multiple auxiliary targets on the same input do not represent additional input data, but their compute and all their parameters count; also account for extra corpus positions read only to supply future labels. For masked objectives, context/prompt exposures still count even if their labels are excluded from loss. For SFT, report total non-padding input tokens and unmasked assistant loss tokens separately. Padding is not useful data and should not inflate a sample-efficiency claim. Retrieval context, overlapping windows, masking, and any extra token passes must be reported explicitly.

Preserve effective batch size **163,840** initially. You may change physical microbatch and accumulation, but the starter selects a data source per microbatch: changing `20×4` to `10×8` changes which source examples share an optimizer update. For a strict systems-only comparison, preserve the original logical batches/source order independently of physical chunking and verify equivalence. Otherwise disclose the data-order change as a D/combined condition, even if aggregate mixture percentages match.

If you change effective batch size, log actual exposures and make the final equal-token checkpoint exact (for example with a final partial batch). Do not compare a 2.98B checkpoint with the 3B checkpoint and label it equal tokens. The starter floors stage budgets and enforces complete mixture cycles; changing geometry can require schedule/loader changes.

## Frozen loss measurement

The primary metric is mean **full-vocabulary, ordinary next-token cross-entropy in nats** over the first **128 × 2048 = 262,144 targets** in `baselines/validation/val_l2.bin`, starting at position zero with the same row boundaries. Perplexity is `exp(CE)`. This frozen evaluation reference is included in Git; the prepared 3B training corpus is a separate download, with reconstruction available as an alternative.

| Artifact | SHA256 |
|---|---|
| Entire `val_l2.bin` (5,000,001 uint16 tokens) | `cd9b0e5cbf20e45922905d249a1dfe397d66abe886aa622d2632eaeb604e81a8` |
| `tokenizer.json` | `78025606dd6a505cd74f4b187866bf5540ee83239131aa2f1ddefdfd2e104272` |

Use context 2048 and the 32,768-token course vocabulary. Validation is causal, uses every target, includes no auxiliary losses, and does not train or alter running state. Keep the ordinary next-token head. An architecture change may need an adapter to the evaluator, but cannot change the examples, targets, normalization, or scoring objective.

The supplied `scripts/evaluate_loss.py` checks the artifact hashes and reuses the trainer's validation routine. Its default BF16 CUDA result can differ slightly with batch geometry or kernels. Record precision, backend, and batch size; the original validation used microbatch 20. Resolve tiny threshold differences by rerunning the reference under the same evaluator settings, not by rounding a near miss into a pass.

Plot training objectives separately by candidate count and auxiliary term. A sampled-softmax CE, composite MTP loss, router penalty, and full-vocabulary next-token CE are different quantities. The assignment 3B target is **3.0082439556717873**, taken from the final checkpoint's frozen batch-20 BF16 evaluation in `baselines/3b/validation.json`; it is not final training CE. Trainer/evaluator differences and the first observed scheduled crossing, if any, are retained in `baselines/3b/reference.json`. No first-crossing checkpoint is inferred from the final checkpoint.

## Evaluation cadence and crossing a target

Reference validation occurs every **250 updates = 40,960,000 consumed tokens**, plus the final checkpoint. Use this token grid for A–D. With a different effective batch, evaluate at the first completed update at or after each grid point; log the actual position and overshoot. Prefer batch sizes that divide the grid. Always evaluate at the exact final budget.

For B/C, the qualifying observation is the first **scheduled** validation with CE ≤ the target. Additional diagnostics can be shown but do not create extra opportunities to claim an earlier crossing. Report the preceding scheduled evaluation as well. Without a previous point, report the first observation as an upper bound. No interpolated or extrapolated crossing counts as a measured result.

Save the qualifying checkpoint immediately. The stock trainer logs validation but saves `latest.pt` every 1,000 updates, so students pursuing B/C must add target-checkpoint saving or an equivalent reproducible save policy. Include its cost in any time comparison and use the same policy for the matched reference. If the target is missed, report “not reached”; do not silently relax it.

## Timing and hardware

For C, the primary timer is **training-process launch through completed qualifying validation and saved target checkpoint**, using a monotonic elapsed clock. Include imports, CUDA/model/optimizer/data-loader setup, compilation/autotuning, updates, data waiting/transfers, scheduled evaluation, and checkpoint writes. Start after the GPU allocation and canonical pretokenized data are available; exclude queueing and offline data preparation equally, and report those costs separately. Also report the secondary training-start-to-target timer immediately before the first update, so the historical log has a comparable reference boundary. Use identical boundaries in reference and candidate. CUDA operations must complete before stopping a timer.

The assignment run records 38103.960 seconds for the training launcher and 38089.253 seconds from the trainer's start event to completion. Launcher timing includes its CPU preflight and any race-time GPU queue; preparation and prior queueing are separate. Use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. Raw timing is in `baselines/3b/timing.json`. A first-crossing comparison may require a reference with the same qualifying-checkpoint save policy.

Use matched software, GPU power/clock policy, exclusive GPU access, thermal conditions, storage, CPU threads, and compiler-cache state. Report cold-cache timing as the main result, or clearly label both runs as warm-cache and also report compilation separately. Do not exclude compilation from only one run. For steady-state microbenchmarks, synchronize CUDA, warm up, repeat measurements, and label them separately from end-to-end training.

Log GPU model/UUID, driver, PyTorch/CUDA, FlashAttention/Liger versions, precision, compile flags, thread counts, peak allocated/reserved memory, and source revision. Do not change another user's GPU settings or processes. A resumed run must include all active training segments, repeated-token compute, and restart compilation; disclose downtime and process-level elapsed time separately. A single uninterrupted confirmation run is preferable for a final speed claim.

## Data and benchmark integrity

For architecture/optimizer comparisons A–C, preserve training bytes, tokenizer, data order, and source-mixture boundaries; report schedule and batching changes. A change to data selection/order/mixture belongs to D or a separately labelled combined experiment. A changed learning-rate schedule is allowed and must be controlled/documented.

Training data is kept out of Git. Install the exact prepared 3B corpus with `python scripts/download_data.py`, or reconstruct it with the pinned preparation code. Verify all canonical training bytes using `baselines/3b/data.sha256`, and retain the resulting files for your experiment. For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change. Metadata paths may differ across machines; do not edit a checkpoint's data fingerprint to bypass resume protections. Historical 3B data provenance is archived separately.

All splits must be document-disjoint. Avoid exact and near duplicates across training, validation, and benchmarks. Keep source revisions and filtering manifests. The public 128-sequence validation set is a development measurement and can be overfit through repeated decisions. Final claims should also use a document-disjoint instructor-held-out test set; until provided, label claims as development-set results. The remaining public validation tokens are broader checks, not a secret test set.

Report the full zero-shot four-task suite without `--limit`: HellaSwag `acc_norm`, PIQA `acc`, ARC-Easy `acc`, ARC-Challenge `acc_norm`. Preserve task/dataset revisions, prompts, choice scoring, and the fixed tokenizer. Report raw accuracy and normalized accuracy when available; never swap them to improve a table. SFT/Italian tracks additionally need their task-specific generation/language evaluation.

## Evidence and statistical interpretation

Use seed 1337 for the initial controlled comparison. If the allocated budget permits, repeat the final reference/candidate comparison with two additional seeds (for example 1338 and 1339), and report all runs, mean, and spread. One seed supports a preliminary observation, not a robust universal win. Do not spend unallocated GPU-hours to meet an unstated replication requirement.

Record all attempted configurations and their cost, including failed pilots. Match the tuning budget when comparing methods. Save raw logs, exact commands, code revision/diff, environment, model and data hashes, evaluation outputs, and checkpoint lineage. Do not delete inconvenient runs or report only a favorable seed. Grade scientific reasoning and reproducibility alongside the improvement, especially when results are negative.
