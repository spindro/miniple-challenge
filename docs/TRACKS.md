# Track briefs

Choose one primary track. You may report secondary benefits, but decide what would count as success before looking at the result. Apply [the common protocol](PROTOCOL.md) throughout.

## A. Lower loss at the same token budget

**Question:** Does your architecture or optimization change extract more useful learning from the same data exposure?

Start from random initialization. Train for exactly **2,999,910,400 predicted tokens** and compare final full-vocabulary validation CE with the supplied assignment control (**3.0082439556717873**). For A/B, you may use the supplied assignment baseline as the unchanged control without retraining it after all four full training-stream SHA256s match its canonical data manifest. Keep the frozen tokenizer/validation, source order, mixture, token accounting and seed 1337 fixed, and disclose software and numerical differences. Otherwise run an unchanged control on your corpus. For C, use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. For D, the supplied default-data run may be the control arm when other conditions match; otherwise run a control that isolates the intentional data change. Show the complete loss-versus-tokens curve and all four base benchmarks. Lower training loss alone does not establish better generalization.

Possible experiments:

- Rebalance depth, width, FFN size, PLE dimensions, or local/global attention within the parameter budget.
- Introduce small sparse experts; measure router overhead, expert utilization, dropped tokens, and all expert parameters.
- Add causal n-gram embeddings; budget their tables and test whether benefits justify their memory traffic.
- Change learning rates, warmup/cooldown, optimizer groups, initialization, or regularization.
- Compare next-token prediction, ordinary hard MTP, and Soft-MTP with correct alignment and gradient checks.

**Minimum evidence:** unchanged baseline/control, one main modification, one ablation that tests the proposed mechanism, parameter inventory, equal-token curves, final CE and benchmarks, GPU time and memory. A smaller loss difference than run-to-run variability is inconclusive.

## B. Reach the same loss with fewer tokens

**Question:** Can your model reach the 3B baseline's validation quality with fewer training-token exposures?

Start from scratch. The target is **CE ≤ 3.0082439556717873** under the frozen evaluator. Your score is the consumed tokens at the first scheduled validation meeting it; save that checkpoint and corroborate it with the same frozen batch-20 evaluator used for the reference. A confirmation does not create an extra unscheduled crossing opportunity. If the target is missed, report not reached, the budget and best observed CE. Do not interpolate a crossing.

Consider a shorter, deliberately designed training schedule; better optimization; architecture changes; or extra prediction objectives. A shorter run needs its own warmup and cooldown plan. Stopping the original schedule early and comparing it against a fully cooled-down run answers a different question. Report how schedules differ.

**Minimum evidence:** token-to-target table, CE-versus-tokens curve, target checkpoint, schedule, ablation, and GPU time. Count every replayed or repeated training token. MTP's additional targets are extra compute, not a discount to the consumed-token counter.

## C. Reach the same loss sooner on one RTX 3090

**Question:** Does your method reduce actual time to useful model quality?

Start from scratch on **one RTX 3090, 24 GB**. Reach the same CE target as B. Report time to the completed qualifying validation, consumed tokens, and the final model size. Maintain the same validation protocol and the 3B training-token cap. Faster tokens/second alone does not win if convergence becomes slower.

Ideas include better batching, fused operations, compile settings, memory layout, optimizer implementation, sampling overhead, attention kernels, and parameter-budget-compatible architecture changes. Preserve correct losses and gradients. Check numerical changes against a small reference calculation before a long run.

Use the supplied timing only when the documented hardware, software, cache state, timer boundaries and comparison endpoint match; otherwise run a matched reference. The supplied complete-run endpoint is not a verified earliest saved time-to-target. The primary timer includes setup, compilation, data stalls, validation and checkpoint work. Report the secondary trainer-start timer separately. A first-crossing comparison needs a reference with compatible qualifying-checkpoint evidence and save policy. See [timing rules](PROTOCOL.md#timing-and-hardware). H100 speedrun numbers are inspiration, not a 3090 timing reference.

**Minimum evidence:** matched baseline and candidate time-to-target, loss-versus-time and loss-versus-tokens, peak allocated/reserved memory, measured throughput, correctness checks, exact hardware/software and cache conditions.

## D. Improve the data mixture

**Question:** Which training examples, mixture weights, or ordering make this model learn more effectively?

Choose A, B, or C as the primary objective and label the result **data track**. Keep architecture, optimizer, and schedule fixed initially. Change one data factor: mixture, curriculum, filtering, deduplication, or a documented domain subset. The supplied default-data baseline can be the reference arm when its other conditions match; otherwise run the needed unchanged control. Combined data/model changes require separate ablations.

Use the same tokenizer and unchanged validation set. Publish source names, revisions, licenses/terms, selection rules, mixture percentages by token count, unique versus repeated exposures, and data-preparation cost. Keep validation and benchmark documents out of the training set, including near duplicates or transformed copies. External synthetic data must disclose its generator and cost; do not quietly add teacher supervision to a pure pretraining comparison.

The preparation script sizes its files for the default mixture. A new mixture can exhaust one source early. Prepare additional disjoint data or explicitly budget repeats; never silently wrap a stream. The existing trainer validates its two mixture patterns, metadata, and resume state. Changing a mixture requires coordinated changes, not just a comment or one metadata value.

**Minimum evidence:** mixture table, dataset provenance, frozen evaluation, source-wise loss if relevant, quality-versus-token curves, preparation and training cost, and a control that isolates the data change. Also report general benchmarks so an easier or narrower corpus does not masquerade as universal progress.

## E. Instruction-tune the 50B base model

**Question:** Can supervised fine-tuning make the pretrained model follow instructions reliably?

Start from `baselines/50b/final_l3_mix.pt`. Keep the frozen tokenizer initially. Implement an SFT pipeline with explicit roles, a documented chat template, correct token boundaries, assistant-response loss masks, padding masks, and a clear end-of-turn convention. Train a fresh optimizer and schedule from the pretrained **model weights**. The supplied trainer is for pretraining, not a completed SFT implementation.

Possible studies: data quality versus quantity, full tuning versus adapters, learning rate, conversation formatting, response-only versus all-token supervision, or a carefully defined task domain. Keep any adapter/MTP/extra embedding parameters in the inventory. Account for the inherited 50B pretraining budget separately from new SFT tokens and GPU-hours.

Split by source/conversation before packing; remove near duplicates. Keep prompts, gold responses, and evaluation templates out of tuning data. Use a fixed development split for decisions and an untouched test split for the final report. Run both the unmodified base model and tuned model under documented prompting and decoding conditions.

Evaluate held-out assistant-token loss, objectively checkable instruction following (for example a frozen IFEval subset/protocol), a preregistered prompt set with a scoring rubric, and the original four multiple-choice benchmarks for retention. Report prompt-level versus instruction-level accuracy separately. Generated examples illustrate behavior but cannot replace systematic evaluation. Disclose any external model judge and calibrate it with human examples.

**Minimum evidence:** tokenizer/template/mask specification, a small mask/shift test, training/validation curves, base-versus-SFT evaluation, reproducible chat inference, checkpoint or adapters plus exact base hash, and failure cases. The existing multiple-choice harness does not implement generation tasks; adding a suitable evaluator is part of this track.

## F. Build an Italian model

Choose and clearly label one starting condition:

1. **Continued pretraining (recommended):** adapt the supplied 50B English base model on Italian text using a new optimizer/schedule. Study Italian quality versus forgetting, optionally with English replay.
2. **Italian from scratch:** train the same-size model for a declared budget up to the 3B cap. Compare with a matched Italian scratch control; do not present this as the same experiment as adapting a 50B checkpoint.

Optional Italian instruction tuning comes after the chosen pretraining condition and must be reported as a separate stage. An English SFT model, an Italian base model, and an Italian chat model have different objectives.

Use the existing tokenizer first. Measure Italian tokens per character/byte, corpus coverage, and document lengths before choosing a dataset. An advanced tokenizer experiment must explain embedding initialization/remapping, account for vocabulary parameters, and evaluate the same raw texts with byte-normalized likelihood and downstream tasks; per-token perplexities across tokenizers are not directly comparable.

Build document-disjoint Italian train/dev/test sets from a pinned, bounded source. Evaluate Italian held-out loss and a fixed set of suitable Italian benchmarks, such as tasks from ItaEval or Evalita-LLM, with their original prompts and metrics documented. Compare against the unmodified 50B model for adaptation, and measure English retention with the frozen L2 set and original benchmarks. Translated English benchmark performance alone does not establish broad Italian ability.

**Minimum evidence:** starting checkpoint/seed, Italian corpus and filtering details, exact adaptation token budget and GPU-hours, before/after Italian and English results, tokenizer statistics, and representative errors. Use separate tables for scratch, continued pretraining, and SFT.

## Choosing a manageable project

Start with one mechanism. A learning-rate or data-ordering study is usually easier to isolate than a new MoE kernel. MoE, Soft-MTP, custom attention, and tokenizer replacement demand additional correctness tests. Run short pilots, inspect their evidence, and allocate the remaining GPU budget before the final experiment. You are not expected to implement all the ideas in this document.
