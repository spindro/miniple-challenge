# Prepared 3B data: attribution and upstream notices

This document accompanies the `data-3b-v1` prepared-data release. The pinned upstream sources and notices were checked on 23 November 2026. The instructor confirmed upstream permission to redistribute this processed course subset on 28 November 2026. Upstream licenses and attribution notices remain in effect.

## Corpus sources

| Component | Pinned source | Selection used by the course | Upstream license declaration |
|---|---|---|---|
| L2 English | [OpenBMB Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/02c85641e3d19a854be2e09139c25adaa9518063) | `en` split, `content` field | Apache 2.0, with a dataset-card instruction to check the underlying datasets' licenses |
| L3 English QA and style | [OpenBMB Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3/tree/bc3b1ba986fcaef6871b9790a413b16267c2de0f) | `Ultra-FineWeb-L3-en-QA-Synthetic` and `Ultra-FineWeb-L3-en-Multi-Style-Synthetic`, `content` field | Apache 2.0 in LICENSE, plus the redistribution notice preserved below |

Both pinned repositories contain the same [Apache 2.0 license text](licenses/Ultra-FineWeb-Apache-2.0.txt), including the OpenBMB copyright notice. Original files: [L2 LICENSE](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/blob/02c85641e3d19a854be2e09139c25adaa9518063/LICENSE), [L3 LICENSE](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3/blob/bc3b1ba986fcaef6871b9790a413b16267c2de0f/LICENSE).

The [pinned L3 dataset card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3/blob/bc3b1ba986fcaef6871b9790a413b16267c2de0f/README.md) also states a prior-written-permission restriction on reposting/rehosting artifacts. Its complete license-section wording is retained in [the redistribution notice](licenses/Ultra-FineWeb-L3-redistribution-notice.txt). Retain both the LICENSE and this notice with any authorized distribution.

Credit for Ultra-FineWeb: Yudong Wang and collaborators, *Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data* ([technical report](https://arxiv.org/abs/2505.05427)). Credit for Ultra-FineWeb-L3: Hongya Lyu and collaborators, OpenBMB, 2026; the pinned card attributes its refinement to the MiniCPM4 and Qwen3 families and its seed data to Ultra-FineWeb. Full author lists and recommended citations are in the linked dataset cards.

## Upstream English web-data lineage

The [pinned Ultra-FineWeb card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/blob/02c85641e3d19a854be2e09139c25adaa9518063/README.md) attributes its English data to **FineWeb**. Credit: Hugging Face's FineWeb authors and contributors; see the [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [technical report](https://arxiv.org/abs/2406.17557), and [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/). FineWeb's card also identifies [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use). These upstream notices are retained; the course metadata pins Ultra-FineWeb, not a separate FineWeb revision. The course does not claim ownership of the underlying web documents or replace their applicable terms.

## Frozen tokenizer

The course's 32,768-token BPE tokenizer comes from the nanochat family, distributed in [`crellis/d18-9tpp-hf-sft` at revision `79e88e310d774ee11a1dd3c154a639f0cc21bcb2`](https://huggingface.co/crellis/d18-9tpp-hf-sft/tree/79e88e310d774ee11a1dd3c154a639f0cc21bcb2). Its [pinned model card](https://huggingface.co/crellis/d18-9tpp-hf-sft/blob/79e88e310d774ee11a1dd3c154a639f0cc21bcb2/README.md) declares MIT, attributes the implementation to Andrej Karpathy's nanochat, and describes tokenizer training on about two billion characters of the pretraining corpus. Credit: crellis and the [nanochat contributors](https://github.com/karpathy/nanochat). The retained [nanochat MIT notice](licenses/nanochat-MIT.txt) includes Copyright (c) 2025 Andrej Karpathy. The pinned conversion repository has no separate LICENSE file; its MIT declaration is in the model card.

The tokenizer's upstream training corpus is NVIDIA ClimbMix. This prepared release contains the course tokenizer and selected Ultra-FineWeb token streams; it does not distribute ClimbMix, the crellis model weights, or its SFT datasets. The tokenizer SHA256 remains `78025606dd6a505cd74f4b187866bf5540ee83239131aa2f1ddefdfd2e104272`.

## Course transformations and exact-file preservation

The course preparation selected English L2/L3 documents with pinned revisions and seed 1337, shuffled the streams, encoded text with the fixed tokenizer and BOS document delimiters, and packed unsigned 16-bit token files. Documents do not cross output-file boundaries. The files include held-out validation and token margins beyond the 2,999,910,400 predicted tokens consumed by the baseline. These are processed course subsets, not complete upstream Parquet repositories.

The archive contains exactly seven unchanged baseline files: `stage1_l2.bin`, `decay_l2.bin`, `decay_l3_qa.bin`, `decay_l3_style.bin`, `val_l2.bin`, `tokenizer.json`, and `metadata.json`. Preparation code, revisions, selection settings and file hashes are retained in `baselines/3b/`. Metadata includes the original machine paths as provenance; do not rewrite it or the token files during packaging. Attribution and license texts accompany the three archive parts separately so the seven baseline files remain byte-identical.
