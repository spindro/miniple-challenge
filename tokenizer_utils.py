from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# This is a Hugging Face conversion of the nanochat 32,768-token BPE tokenizer
# trained on ~2B characters of the ClimbMix pretraining corpus. We pin the
# repository revision so every student uses exactly the same tokenization.
NANOCHAT_TOKENIZER_REPO = "crellis/d18-9tpp-hf-sft"
NANOCHAT_TOKENIZER_REVISION = "79e88e310d774ee11a1dd3c154a639f0cc21bcb2"
NANOCHAT_TOKENIZER_FILENAME = "tokenizer.json"
EXPECTED_VOCAB_SIZE = 32_768
BOS_TOKEN = "<|bos|>"


def validate_nanochat_tokenizer(tok: Tokenizer) -> None:
    vocab_size = tok.get_vocab_size()
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected nanochat vocab size {EXPECTED_VOCAB_SIZE:,}, got {vocab_size:,}."
        )

    bos_id = tok.token_to_id(BOS_TOKEN)
    if bos_id is None:
        raise RuntimeError(f"Tokenizer is missing required document delimiter {BOS_TOKEN!r}.")

    # The nanochat tokenizer is byte-level/reversible. This catches corrupted or
    # accidentally substituted tokenizer files early.
    probe = "Hello, tokenizer! 12345 — café 🌍"
    ids = tok.encode(probe, add_special_tokens=False).ids
    decoded = tok.decode(ids, skip_special_tokens=False)
    if decoded != probe:
        raise RuntimeError(
            "Tokenizer round-trip check failed. The downloaded/local tokenizer "
            "does not appear to be the expected nanochat artifact."
        )


def load_fixed_nanochat_tokenizer(
    local_dir: Path,
    *,
    repo_id: str = NANOCHAT_TOKENIZER_REPO,
    revision: str = NANOCHAT_TOKENIZER_REVISION,
    local_json: Path | None = None,
) -> Tuple[Tokenizer, Path]:
    """Load the fixed 32,768-token nanochat tokenizer.

    Resolution order:
      1) explicit --tokenizer-json path, if supplied;
      2) tokenizer.json already present in local_dir;
      3) download the pinned tokenizer.json from Hugging Face once and copy it
         into local_dir.

    """

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = local_dir / NANOCHAT_TOKENIZER_FILENAME

    if local_json is not None:
        source = Path(local_json)
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file does not exist: {source}")
        if source.resolve() != frozen_path.resolve():
            shutil.copy2(source, frozen_path)
    elif not frozen_path.exists():
        print(
            "Downloading fixed nanochat tokenizer once: "
            f"{repo_id}@{revision[:12]}"
        )
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=NANOCHAT_TOKENIZER_FILENAME,
            revision=revision,
        )
        shutil.copy2(cached, frozen_path)

    tok = Tokenizer.from_file(str(frozen_path))
    validate_nanochat_tokenizer(tok)
    return tok, frozen_path
