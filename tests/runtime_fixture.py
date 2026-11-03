"""Tiny temporary tokenizer for offline transport/factory tests only.

Never writes to the live tokenizer cache or estimates production model usage.
"""
from pathlib import Path


def write_test_tokenizer(folder: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = Path(folder) / "test-tokenizer.json"
    tokenizer.save(str(path))
    return path
