#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer for the MiniFin-QA LM corpus.

Example:
    python train_tokenizer.py --files data/train.txt --output_dir tokenizer --vocab_size 9000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tokenizers import ByteLevelBPETokenizer


DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Byte-Level BPE tokenizer.")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        required=True,
        help="Text files used for tokenizer training (e.g., data/train.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tokenizer"),
        help="Directory to store tokenizer artifacts (default: ./tokenizer).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=9000,
        help="Tokenizer vocabulary size (default: 9000).",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum token occurrence required to be included in the vocab (default: 2).",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase input text before training (default: False).",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=DEFAULT_SPECIAL_TOKENS,
        help="List of special tokens to seed the tokenizer vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: List[str] = [str(f) for f in args.files if f.exists()]
    if not files:
        raise SystemExit("No valid training files supplied.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer(lowercase=args.lowercase)
    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens,
    )

    tokenizer.save_model(str(args.output_dir))
    tokenizer.save(str(args.output_dir / "tokenizer.json"))

    config = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "lowercase": args.lowercase,
        "special_tokens": args.special_tokens,
    }
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Tokenizer saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
