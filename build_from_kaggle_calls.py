#!/usr/bin/env python3
"""
Utility script for building MiniFin-QA LM training corpora from finance transcripts.

Reads a mixture of .txt, .json/.jsonl, and .csv sources inside dataset_finance/,
performs light cleaning while preserving dialogue speaker turns, filters out short
documents, caps total size, and writes 90/10 character-wise splits to data/train.txt
and data/val.txt.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List


TRANSCRIPT_KEYS = (
    "transcript",
    "text",
    "content",
    "full_transcript",
    "body",
    "article",
    "document",
)

ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # byte order mark
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build finance LM corpus from transcripts.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing transcript sources (txt/json/csv).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data"),
        help="Directory to write train/val text files (default: ./data).",
    )
    parser.add_argument(
        "--target_mb",
        type=float,
        default=8.0,
        help="Approximate target corpus size in megabytes after cleaning (default: 8).",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=2000,
        help="Minimum number of characters required to keep a cleaned document.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Validation split fraction (character-wise).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document shuffling prior to concatenation.",
    )
    return parser.parse_args()


def clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_TAG_RE.sub(" ", text)
    for z in ZERO_WIDTH_CHARS:
        text = text.replace(z, "")
    text = text.replace("\xa0", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def load_txt(path: Path) -> Iterable[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    if cleaned:
        yield cleaned


def extract_from_json_obj(obj) -> Iterable[str]:
    if isinstance(obj, str):
        cleaned = clean_text(obj)
        if cleaned:
            yield cleaned
        return

    if isinstance(obj, dict):
        for key in TRANSCRIPT_KEYS:
            value = obj.get(key)
            if isinstance(value, str):
                cleaned = clean_text(value)
                if cleaned:
                    yield cleaned
        for value in obj.values():
            yield from extract_from_json_obj(value)
        return

    if isinstance(obj, list):
        for item in obj:
            yield from extract_from_json_obj(item)


def load_json(path: Path) -> Iterable[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        print(f"[warn] Skipping {path}: JSON decode error ({exc})", file=sys.stderr)
        return []
    return extract_from_json_obj(data)


def load_jsonl(path: Path) -> Iterable[str]:
    results: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for text in extract_from_json_obj(obj):
                    results.append(text)
    except OSError as exc:
        print(f"[warn] Could not open {path}: {exc}", file=sys.stderr)
    return results


def load_csv(path: Path) -> Iterable[str]:
    texts: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return texts
            candidate_columns = [c for c in reader.fieldnames if c.lower() in TRANSCRIPT_KEYS]
            if not candidate_columns:
                # Try fuzzy match
                lowered = {c.lower(): c for c in reader.fieldnames}
                for key in TRANSCRIPT_KEYS:
                    if key in lowered:
                        candidate_columns.append(lowered[key])
            if not candidate_columns:
                print(f"[warn] No transcript column detected in {path}", file=sys.stderr)
                return texts
            for row in reader:
                for column in candidate_columns:
                    value = row.get(column, "")
                    if isinstance(value, str) and value.strip():
                        cleaned = clean_text(value)
                        if cleaned:
                            texts.append(cleaned)
                            break
    except OSError as exc:
        print(f"[warn] Could not open {path}: {exc}", file=sys.stderr)
    return texts


def gather_documents(root: Path) -> List[str]:
    docs: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt":
            docs.extend(load_txt(path))
        elif suffix == ".json":
            docs.extend(load_json(path))
        elif suffix in {".jsonl", ".ndjson"}:
            docs.extend(load_jsonl(path))
        elif suffix == ".csv":
            docs.extend(load_csv(path))
    return docs


def trim_to_target(texts: List[str], target_bytes: int) -> List[str]:
    if target_bytes <= 0:
        return texts
    trimmed: List[str] = []
    running = 0
    for doc in texts:
        doc_bytes = len(doc.encode("utf-8"))
        if running + doc_bytes > target_bytes and trimmed:
            break
        trimmed.append(doc)
        running += doc_bytes
    return trimmed


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist.")

    raw_docs = gather_documents(input_dir)
    filtered_docs = [doc for doc in raw_docs if len(doc) >= args.min_chars]
    if not filtered_docs:
        raise SystemExit("No documents found after filtering; check input directory and min_chars.")

    random.Random(args.seed).shuffle(filtered_docs)
    target_bytes = int(args.target_mb * (1024**2))
    trimmed_docs = trim_to_target(filtered_docs, target_bytes)

    corpus = "\n\n".join(trimmed_docs).strip() + "\n"
    total_chars = len(corpus)
    split_idx = int(total_chars * (1 - args.val_ratio))
    split_idx = max(split_idx, 1)
    train_text = corpus[:split_idx]
    val_text = corpus[split_idx:]
    if len(val_text.strip()) == 0 and len(trimmed_docs) > 1:
        val_text = trimmed_docs[-1]
        train_text = "\n\n".join(trimmed_docs[:-1]) + "\n"

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_path = args.outdir / "train.txt"
    val_path = args.outdir / "val.txt"

    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    stats = {
        "total_documents": len(filtered_docs),
        "used_documents": len(trimmed_docs),
        "total_chars": total_chars,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
    }
    print("=== Corpus build summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
