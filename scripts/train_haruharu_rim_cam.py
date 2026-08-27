#!/usr/bin/env python3
"""Train the HaruharuRIM CAM (Code Automation Model) from JavaScript files.

This intentionally lightweight trainer turns repository JavaScript into a
small, deterministic bag-of-token/ngram model and stores the learned weights as
JSON artifacts. It has no third-party dependencies so it can run on GitHub
Actions without setup beyond Python itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|==={0,1}|!==|=>|&&|\|\||[{}()[\].,;:+\-*/%<>!=?]")


def iter_js_files(source_dir: Path) -> Iterable[Path]:
    """Yield JavaScript files below source_dir in stable order."""
    return sorted(path for path in source_dir.rglob("*.js") if path.is_file())


def tokenize(code: str) -> list[str]:
    """Tokenize JavaScript-ish text into stable CAM tokens."""
    return TOKEN_PATTERN.findall(code)


def ngrams(tokens: list[str], max_order: int) -> Counter[str]:
    """Create token and n-gram counts up to max_order."""
    counts: Counter[str] = Counter()
    for order in range(1, max_order + 1):
        for index in range(0, max(0, len(tokens) - order + 1)):
            gram = " ".join(tokens[index : index + order])
            counts[f"{order}g:{gram}"] += 1
    return counts


def stable_hash(value: str, buckets: int) -> int:
    """Map a feature into a deterministic bucket."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % buckets


def train(source_dir: Path, output_dir: Path, buckets: int, max_order: int) -> dict[str, object]:
    files = list(iter_js_files(source_dir))
    if not files:
        raise FileNotFoundError(f"No JavaScript files found under {source_dir}")

    feature_counts: Counter[str] = Counter()
    file_summaries: list[dict[str, object]] = []
    total_tokens = 0

    for path in files:
        code = path.read_text(encoding="utf-8")
        tokens = tokenize(code)
        counts = ngrams(tokens, max_order)
        feature_counts.update(counts)
        total_tokens += len(tokens)
        file_summaries.append(
            {
                "path": str(path),
                "bytes": len(code.encode("utf-8")),
                "tokens": len(tokens),
                "features": len(counts),
            }
        )

    bucket_weights = [0.0] * buckets
    total_features = sum(feature_counts.values())
    for feature, count in feature_counts.items():
        bucket = stable_hash(feature, buckets)
        # TF-style log scaling keeps the model compact while preserving signal.
        bucket_weights[bucket] += round(1.0 + math.log(count), 8)

    nonzero_weights = [weight for weight in bucket_weights if weight]
    model = {
        "brand": "HaruharuRIM CAM",
        "full_name": "HaruharuRIM Code Automation Model",
        "model_type": "hashed_js_ngram_code_model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "max_order": max_order,
        "buckets": buckets,
        "files": file_summaries,
        "totals": {
            "files": len(files),
            "tokens": total_tokens,
            "raw_feature_events": total_features,
            "unique_features": len(feature_counts),
            "nonzero_weight_buckets": len(nonzero_weights),
        },
        "weights": bucket_weights,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "haruharu-rim-cam-weights.json"
    summary_path = output_dir / "haruharu-rim-cam-summary.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: model[key] for key in ("brand", "full_name", "model_type", "created_at", "source_dir", "max_order", "buckets", "totals")}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model_path": str(model_path), "summary_path": str(summary_path), **summary}


def default_source_dir() -> Path:
    preferred = Path("sourcecode")
    if preferred.exists():
        return preferred
    return Path("sourcecodes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save HaruharuRIM CAM weights from JavaScript source.")
    parser.add_argument("--source-dir", type=Path, default=default_source_dir(), help="Directory containing JavaScript files (default: sourcecode, fallback: sourcecodes).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/haruharu-rim-cam"), help="Directory for saved CAM artifacts.")
    parser.add_argument("--buckets", type=int, default=256, help="Number of hashed weight buckets to save.")
    parser.add_argument("--max-order", type=int, default=3, help="Maximum token n-gram order.")
    args = parser.parse_args()

    result = train(args.source_dir, args.output_dir, args.buckets, args.max_order)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
