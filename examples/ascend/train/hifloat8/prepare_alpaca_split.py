#!/usr/bin/env python3
"""Create and checksum the fixed phase-1 train/eval JSONL split."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()

    rows = args.input.read_text(encoding="utf-8").splitlines()
    if args.train_size <= 0 or args.eval_size <= 0 or args.train_size + args.eval_size > len(rows):
        raise ValueError("train-size and eval-size must be positive and fit within the input")
    for line_number, row in enumerate(rows, 1):
        parsed = json.loads(row)
        if not isinstance(parsed.get("messages"), list):
            raise ValueError(f"line {line_number} does not contain a messages list")

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    train_indices = indices[:args.train_size]
    eval_indices = indices[args.train_size:args.train_size + args.eval_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for name, selected in (("train", train_indices), ("eval", eval_indices)):
        path = args.output_dir / f"{name}.jsonl"
        path.write_text("".join(f"{rows[index]}\n" for index in selected), encoding="utf-8")
        outputs[name] = {
            "path": str(path),
            "rows": len(selected),
            "indices": selected,
            "sha256": sha256(path),
        }

    manifest = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "seed": args.seed,
        "unused_rows": len(rows) - args.train_size - args.eval_size,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
