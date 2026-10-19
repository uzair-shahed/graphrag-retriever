"""Generate seeded, hop-stratified MuSiQue evaluation slice.

Usage:
    uv run python scripts/make_slice.py --seed 20261018 --size 500 --tune 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
DEFAULT_OUTPUT = Path("eval/slices/ans_dev_mvp_v1.json")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_slice(
    input_path: Path,
    output_path: Path,
    seed: int = 20261018,
    size: int = 500,
    tune_size: int = 200,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    source_sha256 = compute_file_sha256(input_path)
    records_by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hop = int(rec["id"].split("hop")[0])
            records_by_hop[hop].append(rec)

    total_records = sum(len(v) for v in records_by_hop.values())
    if total_records == 0:
        raise ValueError(f"No records found in {input_path}")

    # Proportional targets for total slice
    hops = sorted(records_by_hop.keys())
    targets: dict[int, int] = {}
    remaining = size
    for i, h in enumerate(hops):
        if i == len(hops) - 1:
            targets[h] = remaining
        else:
            count = round(size * len(records_by_hop[h]) / total_records)
            targets[h] = count
            remaining -= count

    # Proportional targets for tune split
    tune_targets: dict[int, int] = {}
    remaining_tune = tune_size
    for i, h in enumerate(hops):
        if i == len(hops) - 1:
            tune_targets[h] = remaining_tune
        else:
            count = round(tune_size * len(records_by_hop[h]) / total_records)
            tune_targets[h] = count
            remaining_tune -= count

    rng = random.Random(seed)
    all_selected: list[dict[str, Any]] = []
    tune_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    records_manifest: list[dict[str, Any]] = []

    for hop in hops:
        pool = list(records_by_hop[hop])
        rng.shuffle(pool)
        selected = pool[: targets[hop]]
        all_selected.extend(selected)

        hop_tune = selected[: tune_targets[hop]]
        hop_test = selected[tune_targets[hop] : targets[hop]]

        tune_records.extend(hop_tune)
        test_records.extend(hop_test)

        for r in hop_tune:
            records_manifest.append({"id": r["id"], "hop": hop, "split": "tune"})
        for r in hop_test:
            records_manifest.append({"id": r["id"], "hop": hop, "split": "test"})

    unique_paragraphs = len(
        {(p["title"], p["paragraph_text"]) for r in all_selected for p in r.get("paragraphs", [])}
    )

    tune_ids = [r["id"] for r in tune_records]
    test_ids = [r["id"] for r in test_records]

    hop_distribution = {
        "total": {str(h): targets[h] for h in hops},
        "tune": {str(h): tune_targets[h] for h in hops},
        "test": {str(h): targets[h] - tune_targets[h] for h in hops},
    }

    slice_data: dict[str, Any] = {
        "name": output_path.stem,
        "seed": seed,
        "size": size,
        "tune_size": tune_size,
        "test_size": size - tune_size,
        "source_file": str(input_path),
        "source_sha256": source_sha256,
        "unique_paragraphs": unique_paragraphs,
        "hop_distribution": hop_distribution,
        "tune_ids": tune_ids,
        "test_ids": test_ids,
        "records": records_manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(slice_data, f, indent=2)

    # Also save gold JSONL files for the full slice, tune split, and test split
    gold_dir = output_path.parent
    full_gold_path = gold_dir / f"{output_path.stem}_gold.jsonl"
    tune_gold_path = gold_dir / f"{output_path.stem}_tune_gold.jsonl"
    test_gold_path = gold_dir / f"{output_path.stem}_test_gold.jsonl"

    with full_gold_path.open("w", encoding="utf-8") as f:
        for r in all_selected:
            f.write(json.dumps(r) + "\n")

    with tune_gold_path.open("w", encoding="utf-8") as f:
        for r in tune_records:
            f.write(json.dumps(r) + "\n")

    with test_gold_path.open("w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    return slice_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT, help="Path to input MuSiQue JSONL")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT, help="Path to output slice JSON")
    parser.add_argument("--seed", type=int, default=20261018, help="Random seed for sampling and splits")
    parser.add_argument("--size", type=int, default=500, help="Total number of questions in slice")
    parser.add_argument("--tune", type=int, default=200, help="Number of questions in tune split")

    args = parser.parse_args()
    slice_data = create_slice(
        input_path=args.input_file,
        output_path=args.output_file,
        seed=args.seed,
        size=args.size,
        tune_size=args.tune,
    )
    print(f"Generated evaluation slice at {args.output_file}:")
    print(f"  Total questions: {slice_data['size']}")
    print(f"  Tune split: {slice_data['tune_size']}, Test split: {slice_data['test_size']}")
    print(f"  Hops distribution: {slice_data['hop_distribution']['total']}")
    print(f"  Unique paragraphs: {slice_data['unique_paragraphs']}")


if __name__ == "__main__":
    main()
