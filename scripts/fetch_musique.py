"""Download and verify official MuSiQue v1.0 dataset files with SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Official checksums and record counts from verified audit copy
EXPECTED_FILES = {
    "musique_ans_v1.0_dev.jsonl": {
        "count": 2417,
        "sha256": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    },
    "musique_ans_v1.0_train.jsonl": {
        "count": 19938,
        "sha256": "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a",
    },
    "musique_full_v1.0_dev.jsonl": {
        "count": 4834,
        "sha256": "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a",
    },
    "musique_full_v1.0_train.jsonl": {
        "count": 39876,
        "sha256": "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034",
    },
}

DEFAULT_TARGET_DIR = Path("dataset/musique/data/data")
DEFAULT_MANIFEST_PATH = Path("dataset/MANIFEST.json")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl_records(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def verify_file(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    """Verify record count and SHA-256 for a given file."""
    if not path.is_file():
        return False, "File not found"
    actual_count = count_jsonl_records(path)
    if actual_count != expected["count"]:
        return False, f"Count mismatch: expected {expected['count']}, got {actual_count}"
    actual_hash = compute_file_sha256(path)
    if actual_hash != expected["sha256"]:
        return False, f"SHA-256 mismatch: expected {expected['sha256']}, got {actual_hash}"
    return True, f"Verified (count={actual_count}, sha256={actual_hash[:8]}...)"


def fetch_and_verify(
    target_dir: Path,
    manifest_path: Path,
    source_dir: Path | None = None,
) -> bool:
    """Fetch files from source_dir, verify checksums, and write manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: dict[str, Any] = {}
    all_ok = True

    for filename, meta in EXPECTED_FILES.items():
        dest = target_dir / filename

        # If not present in target, copy from source_dir if provided
        if not dest.is_file() and source_dir:
            candidate = source_dir / filename
            if candidate.is_file():
                print(f"Copying {filename} from {source_dir}...")
                shutil.copy2(candidate, dest)

        if not dest.is_file():
            print(f"Error: {filename} missing in {target_dir}", file=sys.stderr)
            all_ok = False
            continue

        ok, msg = verify_file(dest, meta)
        print(f"[{'OK' if ok else 'FAIL'}] {filename}: {msg}")
        if not ok:
            all_ok = False

        manifest_records[filename] = {
            "record_count": meta["count"],
            "sha256": meta["sha256"],
            "path": dest.as_posix(),
            "verified": ok,
        }

    manifest = {
        "dataset": "MuSiQue v1.0",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": "Trivedi et al., MuSiQue: Multihop Questions via Single-hop Question Composition, TACL 2022",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_records,
        "all_verified": all_ok,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify MuSiQue dataset files.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Target storage directory")
    parser.add_argument("--source-dir", type=Path, default=None, help="Local directory containing pre-downloaded files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path for output manifest JSON")
    args = parser.parse_args()

    success = fetch_and_verify(
        target_dir=args.target_dir,
        manifest_path=args.manifest,
        source_dir=args.source_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
