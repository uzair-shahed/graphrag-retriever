"""Compare two environment fingerprint JSON files to validate cross-platform replication."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def compare_fingerprints(path_a: Path, path_b: Path) -> bool:
    """Compare two environment fingerprints, returning True if matching."""
    if not path_a.is_file():
        print(f"Error: Fingerprint file not found: {path_a}", file=sys.stderr)
        return False
    if not path_b.is_file():
        print(f"Error: Fingerprint file not found: {path_b}", file=sys.stderr)
        return False

    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))

    critical_fields = [
        ("Python Version", "python_version"),
        ("uv.lock SHA-256", "uv_lock_sha256"),
        ("pyproject.toml SHA-256", "pyproject_sha256"),
    ]

    mismatches: list[str] = []

    for label, key in critical_fields:
        val_a = a.get(key)
        val_b = b.get(key)
        if val_a != val_b:
            mismatches.append(f"  - {label} mismatch: '{val_a}' vs '{val_b}'")

    neo4j_a = a.get("neo4j", {})
    neo4j_b = b.get("neo4j", {})
    if neo4j_a.get("reachable") and neo4j_b.get("reachable"):
        for subkey in ("version", "apoc_version", "gds_version"):
            va = neo4j_a.get(subkey)
            vb = neo4j_b.get(subkey)
            if va != vb:
                mismatches.append(f"  - Neo4j {subkey} mismatch: '{va}' vs '{vb}'")

    # Informational differences (allowed between WSL and macOS)
    if a.get("machine") != b.get("machine"):
        print(f"Info: Architecture difference noted (allowed): {a.get('machine')} vs {b.get('machine')}")
    if a.get("system") != b.get("system"):
        print(f"Info: OS difference noted (allowed): {a.get('system')} vs {b.get('system')}")

    if mismatches:
        print("FAIL: Critical replication parameters do not match:")
        for m in mismatches:
            print(m)
        return False

    print("PASS: Both environments match across all critical replication parameters.")
    return True


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_fingerprints.py <fingerprint_a.json> <fingerprint_b.json>")
        sys.exit(2)

    success = compare_fingerprints(Path(sys.argv[1]), Path(sys.argv[2]))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
