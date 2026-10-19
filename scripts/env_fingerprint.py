"""Generate an environment fingerprint JSON to prove cross-platform replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or None if it doesn't exist."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_command_output(cmd: list[str]) -> str | None:
    """Run a CLI command and return trimmed stdout, or None on failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return None


def get_neo4j_metadata() -> dict[str, Any]:
    """Query Neo4j server, APOC, and GDS versions if reachable."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "changeme")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database="neo4j") as session:
            comp = session.run("CALL dbms.components() YIELD name, versions, edition RETURN versions[0] AS v, edition").data()
            apoc = session.run("RETURN apoc.version() AS v").data()
            gds = session.run("RETURN gds.version() AS v").data()
        driver.close()
        return {
            "reachable": True,
            "version": comp[0]["v"] if comp else None,
            "edition": comp[0]["edition"] if comp else None,
            "apoc_version": apoc[0]["v"] if apoc else None,
            "gds_version": gds[0]["v"] if gds else None,
        }
    except Exception as e:
        return {"reachable": False, "error": str(e)[:120]}


def generate_fingerprint() -> dict[str, Any]:
    """Collect system, lockfile, and toolchain versions into a structured dictionary."""
    repo_root = Path(__file__).resolve().parent.parent

    git_head = get_command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_branch = get_command_output(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    uv_version = get_command_output(["uv", "--version"])

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "uv_version": uv_version,
        "git_commit": git_head,
        "git_branch": git_branch,
        "uv_lock_sha256": file_sha256(repo_root / "uv.lock"),
        "pyproject_sha256": file_sha256(repo_root / "pyproject.toml"),
        "neo4j": get_neo4j_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment fingerprint JSON.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path for fingerprint JSON")
    args = parser.parse_args()

    data = generate_fingerprint()
    formatted = json.dumps(data, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted, encoding="utf-8")
        print(f"Fingerprint written to {args.output}")
    else:
        print(formatted)


if __name__ == "__main__":
    main()
