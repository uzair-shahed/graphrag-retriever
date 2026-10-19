"""Durable, atomic progress manifest for resumable MuSiQue batch ingestion.

One JSON file tracks one dataset source, identified by its file SHA-256 and
split label. It records every expected paragraph chunk of each requested
record and that chunk's status. A record is complete only when every expected
chunk is ``verified``. The manifest is local operational state: it is never
trusted over Neo4j on resume (see ``reconcile.py``) and a manifest for a
different source checksum is rejected rather than applied.

File lifecycle: created on the first live run for a source, rewritten
atomically after every state change, kept after completion so later runs skip
verified work, and safe to delete (the next run will reconcile against Neo4j).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from graphrag_pipeline.extraction.schemas import ExtractionRequest

MANIFEST_VERSION = 1
DEFAULT_STATE_DIR = Path("runs/ingestion-state")

PENDING = "pending"
IN_FLIGHT = "in_flight"
VERIFIED = "verified"
FAILED = "failed"
NEEDS_INVESTIGATION = "needs_investigation"


class ManifestError(RuntimeError):
    """Raised when a manifest is unreadable or internally inconsistent."""


class ManifestSourceMismatch(ManifestError):
    """Raised when a manifest belongs to a different dataset checksum or split."""


def default_manifest_path(source_sha256: str, source_split: str, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    """Return the default manifest path for one source checksum and split."""
    safe_split = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in source_split)
    return state_dir / f"{safe_split}-{source_sha256[:16]}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class ProgressManifest:
    """In-memory view of one manifest file with atomic persistence."""

    def __init__(self, path: Path, source_sha256: str, source_split: str, data: dict[str, Any] | None = None):
        self.path = path
        self.source_sha256 = source_sha256
        self.source_split = source_split
        self._data: dict[str, Any] = data or {
            "version": MANIFEST_VERSION,
            "source_sha256": source_sha256,
            "source_split": source_split,
            "records": {},
            "chunks": {},
        }
        self.interrupted_in_flight: list[str] = []
        self._superseded_failures: dict[str, Any] = {}

    @classmethod
    def open(
        cls, path: Path, source_sha256: str, source_split: str, *, fresh: bool = False
    ) -> "ProgressManifest":
        """Load a manifest, or create an empty one if none exists.

        Args:
            path: Manifest file location.
            source_sha256: Checksum of the dataset file being ingested.
            source_split: Split label of that source.
            fresh: If true, an existing file is moved aside (never deleted) and
                a new manifest is started.

        Raises:
            ManifestSourceMismatch: The file records a different checksum or split.
            ManifestError: The file is not a valid manifest.
        """
        superseded_failures = {}
        if path.exists() and fresh:
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ManifestError(f"cannot inspect manifest before preserving retry history: {error}") from error
            if isinstance(previous, dict) and previous.get("source_sha256") == source_sha256 and previous.get("source_split") == source_split:
                superseded_failures = {cid: chunk for cid, chunk in previous.get("chunks", {}).items()
                                       if chunk.get("status") in {FAILED, IN_FLIGHT} or chunk.get("interrupted")}
            path.replace(path.with_name(f"{path.name}.superseded-{time.time_ns()}"))
        if not path.exists():
            manifest = cls(path, source_sha256, source_split)
            manifest._superseded_failures = superseded_failures
            return manifest
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(f"manifest {path} is unreadable: {error}") from error
        if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
            raise ManifestError(f"manifest {path} has an unsupported format or version")
        if data.get("source_sha256") != source_sha256 or data.get("source_split") != source_split:
            raise ManifestSourceMismatch(
                f"manifest {path} was written for source sha256 {data.get('source_sha256')!r} "
                f"split {data.get('source_split')!r}, not {source_sha256!r} / {source_split!r}; "
                "use a different --manifest-path or --fresh-manifest"
            )
        manifest = cls(path, source_sha256, source_split, data)
        manifest._demote_in_flight()
        return manifest

    def _demote_in_flight(self) -> None:
        for chunk_id, chunk in self._data["chunks"].items():
            if chunk["status"] == IN_FLIGHT:
                chunk["status"] = PENDING
                chunk["interrupted"] = True
                self.interrupted_in_flight.append(chunk_id)

    def register(self, requests: Iterable[ExtractionRequest]) -> None:
        """Register expected chunks for each record; never downgrades known state."""
        by_record: dict[str, list[ExtractionRequest]] = {}
        for request in requests:
            by_record.setdefault(request.record_id, []).append(request)
        for record_id, record_requests in by_record.items():
            expected = sorted(request.chunk_id for request in record_requests)
            known = self._data["records"].get(record_id)
            if known is not None and known["expected_chunk_ids"] != expected:
                raise ManifestError(
                    f"record {record_id!r} expects different chunks than the manifest for the same source"
                )
            self._data["records"][record_id] = {
                "expected_chunk_ids": expected,
                "source_line": record_requests[0].provenance.source_line,
            }
            for request in record_requests:
                self._data["chunks"].setdefault(
                    request.chunk_id,
                    {"record_id": record_id, "status": PENDING, "attempts": 0},
                )
        self.flush()

    def legacy_failure_snapshot(self, chunk_id: str) -> dict[str, Any]:
        """Include retained failure counts when --fresh-manifest archived old state."""
        from copy import deepcopy
        return deepcopy(self._superseded_failures.get(chunk_id, self._data["chunks"][chunk_id]))

    def chunk_snapshot(self, chunk_id: str) -> dict[str, Any]:
        """Read legacy operational state without granting mutation access."""
        from copy import deepcopy
        return deepcopy(self._data["chunks"][chunk_id])

    def status(self, chunk_id: str) -> str:
        return self._data["chunks"][chunk_id]["status"]

    def _set(self, chunk_id: str, status: str, **extra: Any) -> None:
        chunk = self._data["chunks"][chunk_id]
        chunk["status"] = status
        chunk["updated_at"] = time.time()
        for key, value in extra.items():
            chunk[key] = value
        self.flush()

    def require_verification(self, chunk_id: str) -> None:
        """A new run must not count an old local success before reconciliation."""
        if self.status(chunk_id) == VERIFIED:
            self._set(chunk_id, PENDING)

    def mark_in_flight(self, chunk_id: str) -> None:
        chunk = self._data["chunks"][chunk_id]
        self._set(chunk_id, IN_FLIGHT, attempts=chunk.get("attempts", 0) + 1)

    def mark_verified(self, chunk_id: str) -> None:
        chunk = self._data["chunks"][chunk_id]
        chunk.pop("error", None)
        chunk.pop("investigation", None)
        chunk.pop("interrupted", None)
        self._set(chunk_id, VERIFIED)

    def mark_failed(self, chunk_id: str, error: dict[str, Any]) -> None:
        self._set(chunk_id, FAILED, error=error)

    def mark_needs_investigation(self, chunk_id: str, report: dict[str, Any]) -> None:
        self._set(chunk_id, NEEDS_INVESTIGATION, investigation=report)

    def is_record_complete(self, record_id: str) -> bool:
        record = self._data["records"].get(record_id)
        if record is None:
            return False
        return all(self._data["chunks"][chunk_id]["status"] == VERIFIED for chunk_id in record["expected_chunk_ids"])

    def chunk_ids_with_status(self, status: str) -> list[str]:
        return sorted(chunk_id for chunk_id, chunk in self._data["chunks"].items() if chunk["status"] == status)

    def flush(self) -> None:
        """Persist the manifest atomically (temp file, fsync, rename)."""
        try:
            _atomic_write_json(self.path, self._data)
        except OSError as error:
            raise ManifestError(f"cannot persist manifest {self.path}: {error}") from error


class ManifestStore:
    """Resolves the right manifest for each request's source checksum and split."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        state_dir: Path = DEFAULT_STATE_DIR,
        fresh: bool = False,
    ) -> None:
        self._manifest_path = manifest_path
        self._state_dir = state_dir
        self._fresh = fresh
        self._manifests: dict[tuple[str, str], ProgressManifest] = {}
        self._existing: dict[tuple[str, str], ProgressManifest | None] = {}

    def for_request(self, request: ExtractionRequest) -> ProgressManifest:
        key = (request.provenance.source_sha256, request.provenance.source_split)
        if key not in self._manifests:
            if self._manifest_path is not None and self._manifests:
                raise ManifestError(
                    "--manifest-path supports one source only, but the selection spans several"
                )
            path = self._manifest_path or default_manifest_path(key[0], key[1], self._state_dir)
            self._manifests[key] = ProgressManifest.open(path, key[0], key[1], fresh=self._fresh)
        return self._manifests[key]

    def existing_failure_snapshot(self, request: ExtractionRequest) -> dict[str, Any] | None:
        """Read old failures before fresh-state archiving can change the file."""
        key = (request.provenance.source_sha256, request.provenance.source_split)
        if key in self._manifests:
            manifest = self._manifests[key]
        else:
            if key not in self._existing:
                path = self._manifest_path or default_manifest_path(*key, self._state_dir)
                try:
                    self._existing[key] = ProgressManifest.open(path, *key) if path.exists() else None
                except ManifestSourceMismatch:
                    if not self._fresh:
                        raise
                    self._existing[key] = None
            manifest = self._existing[key]
        if manifest is None or request.chunk_id not in manifest._data["chunks"]:
            return None
        return manifest.legacy_failure_snapshot(request.chunk_id)

    def register(self, requests: list[ExtractionRequest]) -> None:
        grouped: dict[tuple[str, str], list[ExtractionRequest]] = {}
        for request in requests:
            grouped.setdefault(
                (request.provenance.source_sha256, request.provenance.source_split), []
            ).append(request)
        for group in grouped.values():
            self.for_request(group[0]).register(group)

    def manifests(self) -> list[ProgressManifest]:
        return list(self._manifests.values())

    def complete_record_ids(self, requests: list[ExtractionRequest]) -> set[str]:
        return {
            request.record_id
            for request in requests
            if self.for_request(request).is_record_complete(request.record_id)
        }
