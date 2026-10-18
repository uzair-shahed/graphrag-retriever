"""CLI outcomes and real-process signal/resume tests without live services."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import AsyncMock

import pytest

from graphrag_pipeline.ingestion import load_musique_dataset as loader
from graphrag_pipeline.ingestion.progress_manifest import ManifestError


@pytest.mark.parametrize("outcome,exit_code", [("success", 0), ("failed", 1), ("manifest_error", 2), ("stopped", 130)])
def test_cli_exit_codes_and_summary(monkeypatch, capsys, outcome, exit_code):
    summary = loader.LoadSummary("fixture.jsonl", "fixture", 1, 1, 1)
    if outcome == "failed":
        summary.chunks_failed = [{"chunk_id": "c1"}]
    if outcome == "stopped":
        summary.stopped_early = True
    run = AsyncMock(return_value=summary)
    if outcome == "manifest_error":
        run.side_effect = ManifestError("checksum mismatch")
    monkeypatch.setattr(loader, "load_dataset", run)
    monkeypatch.setattr(loader.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["loader", "--records", "1", "--no-progress", "--manifest-path", "custom.json", "--fresh-manifest"])
    if exit_code:
        with pytest.raises(SystemExit) as error:
            loader.main()
        assert error.value.code == exit_code
    else:
        loader.main()
    payload = json.loads(capsys.readouterr().out)
    assert ("error" in payload) == (outcome == "manifest_error")
    assert run.call_args.kwargs["manifest_path"] == Path("custom.json")
    assert run.call_args.kwargs["fresh_manifest"] is True


# This child uses the real CLI, load_dataset, run_requests and manifest code.
# Only the graph/provider boundary is fake; graph state persists across exits.
CHILD = r'''
import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path.cwd() / 'tests/unit'))
from test_load_resume import FakeWorld, four_chunks
from graphrag_pipeline.ingestion import load_musique_dataset as loader

root = Path(sys.argv[1])
stopping = sys.argv[2] == 'stop'
world = FakeWorld()
graph_path = root / 'graph.json'
if graph_path.exists():
    world.graph = json.loads(graph_path.read_text())
pipeline = world.pipeline()
original_run = pipeline.run_async

async def run_async(**kwargs):
    chunk_id = kwargs['document_metadata']['chunk_id']
    with (root / 'calls.jsonl').open('a') as handle:
        handle.write(json.dumps(chunk_id) + '\n')
    if stopping and chunk_id == 'musique:r2:paragraph:0':
        (root / 'ready').write_text('in flight')
        while not (root / 'release').exists():
            await asyncio.sleep(0.01)
    result = await original_run(**kwargs)
    graph_path.write_text(json.dumps(world.graph))
    return result

pipeline.run_async = run_async
with (
    patch.object(loader, 'collect_requests', return_value=(four_chunks(), [], 2)),
    patch.object(loader.Neo4jSettings, 'from_env', return_value=MagicMock()),
    patch.object(loader, 'model_settings_from_env', return_value=MagicMock()),
    patch.object(loader, 'Neo4jConnection'),
    patch.object(loader, 'build_simple_kg_pipeline', return_value=pipeline),
    patch.object(loader, 'classify_chunk', side_effect=world.classify),
    patch.object(loader, 'synchronize_chunk_provenance'),
):
    sys.argv = ['loader', '--records', '2', '--no-progress', '--manifest-path', str(root / 'manifest.json')]
    loader.main()
'''


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery; CLI outcome tests run on Windows")
@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("interrupt_again", [False, True])
def test_signal_boundary_and_new_process_resume(tmp_path, stop_signal, interrupt_again):
    repo = Path(__file__).resolve().parents[2]
    process = subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path), "stop"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "child did not reach in-flight boundary"
            time.sleep(0.01)
        assert json.loads((tmp_path / "manifest.json").read_text())["chunks"]["musique:r2:paragraph:0"]["status"] == "in_flight"
        process.send_signal(stop_signal)
        # Wait for the handler's flushed message before releasing the pipeline.
        import select
        assert select.select([process.stdout], [], [], 20)[0], "stop handler did not respond"
        assert process.stdout.readline().startswith("Stop requested:")
        if interrupt_again:
            process.send_signal(stop_signal)
        else:
            (tmp_path / "release").write_text("finish current chunk")
        stdout, stderr = process.communicate(timeout=20)
        if interrupt_again:
            assert process.returncode == -stop_signal, stderr
            assert not stdout.strip()  # immediate termination need not emit JSON
        else:
            assert process.returncode == 130, stderr
            summary = json.loads(stdout)
            assert summary["records_loaded"] == 1
            assert summary["chunks_succeeded"] == 3
            assert summary["chunks_not_attempted"] == 1
            assert summary["stopped_early"] is True
        persisted = json.loads((tmp_path / "manifest.json").read_text())
        expected_statuses = ["in_flight", "pending", "verified", "verified"] if interrupt_again else ["pending", "verified", "verified", "verified"]
        assert sorted(c["status"] for c in persisted["chunks"].values()) == expected_statuses
        resumed = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path), "resume"], cwd=repo, capture_output=True, text=True, timeout=20)
        assert resumed.returncode == 0, resumed.stderr
        summary = json.loads(resumed.stdout)
        assert summary["records_loaded"] == 2
        assert summary["chunks_skipped_already_verified"] == (2 if interrupt_again else 3)
        assert summary["chunks_succeeded"] == (2 if interrupt_again else 1)
        assert summary["interrupted_in_flight_chunks"] == (["musique:r2:paragraph:0"] if interrupt_again else [])
        calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        expected_calls = ["musique:r1:paragraph:0", "musique:r1:paragraph:1", "musique:r2:paragraph:0"]
        if interrupt_again:
            expected_calls.append("musique:r2:paragraph:0")
        assert calls == expected_calls + ["musique:r2:paragraph:1"]
        persisted = json.loads((tmp_path / "manifest.json").read_text())
        for chunk_id, chunk in persisted["chunks"].items():
            assert chunk["status"] == "verified"
            assert chunk["attempts"] == (2 if interrupt_again and chunk_id == "musique:r2:paragraph:0" else 1)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
