"""Keep durable retry journals isolated, including child-process test runs."""

import pytest


@pytest.fixture(autouse=True)
def isolated_retry_state(tmp_path, monkeypatch):
    monkeypatch.setenv("INGESTION_RETRY_STATE_DIR", str(tmp_path / "retry-state"))
