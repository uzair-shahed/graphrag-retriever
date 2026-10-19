"""Live tests verifying server-side transaction timeout behavior against Neo4j."""

from __future__ import annotations

import os
import pytest
from neo4j.exceptions import ClientError

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.extraction.pipeline import GraphWriteError, ensure_writer_succeeded
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def test_live_transaction_timeout_terminates_long_running_write() -> None:
    """A write transaction exceeding its timeout is terminated with a timeout error."""
    settings = Neo4jSettings.from_env()
    try:
        conn = Neo4jConnection(settings)
        conn.verify_connectivity()
    except Exception as exc:
        if not os.environ.get("REQUIRE_LIVE"):
            pytest.skip(f"Live Neo4j instance not available at {settings.uri}: {exc}")
        raise

    with conn:
        driver = conn.connect()
        with driver.session() as session:
            with pytest.raises(ClientError) as exc_info:
                # 2-second client override for the transaction timeout
                with session.begin_transaction(timeout=2.0) as tx:
                    tx.run("CALL apoc.util.sleep(5000)")
                    tx.commit()

            assert "TransactionTimedOut" in str(exc_info.value) or "terminated" in str(exc_info.value)

    # In neo4j-graphrag SimpleKGPipeline writer, a ClientError results in status="FAILURE".
    # Verify that ensure_writer_succeeded (from M1.1) raises GraphWriteError.
    synthetic_timed_out_result = {
        "writer": {
            "status": "FAILURE",
            "error": "TransactionTimedOutClientConfiguration: The transaction has been terminated.",
        }
    }
    with pytest.raises(GraphWriteError, match="Neo4j writer did not succeed"):
        ensure_writer_succeeded(synthetic_timed_out_result)
