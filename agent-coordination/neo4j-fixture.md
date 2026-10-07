# Neo4j Setup And Smoke Test

The root [README](../README.md) is the clean-clone macOS quick start and the
canonical live GraphRAG procedure. This page records the narrower Neo4j
connection and test workflow.

## Package Files

- `pyproject.toml`

  Defines the Python project, dependencies, package discovery, and pytest configuration. The distribution name is `graphrag-pipeline`, while the Python import name is `graphrag_pipeline`.

- `.env.example`

  Template for local Neo4j credentials and connection limits:

  - URI and credentials
  - connection timeout
  - transaction timeout
  - total retry window

- `src/graphrag_pipeline/__init__.py`

  Marks `graphrag_pipeline` as the application package.

- `src/graphrag_pipeline/config.py`

  Defines `Neo4jSettings`. It loads Neo4j configuration from environment variables or `.env`. It only parses configuration; it does not connect to Neo4j.

- `src/graphrag_pipeline/storage/__init__.py`

  Marks `storage` as the storage-layer package.

- `src/graphrag_pipeline/storage/neo4j_connection.py`

  Defines `Neo4jConnection`, which:

  - creates and closes the Neo4j driver;
  - verifies connectivity;
  - runs read and write transactions;
  - applies a connection timeout;
  - applies a transaction timeout;
  - retries transient Neo4j failures with bounded attempts and backoff;
  - does not retry invalid Cypher, authentication failures, or other non-transient errors.

## Tests

- `tests/unit/test_config.py`

  Tests environment parsing and default timeout values without requiring Neo4j.

- `tests/unit/test_neo4j_connection.py`

  Uses a fake Neo4j driver to test commits, rollbacks, retry behavior, access mode, timeouts, and driver cleanup. It does not contact a real database.

- `tests/integration/test_neo4j_connection_smoke.py`

  Connects to a real Neo4j instance, creates a temporary test node, reads it, and deletes it. It skips automatically when credentials or Neo4j are unavailable.

- `src/graphrag_pipeline.egg-info/`

  Generated packaging metadata from the editable install. It is not application code and is ignored by Git through `*.egg-info/`.

## Import Path

The installed package is intended to be imported as:

```python
from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
```

The tests use the installed package import path, `graphrag_pipeline`, rather
than including the source directory name in the import. The editable package
installation performed by `uv sync` keeps runtime imports consistent.

## Run Unit Tests

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked python -m pytest tests/unit -v
```

These tests do not need Neo4j running.

## Configure And Run The Real Neo4j Smoke Test

Create the ignored local configuration file:

```bash
cp .env.example .env
```

Edit `.env` and replace at least:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-real-password
NEO4J_DATABASE=neo4j
```

After starting Neo4j, run:

```bash
uv run --locked python -m pytest tests/integration/test_neo4j_connection_smoke.py -v
```

To run all tests:

```bash
uv run --locked python -m pytest tests -v
```

## Current Scope

The connection smoke test checks Bolt connectivity and a temporary write/read/delete
cycle. It does not initialize the GraphRAG graph or vector index. Follow the root
README for fixture extraction, index setup, prerequisite verification, and a
live query.
