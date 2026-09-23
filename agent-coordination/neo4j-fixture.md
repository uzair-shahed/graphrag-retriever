# Neo4j Fixture And Connection Package

The current implementation is a small Python package for testing a Neo4j connection. It is not yet a complete GraphRAG application and does not have a command-line entry point.

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

The tests use the installed package import path, `graphrag_pipeline`, rather than including the source directory name in the import. VS Code/Pylance must use the project virtual environment and treat `src` as an analysis path; that configuration is stored in `.vscode/settings.json`. This keeps editor resolution and runtime imports consistent.

## Run Unit Tests

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests/unit -v
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
source .venv/bin/activate
python -m pytest tests/integration/test_neo4j_connection_smoke.py -v
```

To run all tests:

```bash
python -m pytest tests -v
```

## Current Scope

The current implementation's executable behavior is exposed through the tests and the `Neo4jConnection` class. There is currently no `main.py`, CLI command, or end-to-end GraphRAG pipeline yet.
