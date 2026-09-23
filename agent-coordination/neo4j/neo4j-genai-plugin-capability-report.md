# Neo4j GenAI Plugin Capability Report

**Last modified:** 23 September 2026

**Purpose of this file:**
Record the verified local Neo4j server, Neo4j GraphRAG Python package, and
`SimpleKGPipeline` capability baseline for the plugin-first extraction goal.

**Files that use this file:**
`agent-coordination/tasks-goals-23-sept-2026.md` uses this report as evidence
for Active Task 1. `pyproject.toml` declares the verified Python package
dependency. The local Neo4j configuration and smoke test provide the connection
behavior used by this report.

**Caveats:**
This report verifies connectivity and API availability only. It does not prove
that `SimpleKGPipeline` has successfully extracted the MuSiQue fixture, that the
selected LLM or embedding provider is configured, that provenance is preserved,
or that the generated graph satisfies Contracts 1-4. The Knowledge Graph
Builder feature is documented by Neo4j as experimental, so its API and behavior
may change.

## Verification Date And Environment

| Item | Verified value |
|---|---|
| Operating system | macOS |
| Python interpreter | `.venv/bin/python` |
| Python version | 3.14.3 |
| Neo4j Python driver | 5.28.6 |
| Neo4j server edition | Enterprise |
| Neo4j server version | 2026.08.1 |
| Cypher versions | 5, 25 |
| Neo4j database | Configured local database; credentials are not recorded here |
| GraphRAG Python package | `neo4j-graphrag==1.21.0` |
| `SimpleKGPipeline` import | Passed |
| Existing connection smoke test | Passed |

## Server Capability Results

The local server was queried through the repository's existing bounded
`Neo4jConnection` wrapper.

| Capability | Result | Interpretation |
|---|---|---|
| Bolt connectivity | Passed | The configured local server accepts the project's Python connection. |
| `CALL dbms.components()` | Passed | Server is Neo4j Enterprise `2026.08.1`. |
| APOC procedures | Present | APOC procedures are available to the local database. |
| `genai.vector.encodeBatch` | Present | The server exposes a GenAI vector encoding procedure. |
| `genai.vector.listEncodingProviders` | Present | The server exposes GenAI vector provider discovery. |
| `SimpleKGPipeline` server procedure | Not present | `SimpleKGPipeline` is not a server procedure; it is supplied by the Python package. |
| Vector indexes | Not yet configured | `SHOW INDEXES` returned only built-in node and relationship lookup indexes. |
| APOC/vector plugin configuration | Not fully documented | The installed procedures were observed, but configuration files and provider settings still need recording. |

## Python API Results

The official Neo4j GraphRAG Python package was installed in the active virtual
environment and imported successfully:

```python
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
```

The verified constructor accepts the expected integration inputs:

- an LLM implementation;
- a Neo4j driver;
- an embedder;
- an optional graph schema;
- `from_file` input mode;
- `on_error` behavior;
- entity-resolution configuration;
- lexical graph configuration;
- an optional Neo4j database name.

The installed API also exposes `LLMEntityRelationExtractor`. The package
documentation states that Knowledge Graph Builder is experimental and that
structured output is available only for supported LLM implementations. The
selected LLM must therefore be decided and tested before the fixture run.

## Official Documentation Checked

- Neo4j GraphRAG for Python installation and compatibility documentation:
  `https://neo4j.com/docs/neo4j-graphrag-python/current/`
- Knowledge Graph Builder and `SimpleKGPipeline` documentation:
  `https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html`
- Pipeline configuration and asynchronous execution documentation:
  `https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_pipeline.html`

The documentation identifies `neo4j-graphrag` as the maintained package and
describes `neo4j-genai` as its deprecated predecessor.

## Still Required Before Marking Active Task 1 Complete

- Record the exact local Neo4j installation method and APOC/plugin configuration.
- Select and document an LLM provider and embedding provider.
- Add the required provider extra to the dependency configuration, if needed.
- Confirm that the selected provider works with Python 3.14.3 and structured
  output if structured extraction is required.
- Initialize `SimpleKGPipeline` with the local Neo4j driver without exposing
  secrets.
- Run the focused MuSiQue fixture through the pipeline.
- Inspect the resulting labels, relationships, lexical graph, entity-resolution
  behavior, and source-reference properties.
- Create and verify any vector indexes required by the selected retrieval path.
- Record plugin run output and provenance mapping evidence.

## Commands Used

```bash
.venv/bin/python -m pytest tests/integration/test_neo4j_connection_smoke.py -v -rs
.venv/bin/python -m pip install 'neo4j-graphrag'
.venv/bin/python -c 'from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline; print(SimpleKGPipeline)'
```