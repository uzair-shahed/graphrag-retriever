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
| Neo4j database | `neo4j`; credentials are not recorded here |
| GraphRAG Python package | `neo4j-graphrag==1.21.0` |
| `SimpleKGPipeline` import | Passed |
| Existing connection smoke test | Passed |
| Neo4j installation | Neo4j Desktop |
| APOC installation | Installed through Neo4j Desktop |
| Local database state | Running during verification |
| Fixture reset | Permitted for isolated fixture runs |

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
| APOC/vector plugin configuration | Desktop-managed; procedures verified | APOC and GenAI vector procedures are available; no manual configuration file is currently required for this fixture path. |

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

## Provider Verification Results

The local `.env` contains the selected OpenRouter configuration. Secrets are
not recorded in this report.

| Capability | Result | Evidence |
|---|---|---|
| Required OpenRouter configuration fields | Passed | Local redacted configuration check found no missing required fields. |
| LLM catalog entry `openai/gpt-4o-mini` | Passed | Live OpenRouter structured-output catalog query found the configured model. |
| LLM structured-output catalog flag | Passed | The live catalog advertises `structured_outputs` and `response_format`. |
| Strict JSON Schema request | Passed | OpenRouter accepted the strict schema request and returned the requested `openai/gpt-4o-mini` model. |
| Embedding catalog entry `openai/text-embedding-3-small` | Passed | Live OpenRouter embedding catalog query found the configured model. |
| Embedding request | Passed | Live request returned a numeric vector. |
| Measured embedding dimension | Passed | Returned dimension was 1,536, matching configuration. |
| OpenRouter fallback behavior | Explicitly configured | Active configuration permits fallbacks, while `require_parameters=true` restricts providers to those supporting the requested schema. |

The updated paid low-cost endpoint completed the structured-output probe
without a rate-limit error. The returned model matched the configured model.
The project should still record the returned model in every run artifact when
fallbacks are enabled.

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

- Record the exact local Neo4j installation method and APOC/plugin configuration. **Passed:** Neo4j Desktop, APOC installed through Desktop, local database running, isolated fixture reset permitted.
- Select and document an LLM provider and embedding provider. **Passed:** OpenRouter models are recorded in `agent-coordination/openrouter/openrouter-instructions.md` and the local `.env`.
- Add the required provider extra to the dependency configuration, if needed. **Passed:** `neo4j-graphrag[openai]` is declared in `pyproject.toml`; installation verification remains part of the environment check.
- Confirm that the selected provider works with Python 3.14.3 and structured output if structured extraction is required. **Passed:** catalog support, strict JSON Schema request, and returned-model verification succeeded.
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
.venv/bin/python -m pytest tests/integration/test_neo4j_connection_smoke.py -v -rs
.venv/bin/python -m pip check
```