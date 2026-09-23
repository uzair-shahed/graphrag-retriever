# Neo4j GenAI Compatibility Notes

**Last modified:** 23 September 2026

**Purpose of this file:**
Record compatibility, installation, provider, and local-server facts that affect
Neo4j GenAI extraction and retrieval implementation.

**Files that use this file:**
`agent-coordination/tasks-goals-23-sept-2026.md` and the project setup/configuration
work should use this file before changing dependencies or Neo4j indexes.

**Caveats:**
This file records the local verification performed on 23 September 2026. It is
not a promise that future package releases or server upgrades retain the same
API. Provider credentials, model selection, and package extras remain project
decisions rather than checked-in values.

## Local Compatibility Baseline

| Item | Verified value |
|---|---|
| OS | macOS |
| Python | 3.14.3 |
| Neo4j server | Enterprise 2026.08.1 |
| Neo4j Python driver | 5.28.6 |
| `neo4j-graphrag` | 1.21.0 |
| SimpleKGPipeline import | Passed |
| Local Bolt smoke test | Passed |
| APOC procedures | Present |
| GenAI vector procedures | `genai.vector.encodeBatch`, `genai.vector.listEncodingProviders` present |
| Vector indexes currently configured | None; only lookup indexes were present during audit |

Official package documentation lists support for Neo4j 5.18.1+, Aura 5.18.0+,
and Neo4j 2026.01+, and Python 3.10 through 3.14.

## Installation And Dependency Notes

Base package:

```bash
pip install neo4j-graphrag
```

Provider extras are separate. Official documentation lists extras for OpenAI,
Vertex AI, Anthropic, Mistral, Cohere, Ollama, sentence-transformers, and
experimental KG Builder functionality. Select only the provider used by the
project. The local project currently declares the base package, not a provider
extra.

The package documentation states that its `nlp` extra (spaCy) is not supported
on Python 3.14 because of an upstream issue. Avoid the spaCy resolver on this
interpreter unless the project moves to Python 3.13 or earlier. The fuzzy-match
resolver has a separate RapidFuzz extra.

## Provider Selection Constraints

For extraction structured output, official documentation identifies OpenAI,
Vertex AI, and Anthropic implementations as supporting structured output in the
current package. The selected provider must be tested for:

- package extra availability;
- credentials loaded only from environment variables;
- structured-output support;
- model response limits;
- rate-limit behavior; and
- reproducible temperature/configuration.

For retrieval text queries, an embedder is required. A supplied query vector can
avoid a query-time embedder call, but indexed document vectors still need to be
created consistently.

## Server And Index Checks

Use these checks before an integration run:

```cypher
CALL dbms.components()
SHOW PROCEDURES YIELD name
WHERE name STARTS WITH 'genai.' OR name STARTS WITH 'apoc.'
RETURN name ORDER BY name
SHOW VECTOR INDEXES YIELD name, state, populationPercent, entityType,
  labelsOrTypes, properties, indexProvider
RETURN *
```

Expected project behavior:

- the server is reachable through the configured Bolt URI;
- required procedures are present;
- the target vector index is `ONLINE`;
- the index dimension matches the selected embedding model;
- the index label/property match the graph nodes returned by the retriever.

## Version-Specific Retrieval Notes

The local server is Neo4j 2026.08, so the current official documentation's
`SEARCH`-based vector filtering path is available. Vector indexes support
explicit dimensions from 1 to 4096 and `cosine` or `euclidean` similarity.
Creating an index with explicit dimensions is recommended so mismatches fail
clearly.

Neo4j 2026.08 also documents the `vector-2026.08` provider and binary
quantization defaults. Do not tune quantization or HNSW settings before a
baseline measurement; record any non-default values in the benchmark metadata.

## Open Verification Items

- Choose the LLM provider and install only its package extra.
- Choose the embedding provider and model/dimension.
- Run `SimpleKGPipeline` on the focused MuSiQue fixture.
- Verify the plugin's document/chunk metadata propagation.
- Create the project vector index only after the embedding dimension is known.
- Run one VectorRetriever query and one graph-aware retrieval query.
- Capture plugin and retriever errors as reproducible artifacts.

## Official Sources

- [GraphRAG for Python](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [GraphRAG Installation](https://neo4j.com/docs/neo4j-graphrag-python/current/#installation)
- [GraphRAG API](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Neo4j Installation](https://neo4j.com/docs/operations-manual/current/installation/)

Documentation consulted: 23 September 2026.
