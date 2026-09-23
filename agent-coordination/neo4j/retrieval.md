# Neo4j GenAI Retrieval

**Last modified:** 23 September 2026

**Purpose of this file:**
Summarize the official Neo4j GraphRAG Python retrieval APIs and the decisions
needed for this project's VectorRAG and graph-aware retrieval paths.

**Files that use this file:**
The retrieval tasks in `agent-coordination/tasks.md`, Contract 5 and later
retrieval implementations should use these notes when configuring indexes,
retrievers, GraphRAG generation, filters, and evidence formatting.

**Caveats:**
Neo4j vector search is approximate nearest-neighbor search. Retrieval results,
scores, and filtering behavior must be benchmarked on the installed server.
The notes do not claim that retrieval is implemented in this repository.
Hybrid and automatic tool routing remain outside the current core goal unless
explicitly re-enabled.

## Retrieval Layers

The official package separates:

- a retriever that returns records/context;
- an optional LLM that generates an answer; and
- `GraphRAG`, which performs retrieval, prompt augmentation, and generation.

This separation fits the project contracts: retrieval should return inspectable
chunks, entities, paths, scores, and provenance before answer generation.

## VectorRetriever

`VectorRetriever` searches a named Neo4j vector index. It accepts either:

- `query_vector`, which avoids an embedder call; or
- `query_text`, which requires an embedder.

Important inputs include `top_k`, `effective_search_ratio`, optional metadata
`filters`, `return_properties`, and an optional result formatter. Results expose
the matched node and similarity score. Use a result formatter or a retrieval
query when the answer/reflection layer needs stable citation metadata.

The query-time embedding model must match the index's embedding dimension and
semantic model. Record the model name, dimensions, similarity function, and
index name in every benchmark configuration.

## VectorCypherRetriever

`VectorCypherRetriever` performs vector seed retrieval and then executes a
configured Cypher retrieval query. The query receives:

- `node`: the node returned by vector search;
- `score`: its similarity score; and
- optional `query_params` supplied by the caller.

This is the package-native way to expand vector hits into graph context. Return
explicit properties and source references from the retrieval query rather than
returning opaque nodes when claim-level citations are required.

## Text2CypherRetriever

`Text2CypherRetriever` converts a natural-language question into Cypher using an
LLM and executes the generated query. It does not require an embedding model.
The generated query is not guaranteed to be valid, so treat it as a bounded,
read-only retrieval path and validate failures before answer generation. Supply
an explicit schema and examples where possible.

## GraphRAG

`GraphRAG(retriever, llm)` combines a retriever with an answer-generation LLM.
The `search` method supports retriever configuration such as `top_k`, can return
retrieved context, and can return a fallback when no context is found. The
project should enable context return and preserve it in the execution trace so
reflection can inspect the actual evidence.

## Index Creation

The package exposes helpers such as `create_vector_index`. Neo4j's documented
vector index requirements include:

- a unique index name;
- a node label or relationship type;
- one embedding property;
- an explicit vector dimension, recommended for clear mismatch failures; and
- `cosine` or `euclidean` similarity.

The local Neo4j server is 2026.08. On Neo4j 2026.01+, filterable properties can
be included in the index for in-index filtering through the `SEARCH` clause.
The index is not usable until its state is `ONLINE`; inspect it with
`SHOW VECTOR INDEXES`.

Example shape:

```python
from neo4j_graphrag.indexes import create_vector_index

create_vector_index(
    driver,
    name="musique_chunk_embedding",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,
    similarity_fn="cosine",
    filterable_properties=["source_split"],
)
```

The actual dimensions and model must be selected together. Do not copy `1536`
unless the chosen embedder produces that dimension.

## Vector Search Caveats

- Vector search is approximate nearest-neighbor search, not exact ranking.
- Similarity scores are meaningful within a result set; do not compare raw
  scores across unrelated retrieval systems without normalization.
- On Neo4j 2026.01+, simple filters can use in-index `SEARCH` filtering when the
  properties are declared in the vector index.
- Unsupported filter operators may fall back to a slower procedure or brute
  force path; measure this behavior.
- The older `db.index.vector.queryNodes` and relationship procedure paths are
  deprecated in newer Neo4j versions in favor of `SEARCH`.

## Retrieval Modes For This Project

| Mode | Package path | Current project role |
|---|---|---|
| Vector RAG | `VectorRetriever` + `GraphRAG` | Required baseline after index setup |
| Vector plus graph expansion | `VectorCypherRetriever` | Candidate for graph-enhanced retrieval |
| Structured Cypher | `Text2CypherRetriever` | Use only with bounded, validated, read-only queries |
| Hybrid vector/full-text | `HybridRetriever` | Deferred unless approved |
| Tool-selected routing | `ToolsRetriever` | Deferred routing extension |

## Evidence Contract

Every retrieval result used by this project should preserve:

```text
trace_id
retriever_type
index_name
query_text or query_vector metadata
top_k
similarity score
chunk/entity/path identifiers
source_split
record_id
chunk_id
source citation
```

Use the package's `retrieval_query` and `result_formatter` hooks to return
stable evidence records. Do not let the final answer depend on hidden retriever
state or on an unrecorded graph query.

## Official Sources

- [Neo4j GraphRAG RAG User Guide](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
- [GraphRAG API](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)

Documentation consulted: 23 September 2026.
