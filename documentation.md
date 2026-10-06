### Issues with incomplete JSONs
- I have noticed that if the 'OPENROUTER_LLM_MAX_TOKENS' option in .env is not high enough, the JSON output produced by the LLM can be cut-off and therefore is invalid. It is important to keep it high enough to avoid any errors.

### How does neo4j's KGPipeline chunk text from 1 paragraph?
The splitter is configured generically because SimpleKGPipeline also accepts entire files and long documents. It needs to break arbitrary input into manageable units for:

- LLM context limits;
- one extraction call per chunk;
- chunk embeddings;
- independent graph extraction and processing.

For my current loader, the important consequence is:
    one MuSiQue paragraph
        -> one ExtractionRequest
        -> potentially multiple Neo4j TextChunk objects
        -> one extraction LLM call per TextChunk

So a paragraph of 8,000 characters may produce roughly three extraction chunks, and therefore roughly three extraction calls. With OpenRouter automatic schema extraction still enabled, the pipeline may also perform a schema-generation call for the input before those extraction calls.

### GraphRAG retrieval prerequisites

Before running retrieval, verify the Neo4j and vector-index contract from the
repository root:

```text
.venv/bin/python -m graphrag_pipeline.retrieval_prerequisites
```

The check confirms connectivity, Neo4j version and edition, the installed
`neo4j-graphrag` API, a non-empty `Document`/`Chunk` graph, and an `ONLINE`
vector index targeting `Chunk.embedding`. The expected index name, similarity
function, and embedding dimension are configured through
`GRAPH_RAG_VECTOR_INDEX_NAME`, `GRAPH_RAG_VECTOR_SIMILARITY`, and
`GRAPH_RAG_VECTOR_DIMENSIONS`. The dimension must match the embedding provider
used during extraction. Optional `--chunk-id` and `--record-id` arguments run
the existing read-only provenance verifier for one persisted result.

The installed 1.21.0 API uses `VectorRetriever(driver, index_name, ...)` and
`GraphRAG.search(query_text, retriever_config=..., return_context=True)`. The
verification command does not create or mutate indexes; it fails instead when
the configured index is absent, not online, aimed at a different property,
or configured with incompatible dimensions or similarity.

### GraphRAG retrieval adapter

The application retrieval boundary is in
`src/graphrag_pipeline/retrieval.py`. After prerequisite verification, build a
retriever with `build_retriever()` and submit a validated `QueryRequest` to
`retrieve()`. The adapter supports `vector` and `vector_cypher` modes, passes
the configured `top_k` to Neo4j, bounds the returned context, and preserves
chunk and source provenance in `EvidenceChunk` values. Empty results remain
explicitly empty; provider failures and malformed records raise
`RetrievalError` rather than becoming successful answers.

### Grounded answer boundary

The answer-generation boundary is in `src/graphrag_pipeline/answer.py`. It
accepts the exact `QueryRequest` question and bounded `RetrievalResult`, then
invokes the configured OpenRouter or Ollama `neo4j-graphrag` LLM wrapper. The
prompt contains only the question and returned evidence, with each chunk's
`chunk_id` exposed as a citation candidate; MuSiQue gold answers, supporting
fact labels, and decomposition fields are never sent to the provider.

The provider must return JSON with exactly `answer` and `citations` fields.
Citations must refer to returned evidence chunk IDs. Empty retrieval produces
`insufficient_evidence` without an LLM call. Provider exceptions, malformed
JSON, empty answers, missing citations, and citations for unknown chunks
produce `failed`; a `succeeded` `QueryResponse` always has a non-empty answer
and citations while preserving request and evidence provenance.

### MuSiQue correctness verification

The MuSiQue-specific checks are in `src/graphrag_pipeline/musique_verification.py`.
They run after retrieval and answer generation and keep evaluation labels out
of all model-visible inputs. For each returned chunk, the verifier compares the
adapter-generated paragraph index and retrieved text with the selected record's
paragraphs. For labeled records it separately reports whether every supporting
paragraph was retrieved; this is an evaluation signal, not a generation hint.

When a read-only Neo4j connection is supplied, the verifier also checks each
returned `Chunk` ID, record ID, and text against the graph and retains its
source metadata as a graph trace. Missing or mismatched chunks are reported as
incomplete rather than accepted as graph-backed evidence. `write_run_artifact`
stores the original request, retrieval evidence, answer/citations, correctness
report, configuration, timings, and redacted errors as deterministic JSON.