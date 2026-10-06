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