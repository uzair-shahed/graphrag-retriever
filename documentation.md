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
uv run --locked python -m graphrag_pipeline.retrieval_prerequisites
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

#### Record-filtered retrieval (MuSiQue)

A query with `--record-id` constrains vector search to that record through the
vector index itself, which requires the index to declare `record_id` filterable
and every `Chunk` to carry the `record_id` of its `Document`. `graphrag-query`
checks both before running and stops with an explicit error if either is
missing; it never falls back to filtering after the search. A graph loaded
before this requirement existed is repaired with:

```text
python -m graphrag_pipeline.vector_index_setup          # report only
python -m graphrag_pipeline.vector_index_setup --apply  # backfill + recreate
```

`--apply` drops and recreates the index with the same dimensions and similarity
(embeddings are not recomputed), so vector search is unavailable until the new
index is `ONLINE`. Records ingested with the current loader already carry
`Chunk.record_id`. `retrieval_prerequisites --filter-record-id ID` reports the
filter state for already-loaded records.

### GraphRAG and VectorRAG retrieval adapter

The application retrieval boundary is in
`src/graphrag_pipeline/retrieval.py`. After prerequisite verification, build a
retriever with `build_retriever()` and submit a validated `QueryRequest` to
`retrieve()`. Select the high-level backend with
`--retrieval-backend vectorrag|graphrag`. `vectorrag` supports `vector` and
`vector_cypher` modes and sends the retrieved context to the local answer
boundary. `graphrag` wraps the same Neo4j retriever with `GraphRAG.search()`;
its generated answer and returned context are normalized without a second LLM
call. Both paths pass the configured `top_k`, bound the returned context, and
preserve chunk and source provenance in `EvidenceChunk` values. Empty results
remain explicit; provider failures and malformed records raise `RetrievalError`
rather than becoming successful answers.

#### MuSiQue query disambiguation

MuSiQue questions can be deliberately ambiguous because the same word may
refer to several different entities in the record. For example, the question
`Who is the spouse of the Green performer?` contains only the word `Green`,
while the record includes paragraphs about the colour green, Green Party,
Grant Green, and the album `Green (Steve Hillage album)`.

For MuSiQue requests, the retrieval adapter therefore creates a separate
embedding-search query from the original question plus a small set of
record-title hints. It prefers specific parenthesized titles such as
`Green (Steve Hillage album)` and falls back to the first few unique titles
when no such title is available. The original question is never changed in
the `QueryRequest`, answer prompt, response, or artifact; the extra text is
only a retrieval aid. The adapter also filters candidates by the selected
MuSiQue `record_id`, over-fetches candidates, and removes duplicate `chunk_id`
rows before applying the final context limit.

This helps the vector search identify the intended meaning of an ambiguous
question without sending `is_supporting`, the gold answer, or other
evaluation labels to the model. It is a disambiguation aid rather than a
guarantee that the correct supporting paragraphs will rank first. The
correctness report records whether the retrieved evidence actually covers the
gold supporting paragraphs.

### The answering pipeline in plain language

When the command receives a question, it performs the following stages:

1. **Select and validate the question.** A MuSiQue request keeps the exact
    question, record ID, source file, source line, paragraph titles, and trace
    ID. A custom question receives no MuSiQue provenance.
2. **Optionally preload the record.** With `--preload`, each selected MuSiQue
    paragraph is sent through `SimpleKGPipeline`. The pipeline extracts graph
    entities and relationships, creates chunk embeddings, and writes the
    resulting graph data to Neo4j. Without `--preload`, the existing graph is
    queried as-is.
3. **Build the retrieval query.** The application preserves the original
    question. For MuSiQue, it may add selected source-title hints only for the
    embedding search, as described above.
4. **Retrieve graph evidence.** Neo4j's vector index compares the question
    embedding with stored chunk embeddings. The application removes duplicate
    chunks, keeps the bounded context, and retains each chunk's text and
    provenance.
5. **Generate a grounded answer.** The selected answer path receives the
    question and retrieved evidence. It is instructed to answer only from that
    evidence, cite evidence chunk IDs, and avoid guessing when the evidence is
    insufficient.
6. **Validate and record the result.** The response contract checks the
    answer and citations. For MuSiQue, a separate correctness step compares
    retrieved text with the original record and reports supporting-paragraph
    coverage. The command writes the request, evidence, answer, citations,
    correctness report, and errors to the JSON artifact.

#### GraphRAG path

With `--retrieval-backend graphrag`, the application builds a Neo4j
`GraphRAG` object around the configured vector retriever. `GraphRAG.search()`
performs retrieval, assembles the retrieved text into its own context, and
calls the GraphRAG-compatible LLM to generate an answer. The application
receives both the generated answer and the retriever context, then normalizes
that context into `EvidenceChunk` values and uses the returned chunk IDs as
citations. The application does not make a second answer-model call.

In this project, GraphRAG still begins with vector similarity. The graph
structure is available through the Neo4j retriever, but the configured
retrieval query currently returns the matched chunk properties; it does not
perform a custom multi-hop Cypher traversal for every question.

#### VectorRAG path

With `--retrieval-backend vectorrag`, the application calls
`VectorRetriever` or `VectorCypherRetriever` directly. It receives the
retrieved chunks but does not ask that retriever to generate an answer. The
local answer boundary then builds a prompt containing the original question
and bounded evidence, calls the configured LLM, parses its JSON response, and
checks that every citation refers to a returned chunk.

The practical difference is therefore the answer boundary: GraphRAG owns both
retrieval and answer generation, while VectorRAG separates those responsibilities
and lets the application own the final grounded-answer prompt and validation.
Both paths depend on the same Neo4j vector index, embedding model, chunk
provenance, and evidence limits.

#### Ollama JSON Schema output

Ollama generation uses `src/graphrag_pipeline/ollama_json_schema.py` to send
Pydantic schemas through Ollama's native chat `format` field on both synchronous
answer calls and asynchronous extraction calls. Answer generation uses the
`GeneratedAnswer` schema; `SimpleKGPipeline` extraction uses Neo4j GraphRAG's
`Neo4jGraph` schema. Ollama embeddings are unaffected because they return
vectors, not generated text.

The adapter keeps structural fields constrained while allowing graph labels,
relationship types, and property-map keys to vary. Returned content is still
validated by the application and GraphRAG. Schema-constrained generation does
not prove that extracted facts are correct, that relationship endpoints
reference existing nodes, or that answer citations support the answer.

### Grounded answer boundary

The answer-generation boundary is in `src/graphrag_pipeline/answer.py`. It
accepts the exact `QueryRequest` question and bounded `RetrievalResult`, then
invokes the configured OpenRouter or Ollama `neo4j-graphrag` LLM wrapper. The
prompt contains only the question and returned evidence, with each chunk's
`chunk_id` exposed as a citation candidate; MuSiQue gold answers, supporting
fact labels, and decomposition fields are never sent to the provider.

For VectorRAG, the provider must return JSON with exactly `answer` and
`citations` fields. For GraphRAG, `GraphRAG.search()` supplies the answer and
the adapter cites the returned context chunk IDs because the 1.21.0 result
model does not expose generated citations. Empty retrieval produces
`insufficient_evidence` without an LLM call. Provider exceptions, malformed
JSON, empty answers, missing citations, and citations for unknown chunks
produce `failed`; a `succeeded` `QueryResponse` always has a non-empty answer
and citations while preserving request, backend, and evidence provenance.

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

### Minimal query command

The minimal end-to-end command is implemented in
`src/graphrag_pipeline/query_command.py`. It can be run from the repository
root as either `python -m graphrag_pipeline.query_command` or, after installing
the project, `graphrag-query`.

A custom question is explicitly external and receives no MuSiQue provenance:

```text
graphrag-query --question "Who acquired Northwind Labs?" --artifact runs/custom.json
```

For MuSiQue questions, configure `GRAPH_RAG_DATASET_PATHS` in `.env`, or pass
one or more `--dataset-path` values, then select a stable record ID:

```text
graphrag-query --record-id fixture_valid_multi_hop_001 --artifact runs/record.json
```

Without `--record-id`, the command processes all configured records up to
`--max-records`. It preserves each stored question unchanged and writes
numbered artifacts for a batch. `--preload` is off by default; when supplied,
the command delegates to the existing bounded ingestion path before querying
each selected MuSiQue record. Without it, no ingestion or re-extraction is
performed.

Retrieval is bounded with `--retrieval-backend vectorrag|graphrag`,
`--retrieval-mode vector|vector_cypher` (the mode applies to VectorRAG),
`--top-k`, and `--max-context-chunks`. The command verifies Neo4j retrieval
prerequisites before execution, then uses the selected backend, shared answer
contract, MuSiQue correctness verifier, and artifact writer. A missing graph,
provider failure, malformed answer, or missing citation produces a nonzero exit
status and does not become a successful answer. Required environment settings
are documented in `.env.example`; both backends require a reachable Neo4j
database and a verified vector index.