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

In this project, GraphRAG begins with vector similarity over `Chunk` nodes.
Its `VectorCypherRetriever` query resolves source provenance through the
`FROM_DOCUMENT` relationship, then calls `apoc.path.subgraphAll` to traverse
the seed chunk's associated entity subgraph with no hop-depth ceiling. The
relationship filter is discovered from the live graph and excludes structural
relationships that could leave the seed subgraph. `--max-graph-nodes` is a
resource budget; evidence is marked incomplete if that budget prevents
returning the full component. This is a per-seed-chunk entity subgraph, not a
traversal that merges separate paragraph components across a whole MuSiQue
record. The graph nodes and relationships are preserved in evidence, while
Neo4j's managed answer prompt sees the retrieved chunk text rather than those
structured graph fields.

#### VectorRAG path

With `--retrieval-backend vectorrag`, the application calls
`VectorCypherRetriever` directly. In `vector` mode it resolves chunk
provenance without entity expansion; in `vector_cypher` mode it also returns
the seed-associated entity subgraph. It does not ask the retriever to generate
an answer. The local answer boundary builds a prompt containing the original
question and bounded chunk evidence, calls the configured LLM, parses its JSON
response, and checks that every citation refers to a returned chunk.

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
`--retrieval-mode vector|vector_cypher`, `--top-k`, and
`--max-context-chunks`. GraphRAG requires `vector_cypher`; VectorRAG supports
either mode. The command verifies Neo4j retrieval prerequisites before
execution, then uses the selected backend, shared answer contract, MuSiQue
correctness verifier, and artifact writer. A missing graph, provider failure,
malformed answer, or missing citation produces a nonzero exit status and does
not become a successful answer. Required environment settings are documented
in `.env.example`; both backends require a reachable Neo4j database and a
verified vector index.

### MuSiQue ingestion and query command reference

Run commands from the repository root. The examples use the locked project
environment; if it is already active, the corresponding `python -m ...`
invocations work too.

#### Load a bounded number of records

The batch loader reads a contiguous range of records by 1-indexed position in
the JSONL file. It loads every paragraph from each selected record; even a few
records can trigger many extraction and embedding calls. Start with `--dry-run`
to inspect the planned work before a live load.

```sh
# Plan the first three records without Neo4j or model-provider calls.
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --records 3 --dry-run

# Load records 4, 5, and 6 into the configured Neo4j database.
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --records 3 --start-record 4
```

Loader options:

| Flag | Required / default | Meaning |
| --- | --- | --- |
| `--dataset-path PATH` | Default: `dataset/musique/data/data/musique_full_v1.0_dev.jsonl` | Input MuSiQue JSONL file. |
| `--records N` | Required | Number of consecutive records to plan or load. |
| `--start-record N` | Default: `1` | 1-indexed starting record position in the file. |
| `--source-split LABEL` | Derived from filename | Override the dataset split label stored as provenance. |
| `--dry-run` | Off | Validate and count planned chunks without connecting to Neo4j or OpenRouter. |
| `--no-progress` | Progress enabled | Disable the progress bar during live ingestion. |
| `--manifest-path PATH` | `runs/ingestion-state/<split>-<first-16-SHA256-characters>.json` | Override the progress file for one dataset source. |
| `--fresh-manifest` | Off | Move the existing manifest aside and start new state; still reconcile Neo4j before extraction. |
| `-h`, `--help` | N/A | Show command help. |

The loader reports records and chunks planned, succeeded, and failed, plus
dead-lettered records. It exits nonzero when records are invalid or chunks fail.
`--dry-run` is not a preload: it does not write data. The live loader can incur
provider cost and writes to Neo4j. Run retrieval prerequisites and ensure the
vector index is online after ingestion.

#### Resume, stop, and local state

Live ingestion automatically checkpoints progress and resumes on the next run.
Use the same dataset path, split, record range, and manifest path to continue:

```sh
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --records 1 --no-progress
# After a stop, repeat the same command to reconcile and resume.
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --records 1 --no-progress
```

The manifest identifies a source by its complete file SHA-256 and split, then
tracks each record's expected stable paragraph chunk IDs. Record IDs alone do
not identify a source across files. The default filename includes the split
and first 16 checksum characters; the full checksum is checked inside the file.
Changing a dataset creates a separate default manifest. An explicit path that
contains a different checksum or split is rejected with exit `2`; choose another
path or deliberately use `--fresh-manifest`.

Chunk states are `pending`, `in_flight`, `verified`, `failed`, and
`needs_investigation`. Before extraction, state is flushed as `in_flight`.
After the pipeline finishes, Chunk provenance is synchronized from its Document
and read back from Neo4j; only a matching complete result becomes `verified`.
A record becomes loaded only when every expected paragraph is verified.
On every resume, Neo4j takes precedence over the manifest: matching complete
chunks skip extraction and embedding calls, absent chunks are ingested, and
partial or conflicting chunks are reported and skipped. A write that completed
before its checkpoint can therefore be detected without another provider call.

Press Ctrl-C once (or send SIGTERM) to request a clean stop. The current
pipeline call finishes and checkpoints before the loader stops at the next
chunk boundary. An early stop prints a JSON summary and exits `130`. If the
signal arrives during the final chunk, the batch may finish normally. Sending
the same signal again uses immediate default signal termination; there may be
no summary and the exit status depends on the signal/platform. An interrupted
`in_flight` entry is treated as pending on reload, reported in
`interrupted_in_flight_chunks`, and reconciled before any new extraction.

The summary includes `records_loaded`, newly succeeded chunks,
`chunks_skipped_already_verified`, failures, investigation reports, interrupted
chunk IDs, `stopped_early`, `chunks_not_attempted`, and `manifest_paths`.
Recovered interrupted entries can still yield success once all work verifies.
Handled batch-loader exit codes are:

| Code | Meaning |
| --- | --- |
| `0` | Successful batch, or successful dry-run validation. |
| `1` | Invalid/dead-lettered records, failed chunks, or unresolved partial/conflicting graph state. Unexpected runtime errors also generally exit nonzero. |
| `2` | Manifest error (including a checksum/split mismatch), or argument-parser error. |
| `130` | Requested clean stop before all chunks were processed. |

Progress is local operational state under gitignored `runs/`, retained after
completion so later runs can reuse it. Updates write a temporary file, flush
and fsync it, then atomically replace the manifest. `--fresh-manifest` moves
old state to `<manifest>.superseded-<timestamp>`; it does not reset graph data.
Deleting a manifest also leaves Neo4j unchanged; the next run reconciles it
again. Keep state backups when investigating interruptions. Dry runs neither
create manifests nor contact Neo4j or providers.

Partial/conflicting state remains `needs_investigation`, with node element IDs,
counts, and a reason in the summary. It is never automatically deleted,
overwritten, or repaired. Investigate the reported nodes before manually
cleaning them up; a later run ingests a chunk only after it classifies absent.
Failed chunks are checked again on the next run; there is currently no durable
retry queue or automatic bounded retry/backoff policy at the chunk level.

Use one ingestion writer at a time. Task 13's database-level idempotency guard
is not yet implemented. Verification currently expects one full-text Chunk per
paragraph: paragraphs exceeding `GRAPH_RAG_CHUNK_SIZE` (default `10000`) can be
split and reported partial. Splitter sizing remains Task 13 work.

Query `--preload` uses the same automatic manifest/reconciliation path, with one
default file per checksum and split when multiple sources are selected. It
does not expose the batch loader's manifest override/fresh-state flags or its
clean-stop signal handler. Query-only runs do not invoke ingestion.

#### Query or preload selected records

The QA command can accept either a custom `--question` or a MuSiQue
`--record-id`. These selectors are mutually exclusive. A record query reads the
stored question and provenance from the dataset JSONL; provide `--dataset-path`
one or more times, or configure `GRAPH_RAG_DATASET_PATHS` in `.env`.

```sh
# Query one existing MuSiQue record without ingesting it.
uv run --locked graphrag-query --record-id UNIQUE_RECORD_ID --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --retrieval-backend graphrag --artifact runs/record-query.json

# Preload that selected record first, then query it and write an artifact.
uv run --locked graphrag-query --record-id UNIQUE_RECORD_ID --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --preload --retrieval-backend graphrag --top-k 8 --max-context-chunks 3 --artifact runs/record-preloaded-query.json

# Query at most two selected records; artifacts are numbered per query.
uv run --locked graphrag-query --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --max-records 2 --retrieval-backend vectorrag --artifact runs/two-records.json
```

Query options:

| Flag | Required / default | Meaning |
| --- | --- | --- |
| `--question TEXT` | One of `--question` or `--record-id` | Run an external custom question; it has no MuSiQue record provenance and cannot be preloaded. |
| `--record-id ID` | One of `--question` or `--record-id` | Select the stored question for this exact MuSiQue ID from the supplied dataset files. The ID must be unique across those files. |
| `--dataset-path PATH` | `GRAPH_RAG_DATASET_PATHS` if omitted | JSONL dataset path; repeat the flag to search multiple files. |
| `--max-records N` | No limit | Bound a selected MuSiQue batch when `--record-id` is omitted. |
| `--preload` | Off | Ingest each selected MuSiQue record before its query. This writes to Neo4j and may make many provider calls; it requires a record-backed selection. |
| `--retrieval-backend {vectorrag,graphrag}` | `vectorrag` | Select vector-only-answering (`vectorrag`) or managed retrieval-plus-answering (`graphrag`). |
| `--retrieval-mode {vector,vector_cypher}` | `vector` for VectorRAG; `vector_cypher` for GraphRAG | Select whether retrieval expands each seed into its entity subgraph. GraphRAG requires `vector_cypher`. |
| `--top-k N` | `5` | Vector seed/result count. Managed GraphRAG caps seed count at `min(top_k, max_context_chunks)` to keep answer and evidence bounds aligned. |
| `--max-context-chunks N` | `5` | Maximum distinct source chunks retained as answer evidence. |
| `--max-graph-nodes N` | `200` | Per-seed graph traversal resource budget in `vector_cypher` mode; it is not a hop limit. An exhausted budget must be represented as incomplete component evidence. |
| `--artifact PATH` | No artifact | Write a JSON run artifact; a multi-query run writes numbered sibling files. |
| `--progress` | Off | Show progress for requested preloading. |
| `-h`, `--help` | N/A | Show command help. |

If a record ID occurs more than once across the selected files, selection fails
as ambiguous rather than choosing an occurrence. Select a source file with a
unique ID or make a one-record JSONL input containing the intended occurrence.
The command does not preload unless `--preload` is explicitly present. A
successful response with `preload: true` indicates preloading was requested and
completed sufficiently for QA to proceed; inspect the live loader output for
the chunk-level counts and failures.

The hidden `--skip-prerequisite-check` option is intended only for internal
testing and is deliberately omitted from the public flag list. Normal runs
should leave prerequisite checks enabled.
