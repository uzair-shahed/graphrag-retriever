# Neo4j GraphRAG Question-to-Answer Goal

**Date:** 27 September 2026

**Last modified:** 6 October 2026

**Purpose of this file:**

Define the minimum implementation tasks required to add question answering to
the now-verified Neo4j extraction path. The immediate aim is deliberately
small: optionally preload selected data into Neo4j, retrieve graph-backed
evidence, and return either a grounded answer or an explicit insufficient-
evidence result with inspectable provenance.

This is a focused successor to the completed extraction work in
`agent-coordination/tasks-goals-23-sept-2026.md`. Record ingestion and
plugin-created graph provenance are current capabilities, not future tasks.
This document now concentrates on the smallest useful query-to-answer slice
and its later MuSiQue-specific expansion.

**Files that use this file:**

The broader roadmap in `agent-coordination/tasks.md` supplies the surrounding
milestones. The active extraction and graph contracts are recorded in
`agent-coordination/data-contracts.md`; this goal adds the next retrieval and
answer boundary without reviving the archived hand-built extraction design.

**Caveats:**

The local package is `neo4j-graphrag 1.21.0`. It exposes
`neo4j_graphrag.generation.graphrag.GraphRAG.search()` and Neo4j-backed
retrievers, but the repository does not yet configure or verify the retrieval
index, query harness, answer contract, or answer-generation run. Those are the
remaining implementation items. The first answer demonstration must not be
described as complete until the returned evidence is proven to come from the
graph. Preloading is controlled explicitly by `--preload` and must never happen
implicitly.

## Goal

Implement and verify one minimal local query flow:

```text
question (custom or selected from MuSiQue JSONL)
  -> validated query request
  -> optional record preload through SimpleKGPipeline
  -> Neo4j GraphRAG retrieval from the available graph
  -> answer generation using returned graph evidence
  -> answer or insufficient-evidence result with sources and trace
```

The first demonstrator may use either a custom question or the exact
`question` field from one selected record in a configured MuSiQue JSONL file.
The query command must accept a particular MuSiQue record ID, locate that
record in the configured JSONL files, and route its unchanged `question` value
to the question-answering system. The dataset file or files used for record
selection must be configured through the `.env` file, for example with
`GRAPH_RAG_DATASET_PATHS`; there is no fixed or implicit dataset filename.
When no specific record ID is provided, the system must select all records from
one or more configured dataset files and process them as a bounded batch. The
configured file list and selection mode must be retained in the run metadata.
The query command must support an explicit `--preload` flag. When the flag is
active, each selected MuSiQue record must be loaded through the existing
bounded ingestion path and `SimpleKGPipeline` before its question is sent to
the QA system. When the flag is absent, the command must only query the
available graph and must not invoke ingestion or re-extraction. In either mode,
graph availability and evidence provenance must be validated. The record's supporting-fact labels remain
evaluation-only and must not be sent to retrieval or answer generation as
model-visible input.

Every implementation step in this goal must include reasonable deterministic
tests that verify its behavior and failure handling. Tests must cover both
each feature in isolation and the composed question-selection, retrieval,
verification, and answer-generation path together. The combined tests must
prove that the selected question, graph evidence, provenance, citations, and
final status remain correctly connected across the full workflow.

## Current Repository State

As of 6 October 2026:

| Area | Status | Evidence |
|---|---|---|
| MuSiQue record-to-paragraph chunk adapter | Complete | `src/graphrag_pipeline/ingestion/record_adapter.py` and unit tests validate source identity, paragraph text, and provenance. |
| Plugin input boundary | Complete | `src/graphrag_pipeline/extraction/plugin_input.py` excludes gold labels and normalizes metadata required by GraphRAG 1.21.0. |
| Neo4j GenAI extraction | Complete for the verified slice | `src/graphrag_pipeline/extraction/pipeline.py` and `run_fixture.py` initialize and run `SimpleKGPipeline`. |
| Bounded official-record ingestion | Complete for ingestion only | `src/graphrag_pipeline/ingestion/load_musique_dataset.py` supports record windows, dry runs, progress, and failure summaries. It does not answer questions. |
| Graph/provenance verification | Complete for extraction evidence | `src/graphrag_pipeline/verification.py` and `tests/unit/test_graph_verifier.py` verify persisted `Document`/`Chunk` provenance read-only. |
| GraphRAG retrieval | Missing in the application | `neo4j-graphrag 1.21.0` exposes `VectorRetriever`, `VectorCypherRetriever`, and `GraphRAG`, but no repository adapter invokes them. |
| Answer generation | Missing in the application | No query request/response models, grounded answer boundary, citations, or insufficient-evidence result exists yet. |
| Retrieval-test command | Missing | No separate command can query an already-loaded graph without re-running ingestion. |
| Retrieval index and live QA run | Unverified | The extraction graph contains chunk embeddings, but index name, dimensions, similarity, retrieval result shape, and a live answer run are not recorded. |

The existing `documentation.md` also notes that a paragraph can be split into
multiple plugin text chunks. The query contract must therefore cite the
actual returned graph chunks, not assume one Neo4j chunk per MuSiQue paragraph.

## Architecture Decision

Maximally use Neo4j's GenAI pipeline and managed capabilities for the active
path. Extraction through `SimpleKGPipeline` is already the verified write path.
For answering, the application should use the installed GraphRAG
`VectorRetriever` or `VectorCypherRetriever` with `GraphRAG.search()` rather
than implementing local vector search or a second retrieval framework.

The active implementation priorities are:

- Neo4j GenAI `SimpleKGPipeline` for paragraph-to-graph extraction (verified).
- Neo4j-managed vector configuration and indexing for graph-backed retrieval
  (to configure and verify).
- `neo4j_graphrag.generation.graphrag.GraphRAG.search()` for retrieval,
  context construction, and answer generation (API available in 1.21.0).
- Neo4j's supported LLM/embedding integrations for the answer call, using the
  existing OpenRouter or Ollama settings.
- Explicit Cypher verification only as a bounded evidence and correctness
  check, not as a replacement for the required Neo4j GraphRAG path.
- A thin local Python layer for MuSiQue record selection, input validation,
  run metadata, failure handling, and the final structured response contract.

Do not build a parallel hand-written extractor, vector search implementation,
or general-purpose retrieval framework for this goal. If a Neo4j capability is
unavailable in the exact local version, record the limitation and add the
smallest compatibility adapter needed to preserve the Neo4j-managed design.

## Minimum Scope

The first successful demonstration needs only:

- one custom question or one question selected from a configured MuSiQue file;
- an explicit `--preload` mode that loads selected MuSiQue records before QA,
  plus a default query-only mode for available graph data;
- a validated query request retaining question origin and a trace ID;
- a Neo4j GraphRAG retrieval call returning graph-backed evidence from the
  existing database;
- an answer generated from the returned evidence;
- citations or source references identifying returned `Chunk` nodes and source
  metadata;
- a read-only verification step proving returned chunks and text exist in the
  graph, with MuSiQue supporting-paragraph comparison when the query has a
  labeled record;
- offline unit tests plus one documented local integration run.

The existing ingestion command remains available as a separate prerequisite.
Query execution may also perform the same bounded preload when `--preload` is
active; with the flag off, ingestion is not part of the query invocation.

Full-dataset ingestion, autonomous entity-resolution research, Hybrid Search,
large-scale benchmarking, LangGraph reflection cycles, and production
parallelism are outside the minimum path until this vertical slice works.

## Active Task Sequence

### 1. Freeze the MuSiQue question-to-query contract

- Define a query request containing `record_id`, `question`, `source_split`,
  `source_file`, and a run/trace ID.
- Accept a specific `record_id` as the primary MuSiQue selection input and
  read that record from the configured JSONL file or files.
- Allow the dataset source path or paths to be supplied through `.env`, with
  support for one or more JSONL files. The configured files are the source of
  truth for record selection; the command must not silently substitute a
  different dataset file.
- If no specific record ID is provided, select all records from the configured
  dataset file or files and execute them as a bounded batch. The batch size,
  selected file list, and per-record outcomes must be recorded.
- Preserve the exact `question` text, record ID, paragraph indices, titles,
  and source line from the selected JSONL record.
- Reject empty or missing questions rather than silently substituting a query.
- Preserve the source split derived from the configured source file; do not
  substitute a fixed split or treat records as labeled when their configured
  source does not provide gold answers.
- Define a query response containing the answer, retrieval mode, evidence
  chunks, citations, graph references, status, latency, and error details.
- Ensure the query request and response retain the relationship between the
  question, its source record, and the evidence used to answer it.

**Evidence:** a validated request/response schema, one redacted real-record
example, and tests proving that a requested record ID selects the matching
JSONL record and routes its question unchanged.

**Testing requirement:** Use deterministic fixture JSONL records and mocked
GraphRAG/provider boundaries to test record-ID selection, configured dataset
paths, no-selector batch selection, empty or duplicate inputs, and exact
question preservation without network or paid-model calls.

### 2. Reuse the existing ingestion path for optional preloading

- Treat the existing MuSiQue record adapter and paragraph-as-chunk contract as
  the input boundary for graph loading; do not create a second adapter for QA.
- Add an explicit `--preload` query flag, defaulting to off, that invokes the
  existing bounded loader and `SimpleKGPipeline` for every selected record
  before that record's question is sent to QA.
- When `--preload` is absent, do not invoke the loader or
  `SimpleKGPipeline`; query only the graph that is already available.
- Record the preload flag, selected records, loader outcomes, and timing in
  run metadata. A preload failure must prevent that record's question from
  being sent to the QA system and must produce an explicit failure outcome.
- Preserve `record_id`, `chunk_id`, `document_id`, paragraph index, title,
  source file, checksum, source line, and original paragraph text.
- Keep `is_supporting`, answers, decomposition labels, and other gold fields
  outside the plugin-visible extraction input.
- Keep loading bounded and repeatable, with dead-letter reporting for malformed
  records and explicit failure for plugin/provider errors.
- Verify or document the reset/idempotency behavior needed for repeated
  `--preload` runs. Query execution must never ingest implicitly when the flag
  is off.

**Existing evidence:** ingestion tests, dry-run planning, and live chunk-level
failure reporting are present. Remaining QA evidence must cover both a
preloaded record whose `Document`/`Chunk` provenance passes the read-only
verifier and a query-only run that does not invoke ingestion.

**Testing requirement:** Keep ingestion and provenance tests deterministic and
verify both `--preload` behavior and default query-only behavior. Tests must
prove that the flag controls whether the loader and `SimpleKGPipeline` are
invoked, and that a preload failure prevents QA execution.

### 3. Verify the Neo4j GenAI graph and retrieval prerequisites

- Record the exact Neo4j edition/version, `neo4j-graphrag` version, database,
  LLM, and embedding configuration used by the run.
- Confirm that the selected Neo4j GenAI version exposes the GraphRAG retrieval
  API and document its expected input/output shape.
- Configure or verify the vector index required by the supported Neo4j GenAI
  retrieval implementation, including its name, dimensions, similarity
  function, and target node/property.
- Verify that the plugin-created `Document`, `Chunk`, entities, source links,
  and relationships are queryable after ingestion.
- Verify that the retrieval result can retain chunk and source provenance.
- Fail clearly when the vector index, embedding dimension, database, or
  plugin capability is unavailable. Do not report an empty result as a valid
  answer.

**Evidence:** a capability/configuration report, idempotent index initialization
or verification command, and read-only graph inspection for the selected
record.

**Testing requirement:** Mock capability and index responses to test valid,
missing, dimension-mismatch, unavailable-database, and malformed-retrieval
conditions deterministically. Include a read-only provenance assertion for
returned evidence.

### 4. Implement the minimal Neo4j GraphRAG retrieval adapter

- Add a small retrieval module that accepts the validated MuSiQue question and
  a fixed configuration for retrieval mode, top-K, and bounded graph context.
- Call Neo4j's supported GraphRAG implementation rather than performing a
  local vector query or manually assembling a competing retrieval algorithm.
- Return structured evidence with chunk IDs, document IDs, source metadata,
  retrieved text, graph entities/relationships when available, and scores or
  ranking information when the plugin exposes them.
- Preserve the original question and trace ID through the retrieval call.
- Handle empty results, plugin errors, malformed plugin responses, and context
  limits explicitly.
**Evidence:** mocked unit tests for mode selection, top-K and empty/error
responses; one local integration result showing Neo4j GraphRAG returned
record-linked evidence.

**Testing requirement:** Test the adapter independently for mode selection,
top-K propagation, structured evidence mapping, empty results, provider
errors, malformed responses, context limits, and record/provenance filtering.

### 5. Implement the answer-generation boundary

- Use the answer-generation capability provided by Neo4j's GenAI/GraphRAG
  stack when supported by the installed version and configuration.
- If the local API separates retrieval from generation, add only the thinnest
  provider adapter required to generate an answer from the Neo4j-returned
  context. Keep model configuration in the existing OpenRouter settings.
- Pass the MuSiQue question and retrieved evidence to the answer generator;
  do not pass supporting labels, gold answers, or decomposition labels as
  hidden hints.
- Require a structured response with answer text, citations/evidence IDs,
  retrieval mode, trace ID, and status.
- Reject or mark incomplete any answer with no supporting evidence or missing
  citations. Never treat a provider failure as a successful answer.
- Keep the first answer path single-shot and bounded. Reflection and automatic
  retries are follow-up work after the basic answer contract is verified.

**Evidence:** offline tests for grounded, empty-evidence, malformed-response,
provider-error, and citation-preservation cases; one live answer response with
its configuration and trace metadata recorded.

**Testing requirement:** Use a deterministic fake answer provider to verify
grounded answers, insufficient evidence, missing citations, malformed output,
provider failure, and preservation of trace and source metadata.

### 6. Add MuSiQue-specific correctness verification

- Compare the returned evidence chunk IDs and source text with the selected
  record's paragraphs.
- For labeled dev records, report whether the answer evidence includes the
  record's supporting paragraphs. Treat this as an evaluation signal, not as
  input to extraction or generation.
- Preserve a read-only graph trace showing the `Document`/`Chunk` provenance
  and the entity/relationship path used by GraphRAG where available.
- Define an explicit insufficient-evidence response when GraphRAG cannot
  support the question.
- Store the raw question, retrieved evidence, answer, citations, plugin/model
  configuration, timings, and errors as a reproducible run artifact.

**Evidence:** one grounded example, one insufficient-evidence or empty-result
example, and a machine-readable run artifact that can be inspected without
re-running paid calls.

**Testing requirement:** Verify supporting-paragraph comparison, graph-source
 membership, explicit insufficient-evidence status, and reproducible artifact
 contents using fixed fixture records and mocked graph responses.

### 7. Create the minimal query command and documentation

- Add one documented command that sends a custom question or a selected
  MuSiQue question through retrieval and answer generation. The command must
  expose `--preload`; with the flag on it preloads selected MuSiQue records
  first, and with the flag off it only queries the available graph.
- Support `--record-id` (or an equivalent explicit record-ID option) for
  targeted MuSiQue selection from the configured source files. No default
  dataset file may be substituted when the configuration is absent.
- Route the selected record's `question` field unchanged to the QA system and
  include the selected record ID in the result provenance.
- Read one or more dataset file paths from `.env` configuration, such as
  `GRAPH_RAG_DATASET_PATHS`. The configured paths are the complete source of
  truth for selection.
- When no record ID is provided, select all records from the configured files
  and execute them as a bounded batch rather than inventing a question or
  silently choosing one record.
- Keep ingestion as a separately documented prerequisite command, while also
  allowing the query command to invoke that same path only when `--preload` is
  explicitly supplied. Never reload or re-extract implicitly.
- Reuse the existing Neo4j/model settings conventions and avoid requiring the
  original AllenNLP environment.
- Add flags for custom question, dataset path, record ID, retrieval mode,
  top-K, output artifact, and progress/timing output as appropriate. Record
  position may be supported as a convenience, but record ID is the required
  stable selection path.
- Make the command print a concise structured result and write a full run
  artifact for audit.
- Document prerequisites, Neo4j startup, environment variables, index setup,
  `--preload` behavior, expected provider cost, and the exact
  record/question used in the demo. State clearly that the flag is off by
  default.
- Include a clear note that a MuSiQue demo question originates from the
  selected record in a configured JSONL source, not from a hand-authored
  benchmark query.

**Evidence:** clean local query runs from the repository root covering both a
query-only run against an available graph and a `--preload` run, with command
output, run artifacts, and updated project documentation. Keep loader output
and preload outcomes in the run artifact when preloading is enabled.

**Testing requirement:** Add a deterministic command-level test for custom
 questions, record-ID questions, configured dataset files, no-selector batch
 execution, output artifacts, and nonzero failure behavior.

### 8. Create a separate GraphRAG retrieval-test program

The project must also provide a retrieval-test Python program for testing the
GraphRAG system against the available Neo4j graph. It must expose the same
explicit `--preload` flag: when active it loads selected MuSiQue records before
answering; when inactive it must not reload or re-extract records.

- Add a separate command/module for query testing that connects to the
  existing Neo4j graph and invokes the same Neo4j GenAI GraphRAG retrieval and
  answer-generation path used by the end-to-end demonstrator.
- Support a custom question supplied directly through the command line or an
  equivalent input option. A custom question must be clearly marked as
  externally supplied and must not be assigned a false MuSiQue record ID.
- Support selecting one or more MuSiQue records from the configured JSONL
  files by record ID and optionally 1-based record position, then use each
  selected record's stored `question` field as the query. Record ID selection
  is the required path for targeting a known preloaded graph record.
- Read one or more dataset file paths from `.env`, and use those configured
  files when resolving record IDs or positions. If no record selector is
  supplied, select all records from all configured files as a bounded batch.
- Preserve the originating dataset file for every question and do not merge
  records with the same ID across files without reporting the collision.
- Allow multiple selected record questions to be executed as a bounded batch,
  with one result per question and a summary of successes, insufficient
  evidence responses, and failures.
- When `--preload` is active, load each selected record through the existing
  bounded ingestion path before querying it. When the flag is inactive,
  verify that the relevant `Document`/`Chunk` provenance exists without
  ingesting missing records; missing graph data must produce an explicit
  validation error or a clearly marked unavailable result.
- Preserve question origin metadata in every result: custom question, or
  MuSiQue dataset, source split, source file, record ID, and source line when
  available.
- Reuse the validated query, retrieval, answer, citation, timing, and error
  contracts from the main path rather than creating a second incompatible
  response format.
- Add flags for custom question, record ID, record position/range, `--preload`,
  retrieval mode, top-K, output artifact path, and progress display as
  appropriate. Record whether preloading was requested in every result.
- Keep Neo4j GenAI GraphRAG as the retrieval implementation. The test program
  is a query harness, not a local vector search or a replacement retrieval
  implementation.
- Add offline tests covering custom-question selection, MuSiQue-question
  selection, multiple-record execution, optional preloading, missing graph
  data in query-only mode, empty retrieval, and preservation of
  question/provenance metadata.
- Document that this program supports repeated query-only experiments against
  available graphs and optional preloading experiments. When `--preload` is
  active, record ingestion time separately from retrieval and answer time.

**Evidence:** a separate retrieval-test command, mocked tests for both input
selection paths, one custom-question run, one run using questions from at
least two selected MuSiQue records, and machine-readable output for each
query.

**Testing requirement:** Add a composed offline test that selects multiple
 records, invokes the shared retrieval and answer path, and verifies one
 result per input plus the aggregate success, insufficient-evidence, and
 failure summary.

## Definition Of Done

This goal is complete when all of the following are true:

- A user can run one documented query command in query-only mode against an
  available local Neo4j graph or pass `--preload` to load selected records
  through `SimpleKGPipeline` before querying them.
- The query command accepts a custom question or submits a selected MuSiQue
  record's own `question` field unchanged.
- Dataset file selection is configurable through `.env`; with no specific
  record selector, the command can process all records from one or more
  configured JSONL files as a bounded batch.
- When `--preload` is active, the preloading step converts each selected
  record's paragraphs into validated chunks and processes them by Neo4j GenAI
  `SimpleKGPipeline`; when it is inactive, no preloading occurs.
- Neo4j's GenAI GraphRAG retrieves evidence from the resulting graph using the
  configured retrieval path.
- The system produces an answer or an explicit insufficient-evidence result.
- The response identifies the source record, chunks, citations, retrieval mode,
  and trace/run ID.
- Read-only verification proves that returned evidence is linked to the
  selected MuSiQue record and persisted source text.
- Provider, plugin, index, validation, and graph failures are visible and do
  not become false successful answers.
- Offline tests cover the contracts and mocked plugin boundaries, and one local
  integration run is recorded without committing secrets or paid raw payloads.
- Each implementation step has deterministic tests for its success and failure
  behavior, and a composed end-to-end test verifies that the implemented
  features work together and preserve question, evidence, provenance,
  citations, status, and trace metadata.
- A separate retrieval-test program supports both query-only execution against
  graphs loaded earlier and explicit `--preload` execution, using either a
  custom question or questions selected from MuSiQue records.
- The implementation uses Neo4j GenAI capabilities maximally for extraction,
  indexing, GraphRAG retrieval, and answer generation, with local Python code
  limited to validation, orchestration, provenance, and verification.

## Explicitly Deferred

The following are not prerequisites for this goal:

- ingesting all MuSiQue train/dev records;
- autonomous entity-resolution decisions and precision/recall benchmarking;
- a general user-facing web application;
- Hybrid Search as a third retrieval mode;
- a custom vector database or hand-built vector retriever;
- a full LangGraph reflection loop;
- large-scale throughput, latency, RAGAS, or cost-reduction claims;
- evaluation on MuSiQue test records with withheld labels.

These remain roadmap work in `agent-coordination/tasks.md` after the basic
question-to-answer path is operational.

## Traceability To The Main Roadmap

| This goal | Source roadmap tasks |
|---|---|
| Question contract and record selection | Week 4 dataset handling; Week 6 query request/response contract |
| Plugin-managed record ingestion | Week 4 ingestion; Week 5 `SimpleKGPipeline` extraction |
| Vector/index prerequisites | Week 4 Neo4j schema/index work; Week 6 plugin retrieval configuration |
| GraphRAG retrieval | Week 6 configurable VectorRAG/GraphRAG retrieval |
| Answer and evidence contract | Week 6 common answer-generation interface; Week 7 grounding checks |
| MuSiQue correctness verification | Week 7 supporting-fact/path cross-checks |
| Tests and reproducibility | Cross-cutting engineering quality; Week 6 retrieval tests; Week 7 deterministic fixtures |
| Run artifacts and documentation | Project controls, risk register, and final evidence checklist |
