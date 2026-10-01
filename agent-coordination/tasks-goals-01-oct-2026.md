# Neo4j GraphRAG Question-to-Answer Goal

**Date:** 27 September 2026

**Last modified:** 27 September 2026

**Purpose of this file:**

Define the minimum implementation tasks required to complete an executable
question-to-answer path for the capstone. The path must start with a question
stored in a MuSiQue dataset record, use Neo4j's GenAI capabilities as the
primary implementation wherever supported, retrieve evidence from the Neo4j
knowledge graph, and return an answer with inspectable provenance.

This is a focused successor to the extraction goal in
`agent-coordination/tasks-goals-23-sept-2026.md`. It extracts only the roadmap
work needed to move from the verified plugin-managed extraction slice to a
working Neo4j GraphRAG answer for a real MuSiQue question.

## Goal

Implement and verify one complete local flow:

```text
MuSiQue JSONL record
  -> record.question
  -> validated paragraph chunks
  -> Neo4j GenAI SimpleKGPipeline
  -> Neo4j knowledge graph and vector index
  -> Neo4j GenAI GraphRAG retrieval
  -> answer generation using retrieved graph evidence
  -> answer, citations, source chunks, and execution trace
```

The initial demonstrator should use a labeled record from
`dataset/musique/data/data/musique_full_v1.0_dev.jsonl`. The question submitted
to the system must be the `question` field from that same record. It must not
be manually invented, copied from a separate query file, or replaced with a
question about the fixture. The record's paragraphs and supporting-fact labels
remain available for evaluation and verification, but gold labels must not be
sent to the extraction or answer-generation prompt as model-visible input.

## Architecture Decision

Maximally use Neo4j's GenAI pipeline and managed capabilities for the active
path. Application code should provide configuration, dataset adaptation,
provenance, orchestration, validation, and evidence reporting while Neo4j's
GenAI tooling performs extraction and GraphRAG retrieval wherever the installed
versions support it.

The active implementation priorities are:

- Neo4j GenAI `SimpleKGPipeline` for paragraph-to-graph extraction.
- Neo4j-managed vector configuration and indexing for graph-backed retrieval.
- Neo4j GenAI GraphRAG for query-time retrieval and context construction.
- Neo4j's supported LLM/embedding integrations for answer generation when the
  installed GraphRAG version exposes that capability.
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

- one `musique_full_v1.0_dev.jsonl` record selected by stable record ID or
  1-based dataset position;
- all paragraphs from that record ingested as paragraph-sized chunks;
- the record's own `question` passed through a validated query request;
- a Neo4j GraphRAG retrieval call returning graph-backed evidence;
- an answer generated from the returned evidence;
- citations or source references that identify the MuSiQue record and chunks;
- a deterministic or read-only verification step connecting the answer evidence
  to the record's supporting paragraphs;
- offline unit tests plus one documented local integration run.

Full-dataset ingestion, autonomous entity-resolution research, Hybrid Search,
large-scale benchmarking, LangGraph reflection cycles, and production
parallelism are outside the minimum path until this vertical slice works.

## Active Task Sequence

### 1. Freeze the MuSiQue question-to-query contract

- Define a query request containing `record_id`, `question`, `source_split`,
  `source_file`, and a run/trace ID.
- Read the selected record from the official local JSONL file and preserve the
  exact `question` text, record ID, paragraph indices, titles, and source line.
- Reject empty or missing questions rather than silently substituting a query.
- Keep the source split explicit. Use `full_dev` or the actual inferred split
  for the initial labeled evaluation path; do not treat test records as if
  they had gold answers.
- Define a query response containing the answer, retrieval mode, evidence
  chunks, citations, graph references, status, latency, and error details.
- Ensure the query request and response retain the relationship between the
  question, its source record, and the evidence used to answer it.

**Evidence:** a validated request/response schema, one redacted real-record
example, and tests proving that the question comes from the selected JSONL
record unchanged.

### 2. Prepare the selected record in the existing ingestion path

- Reuse the existing MuSiQue record adapter and paragraph-as-chunk contract.
- Ingest every paragraph belonging to the selected record so the GraphRAG
  system can reason across the record's multi-hop evidence.
- Preserve `record_id`, `chunk_id`, `document_id`, paragraph index, title,
  source file, checksum, source line, and original paragraph text.
- Exclude `is_supporting`, answers, decomposition labels, and other gold fields
  from the plugin-visible extraction input.
- Make the run bounded and repeatable, with dead-letter reporting for malformed
  records and explicit failure for plugin/provider errors.
- Ensure the selected record can be reset and re-ingested without accumulating
  duplicate source graph structures.

**Evidence:** existing ingestion tests remain passing, one selected-record dry
run, and one live ingestion log identifying the record and number of chunks.

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
- Add a narrow Cypher evidence check that confirms each returned chunk belongs
  to the selected MuSiQue record and that its source text is present.

**Evidence:** mocked unit tests for mode selection, top-K and empty/error
responses; one local integration result showing Neo4j GraphRAG returned
record-linked evidence.

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

### 7. Create the end-to-end command and documentation

- Add one documented command that selects a MuSiQue dev record and runs the
  complete path from record loading through answer generation.
- Reuse the existing ingestion CLI/settings conventions and avoid requiring the
  original AllenNLP environment.
- Add flags for dataset path, record ID or record position, retrieval mode,
  top-K, dry-run, and progress/timing output as appropriate.
- Make the command print a concise structured result and write a full run
  artifact for audit.
- Document prerequisites, Neo4j startup, environment variables, index setup,
  expected provider cost, and the exact record/question used in the demo.
- Include a clear note that the demo question originates from
  `musique_full_v1.0_dev.jsonl`, not from a hand-authored benchmark query.

**Evidence:** a clean local run from the repository root, command output, run
artifact, and updated project documentation.

### 8. Create a separate GraphRAG retrieval-test program

The project must also provide a retrieval-only Python program for testing the
GraphRAG system after selected MuSiQue records have already been ingested into
Neo4j. This program must not assume that it needs to reload or re-extract the
records before answering a question.

- Add a separate command/module for query testing that connects to the
  existing Neo4j graph and invokes the same Neo4j GenAI GraphRAG retrieval and
  answer-generation path used by the end-to-end demonstrator.
- Support a custom question supplied directly through the command line or an
  equivalent input option. A custom question must be clearly marked as
  externally supplied and must not be assigned a false MuSiQue record ID.
- Support selecting one or more MuSiQue records from the local
  `musique_full_v1.0_dev.jsonl` file by record ID and/or 1-based record
  position, then use each selected record's stored `question` field as the
  query.
- Allow multiple selected record questions to be executed as a bounded batch,
  with one result per question and a summary of successes, insufficient
  evidence responses, and failures.
- Assume that the selected records' graphs are already loaded in Neo4j. The
  program may verify that the relevant `Document`/`Chunk` provenance exists,
  but it must not silently ingest missing records. Missing graph data should
  produce an explicit validation error or a clearly marked unavailable result.
- Preserve question origin metadata in every result: custom question, or
  MuSiQue dataset, source split, source file, record ID, and source line when
  available.
- Reuse the validated query, retrieval, answer, citation, timing, and error
  contracts from the main path rather than creating a second incompatible
  response format.
- Add flags for custom question, record ID, record position/range, retrieval
  mode, top-K, output artifact path, and progress display as appropriate.
- Keep Neo4j GenAI GraphRAG as the retrieval implementation. The test program
  is a query harness, not a local vector search or a replacement retrieval
  implementation.
- Add offline tests covering custom-question selection, MuSiQue-question
  selection, multiple-record execution, missing preloaded graph data, empty
  retrieval, and preservation of question/provenance metadata.
- Document that this program is intended for repeated retrieval experiments
  against preloaded graphs and does not measure ingestion time unless the
  caller separately records that process.

**Evidence:** a separate retrieval-test command, mocked tests for both input
selection paths, one custom-question run, one run using questions from at
least two selected MuSiQue records, and machine-readable output for each
query.

## Definition Of Done

This goal is complete when all of the following are true:

- A user can run one documented command against a local Neo4j instance and a
  local `musique_full_v1.0_dev.jsonl` file.
- The command selects a real labeled MuSiQue record and submits that record's
  own `question` field unchanged.
- The record's paragraphs are converted into validated chunks and processed by
  Neo4j GenAI `SimpleKGPipeline`.
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
- A separate retrieval-test program can query graphs that were loaded earlier
  without re-running ingestion, using either a custom question or questions
  selected from MuSiQue records.
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
