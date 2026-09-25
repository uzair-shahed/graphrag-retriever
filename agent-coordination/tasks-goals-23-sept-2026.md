# Neo4j GenAI Plugin Extraction Goal

**Date:** 23 September 2026

**Last modified:** 23 September 2026

**Purpose of this file:**
Track the first extraction vertical slice using Neo4j GenAI Plugin's
`SimpleKGPipeline` as the primary MuSiQue-to-Knowledge-Graph implementation.
This file defines the active tasks, validation evidence, scope boundaries, and
the manual extraction implementation that is retained only as a fallback.

**Files that use this file:**
The project roadmap in `agent-coordination/tasks.md` supplies the broader
milestones. The architecture and data contracts in
`agent-coordination/data-contracts.md` define the interfaces that this goal
must eventually satisfy. The source code, tests, and Neo4j setup created from
this goal should link back to the task evidence here.

**Caveats:**
Neo4j GenAI Plugin feature names, configuration keys, supported Neo4j editions,
and `SimpleKGPipeline` behavior must be verified against the exact local Neo4j
and plugin versions before implementation is treated as complete. The plugin
may create graph data directly rather than return the application's Contract 2
mention/triple JSON shape. Until that response and provenance behavior are
verified, the existing hand-built extraction schemas and provider are archived
fallback work, not the active extraction path. Entity resolution policy,
reflection, retrieval evaluation, and production hardening remain outside this
vertical slice.

## Goal

By the end of this work, run one small MuSiQue fixture through a locally
configured Neo4j GenAI Plugin `SimpleKGPipeline`, create a queryable knowledge
graph, and verify that the resulting nodes, relationships, source text, and
provenance satisfy the project's minimum graph contract.

The active flow is:

```text
MuSiQue record
  -> validated paragraph input
  -> local Python pipeline configuration
  -> Neo4j GenAI Plugin SimpleKGPipeline
  -> Neo4j knowledge graph
  -> deterministic graph/provenance verification
```

This is a plugin-integration vertical slice. Retrieval, answering, reflection,
benchmarking, and custom extraction-agent implementation remain outside today's
active scope.

## Architecture Decision

Neo4j GenAI Plugin is the primary extraction implementation for the current
project path. `SimpleKGPipeline` is expected to handle the plugin-managed
document processing, entity/relation extraction, and graph population. Local
Python remains responsible for:

- selecting and reading the MuSiQue fixture or official split;
- preserving source text and dataset provenance;
- supplying plugin configuration and credentials through environment variables;
- invoking the pipeline and bounding the run;
- inspecting and validating the resulting graph;
- recording failures, configuration, and reproducibility evidence.

The project must not assume that `SimpleKGPipeline` returns the hand-built
`ExtractionResult` model. The integration must document the actual plugin output
surface and map or verify it against the graph-level contracts before later
entity resolution and retrieval work proceeds.

## Repository Audit Before This Goal

| Area | Status | Evidence |
|---|---|---|
| Small MuSiQue fixture | Complete | `tests/fixtures/musique/` contains valid, invalid, alias, distractor, unanswerable, and focused extraction fixtures plus a manifest. |
| Focused extraction fixture | Complete | `tests/fixtures/musique/extraction_chunk.json` contains one paragraph with stable identity and provenance. |
| Local Neo4j connection | Complete | `src/graphrag_pipeline/storage/neo4j_connection.py` and its tests provide bounded local connectivity. |
| Neo4j GenAI Plugin capability verification | Partial | Neo4j Enterprise `2026.08.1`, APOC, `genai.vector.*`, and `neo4j-graphrag==1.21.0`/`SimpleKGPipeline` availability are recorded; configuration and end-to-end smoke test remain. |
| Plugin-backed MuSiQue extraction | Missing | No active plugin adapter or fixture-to-graph command exists. |
| Graph/provenance verification | Missing | No test proves that plugin-created graph data links back to the MuSiQue record and paragraph. |
| Manual Contract 2 extraction path | Archived | Existing schemas/provider code is retained as fallback and is not the active implementation for this goal. |

## Definition Of Done

This goal is complete only when all of the following are true:

- The exact local Neo4j and GenAI Plugin versions are recorded.
- The supported Neo4j edition and required plugin capabilities are verified.
- Required environment variables and secrets are documented without storing
  credentials in the repository.
- `SimpleKGPipeline` can be initialized from the local Python environment with
  the selected LLM and embedding configuration.
- One focused MuSiQue paragraph can be supplied without downloading the full
  dataset or requiring the original AllenNLP environment.
- The pipeline creates a graph containing inspectable source-backed entities,
  relationships, and source/chunk provenance, or produces a documented,
  actionable plugin limitation.
- The fixture run is deterministic enough to be rerun and compared under a
  fixed model, prompt, configuration, and seed where supported.
- A test or verification script checks graph existence, expected labels or
  equivalent plugin-created structures, source text, and provenance.
- Plugin failures and unsupported output are reported without silently marking
  the extraction as successful.
- No manual extraction agent, custom extraction provider, or manual retry loop
  is required by the active path.

## Active Task Sequence

### 1. Verify Neo4j and plugin capabilities

- Record the local Neo4j version, edition, database name, and connection URI.
- Identify the exact Neo4j GenAI Plugin package/version and installation method.
- Verify that the selected local edition supports `SimpleKGPipeline`.
- Confirm which LLM and embedding providers the pipeline supports.
- Confirm whether the plugin creates graph nodes and relationships directly,
  returns intermediate extraction objects, or supports both modes.
- Record required APOC, vector, full-text, or other plugin dependencies only if
  `SimpleKGPipeline` actually requires them.
- Record known limitations, unsupported model providers, rate-limit behavior,
  and local resource requirements.

- Record the results in `agent-coordination/neo4j-genai-plugin-capability-report.md`.

**Evidence:** `agent-coordination/neo4j/neo4j-genai-plugin-capability-report.md`, a
version/capability report, and a checked-in configuration example with secrets
removed.

### 2. Define the plugin input contract

**Decision:** The first plugin invocation receives one validated application
chunk, representing one MuSiQue paragraph. The paragraph text is passed to
`SimpleKGPipeline.run_async(text=...)` unchanged. Source identity and
provenance are passed separately through `document_metadata`.

The mapping is implemented by
`src/graphrag_pipeline/extraction/plugin_input.py` and verified by
`tests/unit/test_plugin_input.py`.

- Decide whether the plugin receives one paragraph, one MuSiQue record, or a
  normalized document batch.
- Use the existing focused extraction fixture as the first input.
- Preserve `record_id`, `paragraph_idx`, title, paragraph text, source split,
  source file, checksum, and source line outside the model prompt if the plugin
  does not support metadata fields.
- Ensure gold labels such as `is_supporting`, answers, and decomposition
  labels are not presented to the extraction system as extraction input.
- Define how the application associates plugin-created graph data with the
  source `chunk_id` when the plugin's input format does not preserve it.

**Evidence:** one documented input example and a source-to-plugin mapping test.
The focused mapping test confirms that `text` is unchanged and that metadata
contains `record_id`, `chunk_id`, `document_id`, `paragraph_idx`, title, source
dataset, split, file, checksum, and source line. It also confirms that
`is_supporting`, answers, aliases, and question decomposition are absent.

### 3. Configure a minimal `SimpleKGPipeline`

**Decision:** Use the installed GraphRAG OpenAI-compatible implementations with
OpenRouter's configured base URL. The factory in
`src/graphrag_pipeline/extraction/pipeline.py` builds the pipeline from
environment settings and does not perform LLM or embedding calls during
construction.

The first-run configuration is:

- LLM: `openai/gpt-4o-mini`
- Embedder: `openai/text-embedding-3-small`
- embedding dimension expectation: `1536`
- `from_file=False`
- `on_error="RAISE"`
- `perform_entity_resolution=False`
- database: `neo4j`
- GraphRAG dependency: `neo4j-graphrag[openai]>=1.21,<2`

The factory and settings parsing are covered by
`tests/unit/test_pipeline.py`. Live initialization against the local Neo4j
database passed without making provider inference calls.

- Add the smallest local configuration needed to initialize the pipeline.
- Keep model, embedding, database, retry, and timeout settings in environment
  variables or ignored local configuration.
- Use a deterministic or low-cost provider configuration for the fixture.
- Add a dry-run or initialization check when the plugin supports one.
- Keep the full MuSiQue dataset, paid-scale ingestion, and concurrency out of
  this first run.
- Document whether the pipeline writes directly to Neo4j and which database it
  writes to.

**Evidence:** a reproducible fixture command and a redacted run configuration.
The live initialization check reported `pipeline_initialization=passed` with
the configured database, models, flags, and embedding dimension. Provider
inference remains part of Task 4, when the focused fixture is run.

### 4. Run the fixture through the plugin

**Result:** The focused fixture was run successfully with
`src/graphrag_pipeline/extraction/run_fixture.py`. Run ID:
`de0b6bc5-f9c3-4725-a954-d1c9ab4f05e7`.

The runner does not reset the database automatically, so the generated graph
remains available in the local `neo4j` database for inspection. The first live
attempt exposed a GraphRAG 1.21.0 boundary: numeric `source_line` and
`paragraph_idx` values are rejected by the plugin's metadata validator. The
plugin input adapter now serializes those two metadata values while retaining
their numeric types in the validated application request.

- Invoke `SimpleKGPipeline` with the focused MuSiQue paragraph.
- Capture run ID, fixture ID, model/provider configuration, plugin version,
  start/end timestamps, status, and error information.
- Preserve the original paragraph and source provenance alongside the run.
- Do not manually construct entities or relationships to make the smoke test
  pass.
- Reset the fixture graph before a clean run and verify behavior on a rerun.

**Evidence:** one successful graph creation trace or one reproducible plugin
failure report with the exact blocker.
The successful run returned a pipeline result and persisted data directly
through the Neo4j writer.

### 5. Verify the resulting graph and provenance

The read-only inspection after the successful run found one `Document`, one
`Chunk`, two `Company` entities, two `FROM_CHUNK` relationships, one
`FROM_DOCUMENT` relationship, and one `ACQUIRED` relationship. The paragraph
text and embedding are stored on `Chunk`. Source metadata is stored on the
`Document`, including `chunk_id`, `record_id`, `paragraph_idx`, source dataset,
split, file, and source line. The `ACQUIRED` relationship contains
`since: 2021`.

- Confirm the graph contains the plugin-created entity and relationship data.
- Identify the actual labels, relationship types, and source-reference
  properties created by the plugin.
- Verify that source text and paragraph/chunk identity can be traced from the
  graph back to the MuSiQue fixture.
- Verify that graph inspection is read-only and does not modify the fixture.
- Check whether repeated execution duplicates nodes, relationships, or source
  references.
- Record any mismatch between plugin output and Contracts 1-4.
- Decide whether a thin provenance adapter is needed after plugin execution.

**Evidence:** a graph verification test/query and a provenance mapping report.

### 6. Update the broader project contracts

- Update Contract 2 to identify `SimpleKGPipeline` as the active extraction
  implementation rather than a hand-built provider.
- Define the boundary between plugin-managed extraction and local validation,
  provenance, graph inspection, and later entity resolution.
- Update Contract 4 with the actual plugin-created labels, relationships, and
  source-reference fields once verified.
- Keep manual Pydantic extraction output as a compatibility/fallback contract
  only if it remains useful.
- Update the main roadmap's Neo4j AI Plugin architecture decision with the
  verified version, limitations, and evidence.

## Focused Tests And Evidence

The active plugin path should eventually include tests for:

- plugin configuration loads without exposing secrets;
- unsupported or missing plugin capabilities fail clearly;
- one fixture paragraph reaches `SimpleKGPipeline` unchanged;
- gold labels are excluded from extraction input;
- the pipeline creates expected graph structures;
- source provenance is preserved or is rejected when it cannot be preserved;
- graph reset is safe and repeatable;
- rerunning the same fixture does not silently duplicate graph data;
- plugin errors produce a failed run artifact rather than a false success.

## Explicitly Deferred From This Goal

- full MuSiQue ingestion;
- entity resolution and candidate ranking;
- Neo4j AI Plugin GraphRAG and VectorRAG query evaluation;
- answering and reflection agents;
- benchmark construction and RAGAS evaluation;
- latency, throughput, and cost benchmarking;
- Hybrid Search and retrieval routing;
- Ray, Docker, Redis, PostgreSQL, Polars, DuckDB, and cloud storage;
- custom extraction and retrieval implementations unless the fallback is
  activated;
- production hardening and broad CI work.

## Archived Manual Extraction Plan

The following tasks were previously the active plan for a hand-built Contract 2
extraction agent. They are preserved for a controlled fallback if
`SimpleKGPipeline` cannot satisfy the required extraction, provenance, or graph
quality requirements. They are not required for the current plugin-first path.

### Archived manual implementation

- Implement `ExtractionProvider` and a deterministic mock provider.
- Implement an extraction agent that calls the provider, parses raw output, and
  validates `ExtractionResult` before graph work.
- Allow the initial provider call plus at most two corrective retries.
- Retry malformed or schema-invalid responses only.
- Return a structured dead-letter result after retry exhaustion.
- Add execution traces containing run ID, chunk ID, attempt, validation status,
  and final status.
- Add a manual LLM provider behind the provider protocol if required.
- Add a LangGraph extraction node if the plugin path is abandoned.

### Archived manual schema and test work

- Retain `src/graphrag_pipeline/extraction/schemas.py` as the fallback Contract
  2 request/result schema.
- Retain `src/graphrag_pipeline/extraction/provider.py` and
  `DeterministicMockProvider` for offline fallback tests.
- Retain `tests/unit/test_extraction_schemas.py` and
  `tests/unit/test_extraction_provider.py` as fallback contract tests.
- Re-enable and extend these tests only if the manual provider becomes the
  active implementation.

### Fallback activation criteria

Activate the archived manual path only after recording a decision that the
plugin path is blocked by one or more of:

- unavailable or incompatible `SimpleKGPipeline` capability;
- inability to preserve source provenance;
- graph output that cannot satisfy the required schema or graph contract;
- unacceptable extraction quality on the approved fixture;
- unsupported local model/provider configuration;
- unresolved plugin reliability, cost, or resource limitations.

The decision record must identify the failed plugin test, preserve its output,
and explain why the manual implementation is the smallest viable fallback.

## Stop Condition

Stop after the focused `SimpleKGPipeline` run and graph/provenance verification
are reproducible. Do not begin retrieval benchmarking or rebuild the manual
extraction agent unless the plugin path fails its documented activation criteria.
