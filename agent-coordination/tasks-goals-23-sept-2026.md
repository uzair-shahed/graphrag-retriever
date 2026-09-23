# One-Agent Initial Draft Goal

**Date:** 23 September 2026

## Goal

By the end of today, produce one working extraction agent that accepts one validated MuSiQue paragraph chunk and returns validated structured mentions and relationships with evidence spans, confidence, and provenance.

This is a vertical slice, not a complete GraphRAG system. Retrieval, entity resolution, graph persistence, answering, reflection, evaluation, and production hardening remain out of scope for today's goal.

## Repository Audit: 23 September 2026

The repository was checked against this goal before implementation work began.

| Goal item | Status | Evidence |
|---|---|---|
| Small MuSiQue fixture exists | **Complete** | `tests/fixtures/musique/` contains valid, invalid, alias, distractor, unanswerable, and focused extraction fixtures plus a manifest. |
| Extraction-specific paragraph fixture exists | **Complete** | `tests/fixtures/musique/extraction_chunk.json` contains one paragraph with stable identity and provenance. |
| Extraction package exists | **Partial** | `src/graphrag_pipeline/extraction/` now contains Contract 2 schemas; provider and agent modules remain TODO. |
| Extraction schemas and validation exist | **Complete for the minimum contract** | Pydantic models validate required fields, provenance, evidence spans, mention references, confidence, and request/result identity. |
| Provider boundary and deterministic mock exist | **Missing** | No `ExtractionProvider`, mock provider, or LLM provider exists. |
| Bounded extraction retry/dead-letter workflow exists | **Missing** | Existing retries belong to Neo4j transactions, not extraction responses. |
| Extraction-focused tests exist | **Partial** | `tests/unit/test_extraction_schemas.py` covers the frozen schema boundary; provider and agent tests remain TODO. |
| Neo4j connection infrastructure exists | **Complete but out of scope for P0** | `Neo4jConnection` and its unit/integration tests exist; graph writes are intentionally deferred for this first agent. |

Therefore, today’s extraction-agent P0 remains open. The existing MuSiQue fixtures reduce the fixture work, but they do not count as a working extraction vertical slice until one paragraph is mapped into the Contract 2 request and processed by an extraction agent.

## Progress Update: Minimum Extraction Contract

The first P0 task, **Freeze the minimum extraction contract**, is complete.

Completed:

- Added Pydantic Contract 2 models under `src/graphrag_pipeline/extraction/schemas.py`.
- Added required request fields: `chunk_id`, `record_id`, `document_id`, `paragraph_idx`, `text`, and provenance.
- Added required result fields: `chunk_id`, `record_id`, `chunk_text`, provenance, `mentions`, and `triples`.
- Added evidence-span bounds and exact-text validation at the request/result boundary.
- Added unique mention-ID and triple-to-mention reference validation.
- Added confidence validation for the inclusive range `[0.0, 1.0]`.
- Added `tests/fixtures/musique/extraction_chunk.json` without the gold `is_supporting` label.
- Added six focused schema tests; all pass with `pytest`.
- Added Pydantic to `pyproject.toml` and installed the editable project environment.

Remaining for the extraction-agent P0:

- Implement `ExtractionProvider` and the deterministic mock provider.
- Implement the extraction agent and bounded retry/dead-letter behavior.
- Add provider and agent tests, including malformed output and retry exhaustion.
- Add the short execution trace and the documented fixture-to-agent command.

Intentionally skipped for this task:

- Contract 1 record adapter and full JSONL normalization.
- LangGraph node wiring.
- Neo4j graph writes, AI-plugin retrieval, and vector indexes.
- Entity resolution, answering, reflection, evaluation, and production hardening.

## Working Assumption

The first agent is the **Extraction Agent** because it is the first agent in the ingestion path:

```text
MuSiQue record -> paragraph chunk -> Extraction Agent -> validated extraction result
```

The agent may use a deterministic mock provider for the initial draft. A paid model is not required to prove the contract or orchestration path.

## Definition Of Done

The goal is complete only when all of these are true:

- One fixture chunk can be loaded without downloading more data or contacting a paid provider.
- The extraction input excludes gold evaluation labels such as `is_supporting`.
- One provider call or deterministic mock call is wrapped by the agent.
- The result is validated before it leaves the extraction boundary.
- Valid output contains mentions, relationships, evidence spans, confidence, chunk identity, record identity, and provenance.
- Invalid JSON or schema-invalid output is rejected.
- The workflow has a bounded retry count, with a dead-letter/failure result after retries are exhausted.
- At least one success test and one malformed-output failure test pass.
- A short execution trace identifies the chunk, attempt number, validation result, and final status.

## Priority Order

### P0: Required today

1. **Freeze the minimum extraction contract.**

   Implement only the fields already specified in Contract 2:

   - input: `chunk_id`, `record_id`, `document_id`, `paragraph_idx`, `text`, and required provenance;
   - output: `chunk_id`, `record_id`, `chunk_text`, provenance, `mentions`, and `triples`;
   - mention fields: `mention_id`, `mention`, `entity_type`, `evidence_span`;
   - triple fields: `source_mention_id`, `relationship_type`, `target_mention_id`, `evidence_span`, `confidence`.

2. **Create the smallest project fixture.** **Partial: ingestion fixtures already exist; extraction fixture still required.**

   Reuse one paragraph from `tests/fixtures/musique/valid_records.jsonl` and add a focused extraction request fixture containing one paragraph with two entities and one relationship. Keep the fixture deterministic and preserve the original text exactly. Do not expand the existing multi-record fixture unless a test needs it.

3. **Implement schemas and validation.**

   Use the project's selected validation library, preferably Pydantic. Validate:

   - required IDs and provenance;
   - unique mention IDs;
   - valid evidence offsets and exact span text;
   - triple references to existing mentions;
   - confidence in the inclusive range `[0.0, 1.0]`;
   - matching input and output `chunk_id` and `record_id`.

4. **Implement one provider boundary.**

   Define a small interface such as:

   ```python
   class ExtractionProvider(Protocol):
       def extract(self, request: ExtractionRequest) -> dict: ...
   ```

   Add a deterministic mock provider first. The agent should not know whether the provider is a mock or an LLM.

5. **Implement bounded retry behavior.**

   Allow the initial response plus at most two corrective retries. Retry malformed or schema-invalid responses only. Do not retry ordinary application bugs indefinitely. Return a structured failure/dead-letter result after exhaustion.

6. **Add focused tests.**

   Required tests:

   - valid deterministic extraction is accepted;
   - invalid evidence span is rejected;
   - unknown triple mention ID is rejected;
   - malformed provider output is retried and then dead-lettered;
   - extraction output never includes or consumes `is_supporting`.

7. **Run the vertical slice.**

   The command should be one documented test or script that demonstrates:

   ```text
   fixture chunk -> extraction agent -> validation -> structured result
   ```

### P1: Do only if P0 is complete

- Add a real LLM provider behind the same provider interface.
- Add corrective validation-error details to retry prompts.
- Add structured JSON logging for `run_id`, `chunk_id`, attempt, latency, and status.
- Add one LangGraph node around the extraction agent if the LangGraph dependency is already installed and stable.
- Add a small command-line entry point for the fixture demonstration.

### P2: Explicitly defer

Do not spend today's time on these tasks:

- Neo4j AI plugin GraphRAG or VectorRAG setup;
- Neo4j vector indexes, full-text indexes, or retrieval configuration;
- entity resolution and candidate ranking;
- canonical entity IDs or graph writes;
- answering and reflection agents;
- Hybrid Search or retrieval routing;
- MuSiQue full-dataset ingestion;
- Ray, Docker, Redis, PostgreSQL, Polars, DuckDB, or cloud storage;
- the 150-pair entity-resolution benchmark;
- the 30-50 query evaluation suite;
- RAGAS, latency benchmarks, cost reports, or final metrics;
- production retries, distributed processing, or broad CI hardening.

## Suggested File Separation

Keep responsibilities separate:

```text
src/graphrag_pipeline/
  extraction/
    __init__.py
    schemas.py          # request/result models and validation
    provider.py         # provider protocol and deterministic mock
    agent.py            # retry loop and extraction orchestration

tests/unit/
  test_extraction_schemas.py
  test_extraction_agent.py

tests/fixtures/
  musique/
    extraction_chunk.json
```

Do not put extraction logic into the Neo4j connection module. Neo4j is not required for this first agent slice because the validated extraction result should be produced before entity resolution and graph writes.

## Required Existing Tasks To Pull Forward

From `agent-coordination/tasks.md`, today's work should pull forward only these items:

- define the project-specific minimal fixture;
- define the paragraph-as-chunk input boundary;
- define Pydantic/Instructor schemas for entities, relations, evidence, confidence, and source spans;
- implement strict parsing and validation for malformed or incomplete model output;
- add bounded retries with corrective prompts and a dead-letter result;
- build the LangGraph extraction agent or equivalent extraction orchestration;
- create sample-chunk tests for extraction schema enforcement.

The Neo4j connection smoke test is useful infrastructure, but it is not on the critical path for the first extraction-agent draft unless the agent is explicitly required to write to Neo4j today.

## End-Of-Day Evidence

Save or be able to show:

- one fixture input;
- one accepted extraction output;
- one rejected/dead-letter output;
- test output showing the focused tests pass;
- the retry limit and provider configuration used;
- a short note confirming that gold labels were not sent to the agent.

A reasonable completion report is:

```text
Extraction agent draft complete.
Input contract validated: yes
Deterministic provider: yes
Valid output test: passed
Malformed output test: passed
Retry limit: 2 corrective retries
Neo4j write: deferred
GraphRAG retrieval: deferred
```

## Stop Condition

Stop after the P0 vertical slice is demonstrably passing. Do not widen the task because the roadmap contains later work. The purpose of today is to establish one trustworthy agent boundary that the later LangGraph, entity-resolution, Neo4j, retrieval, and reflection work can consume.
