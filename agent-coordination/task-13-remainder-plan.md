# Task 13: idempotent paragraph ingestion

Date: 18 October 2026.

## Approved implementation sequence

The user initially required a verification stop after each subtask. After
reviewing 13.1, the user explicitly authorized completing all remaining
subtasks together, superseding those stops. Required live checks must execute
and pass; a skip does not complete a subtask. Task 12 remains separate.

**Final status:** all five subtasks implemented and validated; ready for user
acceptance. Final validation: **277 unit tests and 21 live Neo4j tests passed**.

Confirmed decisions: preserve legacy/partial graph data for investigation;
keep application chunk IDs and reject changed text or provenance under the same
ID; use one complete Chunk per paragraph; keep entity resolution disabled;
retain sequential ingestion; use a custom atomic writer through the supported
`SimpleKGPipeline` component interface. Live tests use deterministic provider
doubles and real Neo4j.

| Subtask | Work and completion gate | Status |
| --- | --- | --- |
| 13.1 | Characterize stock writer replay, partial writes, resolver policy, embeddings and cleanup against live Neo4j. | Complete. |
| 13.2 | Add canonical request fingerprint, completion inventory/version, managed unique node/relationship keys, explicit state classifications and full-paragraph splitter. Validate changed inputs, legacy preservation and live uniqueness/schema setup. | Complete. |
| 13.3 | Install custom `KGWriter`; persist graph, embeddings, provenance and completion in one bounded transaction; verify inventory and reject conflicts. Test rollback, transaction retry and repeat ingestion against real Neo4j. | Complete. |
| 13.4 | Use one validated ingestion boundary for batch, preload and fixture commands; reconcile without requiring a manifest; move provenance synchronization into the atomic write. Prove process restart across pre-commit and post-commit/pre-checkpoint stops, including missing/fresh manifests. | Complete. |
| 13.5 | Finish operational documentation and status records; run all deterministic and Task 13 live checks, audit final source/graph inventories and report evidence. | Complete. |

## 13.1 — Installed writer inspection

The inspected and tested versions are `neo4j-graphrag==1.21.0` (also recorded
in `uv.lock`), Neo4j Python driver `5.28.7`, and Neo4j Enterprise `2026.08.1`.
These findings describe the locked package, not all package versions.

- `components/types.py`: `DocumentInfo.uid` and `TextChunk.uid` default to new
  UUIDs. Application `document_metadata.chunk_id` is a Document property, not
  the plugin's internal node identity.
- `components/entity_relation_extractor.py`: extracted entity IDs are prefixed
  with the generated chunk ID. Repeating identical model output therefore
  does not reuse the previous run's entity identities.
- `components/lexical_graph.py`: the Document carries input metadata; the Chunk
  carries text, index and embedding. The lexical graph supplies `FROM_DOCUMENT`
  and `FROM_CHUNK` relationships.
- `neo4j_queries.py`: the function named `upsert_node_query` uses `CREATE`, with
  a temporary internal ID. Relationship writes match those newly created
  endpoints and use `apoc.merge.relationship`. Merging relationships between
  new endpoints cannot deduplicate a previous graph's relationships.
- `components/kg_writer.py`: node and relationship batches use separate
  `driver.execute_query` calls. A later relationship failure does not roll back
  committed node batches. A caught Neo4j `ClientError` becomes a writer result
  with `status="FAILURE"` rather than an exception.
- The default writer creates a temporary-ID index and clears temporary IDs
  globally after successful writes. That cleanup is not reached after the
  injected relationship failure.
- `experimental/pipeline/config/template_pipeline/simple_kg_builder.py`:
  `perform_entity_resolution=False` omits the resolver; `kg_writer` accepts a
  replacement component. `on_error="RAISE"` controls extraction error handling
  and does not turn the stock writer's returned failure into an exception.

## 13.1 — Regression fixtures and live evidence

Added [the stock writer characterization tests](../tests/integration/test_stock_writer_behavior.py).
They instantiate the real stock `SimpleKGPipeline` with the current production
flags: text input, FREE schema, 10,000-character splitter with zero overlap,
RAISE extraction errors, and disabled entity resolution. The stock writer is
selected by the package, with no replacement writer. The short synthetic
paragraph is validated as an `ExtractionRequest` and passed through the real
application input adapter. Only the LLM and embedder are deterministic doubles.
No paid or local model service is contacted.

The fixed graph contains two distinct people with the same label and name,
one company, and two `FOUNDED` relationships. Both same-name people remain
separate on each run, verifying the configured disabled resolver behavior.

| Persisted fixture object | First ingestion | Same request replayed |
| --- | ---: | ---: |
| Documents | 1 | 2 |
| Chunks | 1 | 2 |
| Extracted entities | 3 | 6 |
| FROM_DOCUMENT relationships | 1 | 2 |
| FROM_CHUNK relationships | 3 | 6 |
| Domain relationships | 2 | 4 |
| Chunk embeddings | 1 | 2 |

The replay test also verifies stable application provenance on both Documents,
unchanged paragraph text, exact embedding values, retained first-run node and
relationship IDs, relationship endpoint membership, and one extraction and
embedding call per invocation. Stock Chunk nodes lack the application chunk ID;
the existing loader supplies that property separately today.

The failure test injects a Neo4j `ClientError` at the relationship-batch boundary
after real node commits. Despite `on_error="RAISE"`, the pipeline returns a
writer `FAILURE` result without raising. Neo4j retains one Document, one Chunk
with embedding and three entities, with zero source or domain relationships.
This establishes the partial-write window the atomic replacement must close.

Each test requires a quiet database with no existing stock-writer temporary
state before execution, uses a fresh UUID and exact fixture paragraph text for
cleanup, and compares full node/relationship/property snapshot hashes before
the test and after cleanup. Both hashes matched in both tests; all fixture
counts returned to zero. If the stock temporary index did not previously exist,
the test removes the index it created. Existing graph data is preserved.

Sanitized JSON reports, including versions, actual pipeline results, full
fixture inventories, provider-call counts and cleanup checks, are retained at:

- `runs/task13-1-stock-writer/test_stock_pipeline_replay_duplicates_graph.json`
- `runs/task13-1-stock-writer/test_stock_writer_client_error_leaves_committed_nodes.json`

These are local gitignored evidence; this tracked report preserves the findings.

## Reproduce 13.1

From the repository root, with the configured local Neo4j database running and
no concurrent ingestion:

```sh
RUN_TASK13_STOCK_WRITER_TESTS=1 TASK13_EVIDENCE_DIR=runs/task13-1-stock-writer .venv/bin/python -m pytest tests/integration/test_stock_writer_behavior.py -q
```

The test is opt-in because it intentionally demonstrates duplicate graph writes
and partial commits before cleaning up its fixture. Without opt-in it skips.
Once enabled, missing configuration or unavailable Neo4j fails the acceptance
run instead of silently skipping. This fixture currently requires database
`neo4j`, matching the production factory and the stock writer's default-database
setup/cleanup behavior.

At the 13.1 handoff, **2 live tests and 232 unit tests passed**, and production
ingestion was unchanged. The user subsequently authorized finishing the rest.

## 13.2 — Source identity and completeness

- Canonical fingerprint covers every validated request/provenance field, with
  no text, Unicode, path or whitespace normalization. Reused IDs with changed
  input are conflicts; a different manifest cannot bypass database checks.
- `IngestionCompletion` stores contract version 1, full canonical request,
  fingerprint and the exact node/relationship inventory. A marker alone is
  insufficient: managed source nodes and current persisted graph must match.
- Added scoped unique completion/node keys and per-type relationship keys.
  Repeated setup verifies actual constraint definitions, including Neo4j 2026's
  `*_PROPERTY_UNIQUENESS` names. Wrong definitions and duplicate keys fail.
- Added explicit `legacy_unverified` classification. Existing unmarked data,
  orphan chunks and damaged managed ownership remain intact for investigation.
- Full paragraphs, including a live fixture exceeding 30,000 characters, remain
  one unchanged Chunk. Missing titles are omitted from string-valued metadata.

Validation: deterministic field-by-field identity and malformed-state tests,
real database uniqueness checks for completion/nodes/relationships, repeatable
schema setup, incompatible schema detection and preserved legacy data passed.

## 13.3 — Atomic persistence

- Installed `AtomicKGWriter` through `SimpleKGPipeline(kg_writer=...)` for both
  provider configurations. Standard lexical labels, entity labels, source
  links, domain types and vectors remain available to retrieval.
- Validate source shape, unique plugin IDs, entity-source membership, reserved
  properties, relationship endpoints and finite vectors before graph writes.
- All nodes, relationships, copied provenance, vectors, inventory verification
  and completion creation share one bounded transaction. Writer errors raise
  and roll back rather than returning a successful pipeline result.
- Added callback transactions to `Neo4jConnection` with the existing bounded
  retry/backoff policy and support for a borrowed driver. Rollback failures
  cannot hide a lost commit acknowledgment; retries reconcile committed work.
- Matching committed inputs reuse the existing graph, including direct writer
  replay. Unattributed existing managed keys are never silently overwritten.

Validation: real pipeline replay, direct writer replay, unchanged entity and
relationship inventories, same-name entities across paragraphs, complete
rollback after node/relationship/completion writes, transient retry without
repeating model work, and lost commit acknowledgment all passed.

## 13.4 — Shared orchestration and process recovery

- Batch loading, query preload and the fixture command now use
  `ingest_paragraph`. Reconciliation is mandatory even without a manifest.
  The separate provenance synchronization write was removed.
- Matching commits skip extraction/embedding; unresolved source/graph states
  remain investigation cases. Manifest verification follows database inventory
  verification. Fixture summaries explicitly report reused work.
- Real subprocess tests exit immediately with an open uncommitted transaction
  (74), or after verified database commit immediately before checkpoint (73).
  Both preserve an on-disk `in_flight` entry for the restart test.
- Before-commit restart finds no graph and performs the pending write. After-
  commit restart verifies/skips the source with unchanged graph and no new
  provider events. Attempt counts and complete-record state agree with this.
- Missing and fresh manifests also skip committed work; fresh state preserves
  the previous manifest as a superseded file. Actual batch/preload/fixture
  commands share identity and refuse tampered graph state without model calls.

Validation: all subprocess, no-manifest, preload, fixture and batch tests passed.
An unresolved paragraph keeps its parent record incomplete.

## 13.5 — Final audit and validation

- Updated README, operational documentation, active ingestion contracts and
  coordination status. Documented schema privileges/setup, legacy handling,
  paragraph/model limits, transactional recovery and sequential-writer scope.
- Final deterministic command: `.venv/bin/python -m pytest tests/unit -q`
  → **277 passed** (45 more than the original 232-test baseline).
- Final combined live command:

  ```sh
  RUN_TASK13_ATOMIC_TESTS=1 RUN_TASK13_STOCK_WRITER_TESTS=1 TASK13_EVIDENCE_DIR=runs/task13-final .venv/bin/python -m pytest tests/integration/test_atomic_ingestion.py tests/integration/test_stock_writer_behavior.py -q
  ```

  → **21 passed, zero skips**: 19 atomic-ingestion checks and the two original
  stock-writer characterization checks. Neo4j Enterprise 2026.08.1 and the
  actual installed GraphRAG 1.21.0 pipeline were used with deterministic provider
  doubles. No paid provider or Ollama service calls were needed.
- An independent raw-Cypher audit (without classifier/inventory helpers) proved
  exactly one Document, one full-text Chunk, three distinct entities, one
  FROM_DOCUMENT link, three FROM_CHUNK links, two domain relationships and one
  completion record. Text, provenance and embedding values matched the source.
  Two ingestion attempts invoked extraction/embedding once. Relationship keys
  were distinct, and every endpoint remained source-scoped.
- All **21** reports confirm before/after graph snapshot hashes match after
  cleanup. No existing data was reset, migrated or certified as atomic.
- Local evidence: `runs/task13-final/validation-summary.json` and one sanitized
  `test_*.json` report for every live test, including manifests, provider events,
  graph inventories, raw audit output and cleanup checks. `git diff --check`
  passed.

## Operational limits and handoff

The supported contract is sequential ingestion with entity resolution disabled.
Unique database keys protect the write boundary; concurrency is not advertised
as supported. Full-paragraph provider limits still apply. Completion inventories
consume storage and require graph reads during verification. The live acceptance
baseline is macOS/Neo4j Enterprise 2026.08.1; other deployments need validation.

Existing unmarked data remains unverified and requires a separately reviewed
audit/repair or isolated rebuild. Legacy entities that lost all source identity
cannot be attributed retroactively. There is no automatic adoption, cleanup or
durable failed-chunk queue. Task 12 was not started. User acceptance is pending.
