# Retrieval Pipeline Hardening Goals

**Date:** 17 October 2026

**Last modified:** 18 October 2026

**Purpose of this file:**

Record the remaining retrieval-pipeline and MuSiQue ingestion hardening tasks
identified during the repository-wide audit on 17 October 2026. The current
pipeline is implemented and its deterministic test suite passes, but these
tasks are required before claiming that GraphRAG answers, evidence, citations,
performance, live artifacts, and long-running ingestion runs are reliable.

**Files that use this file:**

The active retrieval implementation is in
`src/graphrag_pipeline/retrieval.py`, `src/graphrag_pipeline/answer.py`, and
`src/graphrag_pipeline/query_command.py`. Retrieval contracts and correctness
artifacts are defined in `src/graphrag_pipeline/query.py` and
`src/graphrag_pipeline/musique_verification.py`. The broader project roadmap
is maintained in `agent-coordination/tasks.md` and the earlier vertical-slice
goal is maintained in `agent-coordination/tasks-goals-01-oct-2026.md`.
The MuSiQue batch ingestion path is implemented in
`src/graphrag_pipeline/ingestion/load_musique_dataset.py`, with record
adaptation and the `SimpleKGPipeline` boundary in
`src/graphrag_pipeline/ingestion/record_adapter.py` and
`src/graphrag_pipeline/extraction/`.

**Caveats:**

The audit found no current static diagnostics and the deterministic suite
reported 118 passing tests. The live Neo4j preload and query path has also
been exercised, but no dedicated repeatable end-to-end retrieval harness
currently captures that run. Existing worktree changes are user or
automation-owned and must be preserved while completing these tasks. The
batch loader at audit time reported chunk failures without resumable progress
or a durable retry queue; Tasks 11–13 completion logs below supersede that
historical baseline. These tasks cover reliability around the
`SimpleKGPipeline` ingestion and Neo4j write path; they do not claim that the
separate autonomous entity-resolution agent is implemented.

## Completion Log

### Task 1 — Seed-based full-component GraphRAG traversal (2026-10-17)

**Status:** Core traversal implemented, tested, and verified live. Scope was
revised mid-implementation based on live-graph findings (see below); the
multi-paragraph/record-root component design originally envisioned does not
apply to the current ingestion output, and this is documented rather than
assumed.

**What was implemented:**

- Live graph inventory (see `agent-coordination/neo4j/schema-recommendations.md`
  "Live graph inventory" section) found: (a) the vector-indexed `Chunk` node
  does not carry `chunk_id`/`record_id`/provenance — those live one hop away
  on `Document` via `FROM_DOCUMENT`; (b) no cross-chunk or cross-record
  entity connectivity exists today (`perform_entity_resolution=False`,
  per-chunk-scoped extraction), so each paragraph's entity subgraph is
  already a naturally self-contained component — no multi-paragraph
  component-boundary logic is required for the current ingestion design.
- Fixed a real, reproduced bug: `retrieval.py` previously read
  `chunk_id`/`record_id`/`document_id`/`title`/`source_file` directly off the
  vector-matched `Chunk` node, which always returns null, and would raise
  `RetrievalError: retrieval result is missing chunk_id` on any live query.
  Both `vector` and `vector_cypher` modes now always use
  `VectorCypherRetriever` with a `(Chunk)-[:FROM_DOCUMENT]->(Document)` hop
  (`_retrieval_query_vector`/`_retrieval_query_vector_cypher`); plain
  `VectorRetriever` is no longer used anywhere.
- `vector_cypher`/`graphrag` mode additionally traverses the seed chunk's own
  associated entity subgraph via `apoc.path.subgraphAll`, with no hop-depth
  ceiling, a dynamically-discovered relationship allowlist (`CALL
  db.relationshipTypes()`, excluding `FROM_DOCUMENT`/`FROM_CHUNK`/
  `NEXT_CHUNK`, sanitized against a safe-identifier regex since relationship
  names are LLM-extracted under `schema="FREE"`), and an explicit
  node-count resource budget (`RetrievalSettings.max_graph_nodes`, default
  200) that reports `graph_component_complete=False` rather than silently
  presenting a truncated subgraph as complete.
- `FROM_CHUNK` is traversed incoming-only (`FROM_CHUNK<`), which structurally
  guarantees traversal can never reach a different chunk/record's own
  text/provenance even in a hypothetical future where an entity is shared
  across chunks (verified with a synthetic integration fixture) — a shared
  entity's *other* `FROM_CHUNK` edge is always outgoing from its side, so it
  is never followed onward.
- Extended the evidence contract: `EvidenceChunk` gained `graph_nodes`,
  `graph_relationships`, and `graph_component_complete`
  (`graphrag_pipeline.query`), populated only for `vector_cypher`/`graphrag`
  mode and always empty/`True` for the `vector` no-expansion baseline.
- `RetrievalSettings` now validates `max_graph_nodes > 0` and that
  `backend="graphrag"` requires `mode="vector_cypher"`; the CLI
  (`query_command.py`) defaults `--retrieval-mode` from `--retrieval-backend`
  when not explicitly given (`_resolve_retrieval_mode`), and gained
  `--max-graph-nodes`.
- Fixed an unrelated but blocking live-environment issue found during
  verification: the `chunk_embedding` vector index was configured for 1536
  dimensions while the configured Ollama embedding model (and the already-
  stored `Chunk.embedding` values) are 768-dimensional; recreated at 768
  (cosine) to unblock live testing.

**Tests:**

- Unit (mocked, no network): `tests/unit/test_retrieval.py` — Document-hop
  query shape for both modes, dynamic relationship-type discovery and
  sanitization, `RetrievalSettings` validation, `graph_nodes`/
  `graph_relationships`/`graph_component_complete` parsing and rejection of
  malformed entries, vector-mode fields defaulting to empty/complete.
  `tests/unit/test_query_command.py` — `_resolve_retrieval_mode` defaulting.
- Integration (live Neo4j, auto-skip if unreachable):
  `tests/integration/test_retrieval_graph_traversal.py` — a synthetic fixture
  (cycle, a second path, a six-hop chain, a genuinely disconnected
  distractor, and a simulated future shared entity across two "records")
  proves: the full component is returned with no hop-depth ceiling and no
  duplication from the cycle; a distractor with no domain-relationship path
  is never reached; a shared entity's genuine connections correctly merge
  while chunk/document provenance stays seed-specific; a forced small
  resource budget reports `graph_component_complete=False` rather than a
  silently-truncated "complete" result.
- Live manual verification: `python -m graphrag_pipeline.query_command
  --question "Who developed the WorldWideWeb browser?" --retrieval-backend
  graphrag --top-k 3 --max-context-chunks 3` against the real single-record
  graph — succeeded end-to-end with correct citations and a populated
  `graph_nodes`/`graph_relationships` entity subgraph per evidence chunk.
- Full suite: `python -m pytest tests/unit tests/integration` — 131 passed
  (baseline 118 + 13 new/updated for this task).

**Explicitly deferred / not claimed:**

- Cross-record component-boundary enforcement beyond the structural
  `FROM_CHUNK<` guarantee was not built as a separate whitelist/property-
  filter mechanism, because the live schema currently has no path by which
  it would matter (no cross-chunk connectivity exists). This is documented
  as a scope decision, not an oversight — see schema-recommendations.md.
- Entity resolution (enabling genuine cross-paragraph components) remains
  out of scope per "Deferred Scope: Autonomous entity resolution."
  Real-record evaluation (Task 7) will accurately show GraphRAG adding no
  extra evidence beyond its seed chunk for today's ingested data; that is a
  true result of current ingestion, not a retrieval defect.
- Managed `GraphRAG.search()`'s internal answer prompt only sees chunk text,
  not the new `graph_nodes`/`graph_relationships` metadata — that metadata is
  retained for evidence/citation/verification purposes but does not change
  the managed answer's wording. Feeding graph structure into the answer
  prompt itself is left for Task 3 (answer contract) if needed.

### Task 2 — Identical GraphRAG answer context and artifact evidence (2026-10-17)

**Status:** Implemented, tested, and verified live.

**Root cause found:** the GraphRAG backend asked `GraphRAG.search()` for
`candidate_k = 4 * top_k` rows, which the managed answer model read in full,
and only afterwards trimmed the retained evidence to `max_context_chunks`.
The answer could therefore rest on chunks absent from the response, citations,
and artifact. Separately, `generate_answer` used a fixed default evidence
bound (5) unrelated to the configured `--max-context-chunks`, so response
evidence could silently differ from the artifact's retrieval evidence.

**What was implemented:**

- `retrieval.py`: GraphRAG is no longer over-fetched; it requests
  `RetrievalSettings.graphrag_seed_k = min(top_k, max_context_chunks)` rows.
  If the managed context still holds more distinct chunks than
  `max_context_chunks`, `retrieve` raises `RetrievalError` instead of trimming
  evidence the answer already used. Duplicate rows collapse to one evidence
  chunk without tripping the check. VectorRAG keeps its over-fetch because its
  answer is generated afterwards from the already-bounded list.
- `answer.py`: a pre-generated (GraphRAG) answer whose retrieval evidence
  exceeds `max_evidence_chunks` becomes a `failed` response rather than being
  sliced to a smaller list. VectorRAG citation validation is unchanged.
- `query_command.py`: `generate_answer` now receives
  `AnswerSettings(max_evidence_chunks=retrieval_settings.max_context_chunks)`,
  so the prompt, response, and artifact share one bound.
- `musique_verification.py`: `CorrectnessReport` gained
  `evidence_boundary_consistent`, true only when response evidence chunk IDs
  equal the artifact's retrieval evidence chunk IDs.

**Tests:**

- `tests/unit/test_retrieval.py`: GraphRAG seed count is capped at the
  evidence bound; an over-fetched answer-bearing third chunk raises
  `RetrievalError` (the regression the task asks for); duplicate context rows
  do not hide or inflate evidence.
- `tests/unit/test_answer.py`: a pre-generated answer that saw more chunks than
  the bound fails with no answer; within the bound, response evidence and
  citations equal the retrieved evidence.
- `tests/unit/test_musique_verification.py`: `evidence_boundary_consistent`
  is true for matching and false for differing evidence.
- `tests/unit/test_query_command.py`: with three retrieved rows and a bound of
  two, the prompt excludes the third, and response and artifact evidence match.
- Live: `graphrag-query --retrieval-backend graphrag --top-k 8
  --max-context-chunks 3 --artifact ...` returned status `succeeded` with
  identical response, artifact, and citation IDs and
  `evidence_boundary_consistent: true`.
- Full suite: 138 passed.

**Caveats:**

- A GraphRAG `--top-k` larger than `--max-context-chunks` is now silently
  capped to the context bound for the managed answer path. This is recorded
  here rather than in the artifact; Task 6 should add the effective seed count
  to run metadata.
- Each citation still equals every evidence chunk for GraphRAG; narrowing
  citations to the chunks that support the answer is Task 3.

### Task 3 — One strict answer contract for both backends (2026-10-17)

**Status:** Implemented, tested, and verified live.

**What was implemented:**

- New `src/graphrag_pipeline/answer_context.py` holds the single answer
  instruction template, system instruction, and evidence renderer
  (`[chunk_id] text`, plus entity/relationship lines and a truncation note
  when a chunk carries a graph component).
- VectorRAG builds its prompt from that template. GraphRAG's
  `build_graphrag` now passes the same template to `GraphRAG.search` as a
  `RagTemplate`, and each retrieved row's `content` is rendered by the same
  function, so the managed context equals the VectorRAG context byte for byte.
  Raw chunk text moved to `metadata["text"]` so `EvidenceChunk.text` stays
  the unmodified paragraph that graph verification compares.
- `answer.py`: new `validate_generated_answer` is used by both backends. It
  rejects empty or malformed output, extra fields, empty citations, blank
  answers, and citations outside the evidence shown to the model, and
  de-duplicates repeated citations. GraphRAG no longer cites every evidence
  chunk; it cites only what the model declares. One surrounding markdown
  code fence is tolerated for both backends.
- `retrieval.py`: a blank managed answer is no longer a retrieval error; it
  reaches the answer boundary and becomes a `failed` response that keeps the
  evidence. `insufficient_evidence` is still assigned only for empty evidence,
  before any validation.

**Tests:**

- `tests/unit/test_answer.py`: the same seven invalid-output cases (empty,
  blank, malformed, missing citations, extra field, blank answer, citation
  outside evidence) parametrized across both backends, each asserting
  `failed`, no answer or citations, and retained evidence; empty evidence
  gives `insufficient_evidence` for both; subset citation with duplicate
  collapse for both; code-fence tolerance; VectorRAG prompt equals the shared
  template.
- `tests/unit/test_retrieval.py`: formatted row content is citable and keeps
  raw text in metadata; joined managed context equals the VectorRAG
  rendering including graph lines; truncated components are labelled;
  `build_graphrag` uses the shared template; a blank managed answer is
  preserved for validation.
- Live: `--retrieval-backend graphrag` and `vectorrag` on the same question
  both returned `succeeded`, answer "Tim Berners-Lee", three evidence chunks,
  and one citation (paragraph 0).
- Full suite: 163 passed.

**Caveats:**

- For MuSiQue, GraphRAG's `query_text` carries the source-title hints into the
  managed prompt (VectorRAG's prompt uses the plain question). `GraphRAG.search`
  has a single `query_text`, so this cannot be separated without replacing
  the managed call. Left for Task 5, which owns the title-hint behavior.
- The VectorRAG prompt now also shows graph lines when run in
  `vector_cypher` mode; `vector` mode (the baseline) is unchanged.
- The contract has no way for the model to say "insufficient evidence"; a
  model that cannot answer must still cite, or the response fails.
- Observed live with `qwen3:4b` on a 3-hop MuSiQue record: the model wrote
  prose followed by a fenced JSON block, which the strict contract rejects as
  `failed` ("malformed JSON output") although the embedded JSON was valid.
  This is the contract working as designed; a more compliant model or a
  model-side JSON mode is the remedy, not a looser parser.

### Task 4 — Index-aware MuSiQue record filtering (2026-10-17)

**Status:** Implemented, tested, and verified live.

**What was found:** Neo4j 2026.x records an index's filterable properties in
its `properties` list (after the vector property), but neo4j-graphrag 1.21
only reads `vector.filterable_properties`. The stock retriever therefore
never recognised a filterable `record_id` and used a fallback path that is
broken for `VectorCypherRetriever`. `Chunk` nodes also lacked `record_id`
(it lived only on `Document`), so an index filter could not match them.

**What was implemented:**

- `retrieval.py`: `RecordFilterRetriever` (a `VectorCypherRetriever`) reads the
  filterable properties from the index's `properties` and raises
  `RetrievalError` for any filter on an undeclared property, so a record
  constraint is never applied through an unreported fallback.
  `build_retriever` now returns it for both modes.
- `vector_index_setup.py` (new): `python -m graphrag_pipeline.vector_index_setup`
  reports, and with `--apply` repairs, the two requirements: it backfills
  `Chunk.record_id` from `Document` and recreates the vector index with the
  existing dimensions and similarity plus `record_id` filterable. Without
  `--apply` it changes nothing; repeated `--apply` is a no-op.
- `retrieval_prerequisites.py`: the report gains `vector_index.filterable_properties`
  and a `record_filter` section (`not_required` or `in_index`, whether the index
  declares the property, per-record chunk counts). Verification fails early
  when a filter is required but undeclared, a requested record has no chunks,
  or some of its chunks lack `record_id` (they would be silently excluded).
  There is no degraded mode: the supported fallback was judged unreliable, so
  the state is fail-fast. `--filter-record-id` checks loaded records from the
  command line.
- `query_command.py`: record-constrained runs require the filter; records
  about to be preloaded are excluded from the chunk-coverage check because
  they have no chunks yet.
- The live index was repaired with the setup command and now reports
  `filterable_properties: ["record_id"]`.

**Tests:**

- `tests/unit/test_vector_index_setup.py`: index parsing (both metadata
  locations), report-only mode changes nothing, `--apply` backfills and
  recreates preserving dimensions, no recreation when already filterable, and
  unsafe identifiers are rejected.
- `tests/unit/test_retrieval_prerequisites.py`: supported, declared-through-
  config, missing-from-index (fails when required, reported when not),
  partial `Chunk.record_id` coverage, record without chunks, and unsafe
  identifier cases.
- `tests/unit/test_retrieval.py`: the retriever rejects undeclared filters,
  delegates declared ones to the index, allows unfiltered search, and reads
  filterable properties from `properties`.
- `tests/integration/test_record_filtered_vector_search.py` (live Neo4j,
  temporary labelled fixtures): the filter uses the index and excludes nearer
  records from other records, filtered top-k is filled from the requested
  record only even when unrelated records hold near-identical text, and an
  index without the declaration fails instead of falling back.
- Live: a record-constrained GraphRAG query on `3hop2__90098_156850_10557`
  retrieved only that record's chunks (paragraphs 3, 15, 10; two of the three
  gold supporting paragraphs). Full suite: 185 passed.

**Measurements (live graph, 2 records, 35 chunks, 15 runs, median):**

- Latency: in-index filtered search 2.4 ms versus unfiltered search with a
  top-20 over-fetch and filtering afterwards 2.6 ms; no meaningful difference
  at this scale. The benefit is expected to appear only with many records.
- Candidate behavior: asking for 5 results, filtering after an unfiltered
  top-5 returned 4/5 chunks from the target record for one question because
  another record's chunk ranked inside the top 5; it needed k=10 to recover
  5/5. In-index filtering returned 5/5 for both questions.

**Caveats:**

- Recreating the index makes vector search unavailable until it is `ONLINE`;
  the setup command waits for it but is not safe to run during queries.
- The measurements use two records. They show candidate correctness, not
  large-corpus latency; Task 7 should re-measure on the evaluation set.
- Corpus-wide (unfiltered) retrieval is unaffected and needs no declaration.

### Task 11 — Persist resumable MuSiQue ingestion progress (2026-10-18)

**Status:** IMPLEMENTATION AND VALIDATION COMPLETE; ready for user acceptance.
Live stop/resume, independent Neo4j duplicate/provenance audit, final
documentation, and deterministic tests passed. The user authorized finishing
all remaining subtasks together, superseding per-subtask pauses.
Tasks 13 (idempotent writes) then 12 (retry queue) have not been
started (agreed order: 11 → 13 → 12). Approved plan:
`implementation_plan.md` in the Antigravity artifact directory for this
conversation (`a5f19288-eccf-4a8b-b1e8-091c0511dc24`).

**What was implemented:**

- `src/graphrag_pipeline/ingestion/progress_manifest.py` (new): `ProgressManifest`
  and `ManifestStore`. One JSON file per source (dataset SHA-256 + split) at
  `runs/ingestion-state/<split>-<sha prefix>.json` (`runs/` is already
  gitignored) or `--manifest-path`. Atomic writes (temp file, fsync,
  `os.replace`). Chunk statuses: `pending`, `in_flight`, `verified`, `failed`,
  `needs_investigation`. A record is complete only when every expected chunk is
  `verified`. A manifest for a different checksum/split raises
  `ManifestSourceMismatch`. `in_flight` chunks found on load become `pending`
  and are listed as interrupted. `--fresh-manifest` moves an old file aside
  (`*.superseded-<ts>`), never deletes it.
- `src/graphrag_pipeline/ingestion/reconcile.py` (new): read-only
  `classify_chunk` returning `absent`/`complete`/`partial`/`conflict` for a
  chunk against Neo4j (Document provenance, exactly one Chunk, Chunk text equal
  to the paragraph, copied provenance). This is the classifier Task 13 builds
  on; report-and-skip of partial/conflict chunks is already wired in.
- `load_musique_dataset.py`: `run_requests` gained optional `store`,
  `stop_requested`, `outcome` (`RunOutcome`) parameters and still returns
  `(succeeded, failed)`. With a store, each chunk is classified against Neo4j
  first (`complete` → skipped and marked verified; `partial`/`conflict` →
  reported as `needs_investigation` and skipped, never deleted/overwritten;
  `absent` → flushed `in_flight`, run, provenance synchronized, re-classified,
  and marked `verified` only if `complete`). A stop request is polled before
  each chunk. `LoadSummary`/`PreloadSummary` report skipped, needs-investigation,
  interrupted, stopped-early, not-attempted counts and manifest paths, and
  `succeeded` is false for failures, investigation cases, invalid records or an
  early stop; successfully reconciled interruptions may still succeed.
  `load_dataset` and `load_selected_records`
  (query `--preload`) use the store. CLI: `--manifest-path`, `--fresh-manifest`;
  first SIGINT/SIGTERM finishes the running chunk and exits 130 if work remains;
  a repeated same signal interrupts immediately; manifest errors exit 2.

**Original handoff tests (205 passed; final validation below: 232 passed):**

- `tests/unit/test_progress_manifest.py`, `tests/unit/test_reconcile.py`,
  `tests/unit/test_load_resume.py` (new): atomic-write failure, checksum/split
  mismatch, record-complete rule, in-flight demotion, clean stop then "new
  process" resume skipping verified work, failed chunk retried alone on resume,
  crash after Neo4j write before checkpoint (no provider call, no rewrite),
  crash before write (re-ingested), pre-existing partial chunk reported and
  skipped, post-write verification failure, deleted-then-reingested chunk,
  changed-checksum rejection.
- `tests/unit/test_query_preload.py`: the live-preload test's fake runner now
  registers/verifies via the store.

**Original live validation (real Ollama + live Neo4j, record 1 of the dev file,
20 chunks, graph was empty beforehand):** a live load wrote verified chunks
(post-write Neo4j check passed, so real `Chunk.text` equals the paragraph text
and one Chunk per paragraph held for those chunks). SIGINT was sent mid-run; the
loader printed the clean-stop message and was finishing its current chunk. At
the last check the manifest was `verified: 7, pending: 12, in_flight: 1`.

**Remaining for Task 11 (pick up here) — implementation now complete:**

Continuation update (2026-10-18): subtask 1 was verified and the user authorized
continuation. Recovered original output confirms exit
130, 8 succeeded, 12 not attempted, 0 failed, 0 loaded records, and a clean stop.
The pre-resume raw manifest contained 8 verified and 12 pending chunks, with no in-flight
entries; checksum/split and all expected chunk IDs match the actual dataset.
The clean-stop regression now checks on-disk statuses before manifest reload
can demote in-flight entries. Focused ingestion/preload validation: 37 passed.
Subtask 2 is now complete:

- Live preflight classified exactly 8 requests complete and 12 absent in Neo4j;
  Ollama's configured `qwen3:4b-instruct` and `nomic-embed-text` models were
  available. No source, configuration, or progress reset was needed.
- Ran the real CLI entry point with `--records 1 --no-progress` through a
  delegating observation harness. Original manifest preserved; stdout, stderr,
  exit code, before/after manifests, per-chunk pipeline events, and assertions
  saved under `runs/task11-resume-b4bf236db382/` (gitignored).
- Exit **0**; **8 skipped**, **12 succeeded**, **1 record loaded**; no failures,
  investigation cases, interrupted chunks, or unattempted work. Elapsed
  **146.661 seconds**; stderr empty.
- Exact invocation list: paragraphs **8–19** only, each returning successfully.
  Previously verified paragraphs **0–7** made no extraction pipeline calls.
- Final manifest: **20 verified**, each with one attempt; skipped entries'
  attempt counts unchanged. Each newly written chunk passed the loader's
  post-write Neo4j classification before being marked verified.
- Six live evidence assertions passed. Extended the deterministic clean-stop
  and resume test to check persisted post-resume statuses and unchanged skipped
  attempts; focused ingestion/preload suite: **37 passed**.

Final continuation completed subtasks 3–5:

- **Independent live Neo4j audit passed:** both whole-database and selected
  source/record totals are **20 Documents and 20 Chunks**. All 20 expected
  stable IDs have exactly one Document, one full-text Chunk, and one
  `FROM_DOCUMENT` link to the expected Document. No duplicate source nodes or
  links were found. Every source text, Document metadata field, and copied Chunk
  provenance field matched the actual dataset requests. All requests classify
  complete. The audit was read-only and confirmed unchanged manifest bytes.
  Evidence: `runs/task11-resume-b4bf236db382/graph-audit.json`; local audit
  script: `runs/task11_graph_audit.py`.
- **Closed a provenance verification gap:** reconciliation now also checks
  Document title, dataset, split and source file, and Chunk document ID, source
  file, source line and title. Existing ID/checksum/index checks remain. An
  optional absent title remains valid. Added 19 regression cases covering all
  10 Document metadata fields, all 8 copied Chunk fields, and absent title.
  The live audit passed again using this strengthened classifier.
- **CLI and actual process tests:** added `tests/unit/test_ingestion_cli.py`
  with 4 outcome/flag tests (0/1/2/130, custom/fresh manifest wiring) and 4
  POSIX subprocess cases (SIGINT/SIGTERM, single/repeated signal). Single-signal
  cases finish the in-flight chunk, persist clean state, exit 130 and resume
  only pending work in a new process. Repeated-signal cases terminate with the
  default signal status, preserve the in-flight checkpoint, reconcile unwritten
  work after restart, and retain attempt counts. Fake provider/graph boundaries
  avoid network/cost while exercising the real CLI, loader and durable manifest.
  All 4 subprocess cases passed on macOS; they skip on Windows.
- **Documentation finished:** README and `documentation.md` now explain both
  manifest flags, automatic resume, full source identity, local file lifecycle,
  atomic checkpoints, safe and immediate stops, handled exit codes, summary
  fields, partial/conflict investigation, query-preload behavior, and current
  limitations. Corrected README's per-paragraph pipeline construction wording
  to describe the actual shared pipeline instance.
- **Final checks passed:** actual CLI help matches the new flag reference;
  documented dry-run planned 20 chunks with no loaded chunks/manifest paths and
  exited 0. `.venv/bin/python -m pytest tests/unit -q` → **232 passed in
  3.08 seconds**, no skipped tests. `git diff --check` passed. This adds 27 test
  cases over the 205-test handoff, plus stronger existing resume assertions.
- No further provider calls or graph writes were made during this final audit
  and documentation stage. Task 13 and Task 12 have not been started.

See [the remainder plan and evidence](task-11-remainder-plan.md) for full staged
validation. Task 11 is ready for user acceptance; this sequential-run validation
does not establish Task 13's database-level idempotent-write guarantee.

1. **Complete:** Inspect the stopped loader's JSON summary/exit code 130 and manifest
   (`runs/ingestion-state/full_dev-8cab31d56a3a1c4e.json`); confirm no chunk is
   left `in_flight` after a clean stop.
2. **Complete:** Re-run `python -m graphrag_pipeline.ingestion.load_musique_dataset --records 1
   --no-progress`, confirm the verified chunks are skipped without provider
   calls, the rest are loaded, and the record reports as loaded.
3. **Complete:** Confirm in Neo4j: 20 `Document` and 20 `Chunk` nodes, no duplicates.
4. **Complete:** Update `README.md` / `documentation.md` (new flags, manifest
   lifecycle, exit codes), validate CLI behavior, and pass the full unit suite.
5. **Handoff complete; user acceptance pending:** report all Task 11 work and
   verification. Do not start Task 13 without an explicit user instruction.

**Caveats:**

- Resume trusts Neo4j over the manifest; a `partial` chunk (for example a
  crash between the pipeline write and provenance synchronization) is reported
  and skipped until the user deletes its nodes. There is no automatic repair.
- No unique constraint exists yet (Task 13), so concurrent writers are not
  protected. No retry queue exists yet (Task 12): failed chunks are simply
  retried on the next run.
- `GRAPH_RAG_CHUNK_SIZE` defaults to 10000; a paragraph longer than that would
  split into several chunks and be classified `partial` (Task 13 plans to size
  the splitter to the longest paragraph).

### Task 13 — Idempotent paragraph ingestion (2026-10-18)

**Status:** IMPLEMENTATION AND VALIDATION COMPLETE; ready for user acceptance.
After 13.1 the user authorized completing all remaining subtasks together,
superseding the original per-subtask stops.
See [the implementation sequence and writer evidence](task-13-remainder-plan.md).

- Added opt-in live characterization tests using the actual stock
  `SimpleKGPipeline`, deterministic LLM/embedding doubles, and real Neo4j.
- Identical paragraph replay doubled Documents, Chunks, extracted entities,
  source/domain relationships and embeddings. Same-name entities remained
  separate with the configured resolver disabled.
- Injecting a relationship-write client error left five committed nodes and
  returned writer `FAILURE` without raising, even with extraction
  `on_error="RAISE"`.
- Replaced production persistence with a custom atomic `KGWriter`: source
  nodes, entities, relationships, embeddings, copied provenance and versioned
  completion inventory share one bounded transaction. Scoped database unique
  keys protect the identity boundary; writer failures roll back and raise.
- Added exact canonical request fingerprints and one unchanged full-paragraph
  Chunk. Reused IDs with changed text/provenance conflict. Legacy, partial and
  damaged graph data is preserved for investigation instead of certified.
- Batch, query preload and fixture commands share mandatory reconciliation,
  even without a manifest. Real subprocess tests proved pre-commit rollback,
  post-commit/pre-checkpoint recovery without provider work, and safe reuse
  with missing/fresh manifests. Entity resolution remains disabled.
- **277 unit tests and 21 live tests passed**, with no skipped acceptance tests.
  Replay, direct writer reuse, malformed graphs, uniqueness, schema mismatch,
  transient retry, lost commit acknowledgment, changed input and graph tampering
  are covered. An independent raw-Cypher source/graph audit passed.
- All live fixture cleanup snapshot hashes matched the original graph.
  Sanitized evidence: `runs/task13-final/validation-summary.json` and per-test
  reports. `git diff --check` passed. No legacy migration or reset was performed.
- Sequential ingestion remains the supported contract. Task 12 has not started;
  user acceptance of Task 13 has not been presumed.

### Task 12 — Durable failed-chunk queue (2026-10-18)

**Status:** IMPLEMENTATION AND REQUIRED VALIDATION COMPLETE; ready for user
acceptance. The agent stopped after 12.1 and 12.2; the user then explicitly
authorized all remaining subtasks together, superseding the remaining stops.
See [the complete Task 12 report](task-12-remainder-plan.md).

- Durable source queues retain validated requests/siblings, fingerprints,
  attempts, policies, categories, timestamps, model/configuration names and dead
  letters. Default budgets include three total attempts with five/thirty-second
  delays. Requeue requires named terminal IDs and preserves lifetime history.
- Batch, preload and fixture ingestion use mandatory reconciliation and durable
  tracking. Queue/checkpoint failures stop processing. Committed graph effects
  are verified/reused; unsafe graph state stays preserved and unresolved.
- Selected-range resume does not drain unrelated failures or retry within the
  same invocation. Exact paragraph IDs drive processed/new/used/failed/deferred
  chunk counts and complete/partial/unstarted records; no chunk-total division
  or record-boundary bookkeeping is used to claim completion.
- Added offline inspect/requeue and bounded source-independent drain commands.
  Existing linked progress manifests synchronize after identity checks; every
  stored sibling receives independent verification before a completion claim.
- Legacy failure/in-flight attempt counts survive upgrades and fresh manifests.
  CLI process-death and SIGINT/SIGTERM tests verify rollback, commit/checkpoint
  recovery, clean stops, exhaustion and reviewed recovery without duplicate work.
- **429 unit tests and 41 live Neo4j tests passed**, zero skipped required live
  checks, with deterministic providers. Every cleanup hash matched existing
  graph state. No paid provider calls or existing graph resets occurred.
- Evidence: `runs/task12-final/validation-summary.json` and per-test reports.
  README, operational documentation, environment examples and active ingestion
  contracts are updated. Sequential ingestion/entity-resolution limits remain.
  Other hardening tasks in this document retain their independent status.

## Goal

Harden the existing Neo4j GraphRAG and VectorRAG question-to-answer path.
GraphRAG must use vector search to find seed nodes and then traverse the full
seed-associated component without a hop-depth limit, while keeping graph scope,
evidence, citations, performance safeguards, failure states, and live behavior
explicit and reproducible. VectorRAG remains the vector-only comparison path.

The generated answer must be grounded in exactly the graph/chunk evidence
retained in the response and artifact; MuSiQue record/component isolation must
be enforced and verified.

## Current Baseline

- GraphRAG and VectorRAG backend selection is implemented.
- MuSiQue question selection preserves the exact source question and record
  provenance.
- Optional preload reuses the existing `SimpleKGPipeline` ingestion path.
- Retrieval over-fetches, deduplicates, filters by record, and uses selected
  source-title hints for MuSiQue disambiguation.
- `vector_cypher` currently projects properties from vector-matched nodes; it
  does not yet traverse relationships or return a seed-associated component.
- The GraphRAG path must be changed to use `VectorCypherRetriever` for vector
  seed search followed by complete component traversal. No fixed maximum hop
  depth defines completion; component scope and resource budgets do.
- The live graph's component boundaries and relationship vocabulary must be
  inventoried and verified before the traversal query can be considered safe.
- Correctness verification supports both `supporting_facts` and paragraph-level
  `is_supporting` labels.
- The latest deterministic test result was `118 passed, 0 failed`.
- The live vector index is named `chunk_embedding`, targets `Chunk.embedding`,
  uses 1536 dimensions, and uses cosine similarity.

## GraphRAG Architecture Decision

The GraphRAG retrieval path uses Neo4j `VectorCypherRetriever` in two explicit
stages:

1. Vector-search `Chunk` nodes as seed candidates for the question.
2. For each seed, traverse the complete seed-associated orphan component,
   without a hop-depth limit, and return its nodes, relationships, source
   chunks, and provenance as structured evidence.

The required traversal policy is to **traverse the full seed-associated
component without a hop-depth limit, scoped to the orphan subgraph or record
root, with relationship and result-size safeguards**. The record/root or
verified component boundary prevents traversal into other records. An explicit
relationship allowlist must reflect the observed Neo4j schema. A node, edge,
or byte/token budget is a resource safeguard, not a hop limit: if a budget
prevents returning the complete component, retrieval must report that outcome
explicitly and must not present a partial component as complete.

The application must merge duplicate components reached from multiple vector
seeds while retaining each seed's score and identity. Component nodes and
relationships must remain distinct from chunk evidence so graph structure is
not lost through chunk-ID deduplication. Answer citations must resolve to
source chunks/provenance supporting the returned graph context. VectorRAG
continues to retrieve chunks without graph expansion and serves as the
comparison baseline.

## Comparative Evaluation Protocol

Evaluate retrieval quality separately from answer quality so a change in
answer accuracy can be attributed to retrieved context rather than different
generation settings. Run the same questions against the same frozen Neo4j
snapshot, vector index, embedding model, vector seed count/top-k, and applicable
record constraint. For MuSiQue, run a record-constrained evaluation and, when
the corpus permits it, a separate corpus-wide evaluation; report them
separately. Gold answers and supporting-fact labels are evaluation-only and
must not be supplied to either retriever or answer model.

The retrieval comparison is:

- VectorRAG context: the source chunks returned by vector search.
- GraphRAG context: the complete seed-associated component returned by vector
  search plus `VectorCypherRetriever` traversal.

For answer comparisons, use the same answer model, system/prompt policy,
generation parameters, token budget, output schema, citation validation, and
failure-status rules. The context representation may differ to preserve
GraphRAG nodes and relationships, but the context-selection rule and maximum
answer-context token budget must be applied consistently. Record both the
complete retrieved evidence and the exact selected context sent to the model;
do not imply that the model saw a full component if context selection omitted
part of it. A traversal result-size safeguard that stops graph expansion must
remain an explicit incomplete-retrieval outcome, distinct from post-retrieval
answer-context selection.

Measure context quality against gold supporting paragraphs mapped to persisted
Neo4j chunks. Report support coverage/recall, precision or irrelevant-context
rate, whether GraphRAG expansion added supporting evidence not present in its
seed chunks, retrieved node/edge/chunk counts, answer-context token count,
latency, and provider cost where available. Measure answer quality separately
using MuSiQue answer/alias correctness and citation support. Preserve per-query
paired results and aggregate results; report failures and incomplete component
retrieval rather than dropping them from denominators without explanation.

Before implementation, inspect the persisted graph to verify whether MuSiQue
components are truly isolated by `record_id`, whether any entity is shared
across records, and which relationship types connect chunks, documents, and
extracted entities. If `record_id` is not a reliable component boundary, use
or establish a record-root/component identity and enforce it during traversal.

## Active Task Sequence

### 1. Implement seed-based full-component GraphRAG traversal

**Priority:** High

- Inventory labels, relationship types, provenance placement, and component
  connectivity in the persisted graph; verify whether record IDs or explicit
  record roots reliably isolate orphan components, including shared-entity
  cases.
- Configure the GraphRAG retrieval mode to use `VectorCypherRetriever`: vector
  search `Chunk` seed nodes, then traverse the full associated component
  without imposing a maximum hop depth.
- Scope traversal to the verified orphan component or record root and use an
  explicit relationship allowlist derived from the graph inventory. Do not
  traverse unrelated records or rely solely on vector-index filtering to
  enforce the graph boundary.
- Return structured seed, node, relationship, chunk-text, and provenance data.
  Merge duplicate components found from multiple seeds without dropping graph
  edges; preserve the seed IDs and similarity scores.
- Add configurable result-size/resource safeguards. If a safeguard prevents
  returning the full component, return an explicit incomplete/failure outcome
  rather than silently treating a partial component as complete evidence.
- Extend retrieval/evidence contracts and the answer-context builder so the
  model receives graph structure together with source-backed chunk evidence;
  validate citations against the source chunks actually supplied.
- Keep VectorRAG as a vector-only baseline and test that graph expansion is
  exclusive to the GraphRAG traversal mode.
- Add deterministic graph fixtures with cycles, multiple paths, a component
  deeper than a small arbitrary hop count, duplicate seeds in one component,
  and disconnected/cross-record distractors.

**Acceptance criteria:**

- GraphRAG performs vector seed search followed by complete traversal of the
  seed's verified component, with no configured hop-depth ceiling.
- Cycles and alternate paths do not duplicate component nodes or relationships
  in returned evidence.
- Traversal does not include nodes outside the seed component/record boundary.
- Resource-limit outcomes are explicit and never mislabeled as complete
  component retrieval.
- Returned graph structure retains source chunk provenance, and answer
  citations can identify only source evidence present in the answer context.
- Tests prove that a connected node beyond a small fixed hop count is returned,
  while disconnected and cross-record nodes are excluded.

### 2. Make GraphRAG answer context and artifact evidence identical

**Priority:** High

- Ensure GraphRAG generates its answer from the same bounded evidence list
  that is returned in `QueryResponse` and written to the run artifact.
- Do not allow over-fetched candidates used internally by GraphRAG to support
  an answer while being omitted from final evidence or citations.
- Preserve the existing VectorRAG citation validation behavior.
- Add a regression test where an over-fetched candidate contains the answer and
  prove that the final answer path either includes it in bounded evidence or
  does not use it.
- Confirm that response citations identify only evidence retained in the
  response.

**Acceptance criteria:**

- GraphRAG answer context is bounded by the configured context limit before
  answer generation.
- The response, correctness report, and artifact contain the same evidence
  boundary.
- A test fails if an answer depends on evidence omitted from the artifact.

### 3. Apply one strict answer contract to both backends

**Priority:** High

- Define the structured answer and citation contract for GraphRAG as well as
  VectorRAG.
- Validate GraphRAG answer content for empty output, malformed output, and
  citations outside the returned evidence.
- Ensure GraphRAG does not automatically cite every returned chunk when only a
  subset supports the generated answer.
- Preserve explicit `succeeded`, `insufficient_evidence`, and `failed` statuses.
- Add deterministic tests for malformed GraphRAG output, unsupported claims,
  invalid citations, and empty evidence.

**Acceptance criteria:**

- Both backends produce the same `QueryResponse` contract.
- Invalid provider output becomes a structured failure rather than a
  successful answer.
- Citation validation is covered by unit tests for both backends.

### 4. Make MuSiQue record filtering index-aware

**Priority:** Medium

- Add `record_id` to the Neo4j vector-index filterable properties required by
  the retrieval implementation, or document and measure the supported fallback
  if the installed Neo4j version cannot provide this capability.
- Extend prerequisite validation to inspect and report filterability of
  `record_id`.
- Fail early or emit an explicit degraded-mode status when record filtering
  cannot be applied as configured.
- Measure retrieval latency and candidate behavior before and after the index
  change.
- Add deterministic prerequisite tests for supported, missing, and degraded
  filter configuration.

**Acceptance criteria:**

- A live MuSiQue retrieval does not rely on an unreported filter fallback.
- The prerequisite report records the filterability state.
- Record-constrained retrieval remains correct when unrelated records contain
  highly similar text.

### 5. Replace heuristic title disambiguation with tested retrieval behavior

**Priority:** Medium

- Document the purpose and limits of source-title hints in MuSiQue retrieval.
- Add fixture cases containing ambiguous titles, duplicate titles, parenthesized
  titles, and distractor records.
- Verify that title hints improve the intended record without suppressing valid
  evidence or introducing unrelated evidence.
- Record retrieval configuration and selected hints in debug-safe run metadata
  when needed for reproducibility.
- Keep seed count, component identity, result-size budgets, and context limits
  explicit and deterministic; do not impose a hop-depth ceiling on component
  traversal.

**Acceptance criteria:**

- Ambiguous-title tests demonstrate the intended record remains retrievable.
- Distractor and duplicate-title tests do not produce cross-record evidence.
- Any known ranking limitation is documented rather than treated as a verified
  full-component traversal guarantee.

### 6. Add a repeatable retrieval end-to-end harness

**Priority:** Medium

- Add a focused command or integration test that queries an already-loaded
  graph without implicitly re-running ingestion.
- Cover both VectorRAG and GraphRAG backend selection.
- For GraphRAG, verify vector seed selection, full-component traversal,
  component isolation, cycle/duplicate handling, and graph evidence provenance.
- For VectorRAG, verify the vector-only baseline does not expand relationships.
- Record Neo4j version, `neo4j-graphrag` version, model configuration names,
  vector-index configuration, request provenance, evidence, answer status, and
  correctness output.
- Skip cleanly when required live services or credentials are unavailable.
- Keep deterministic unit tests as the default validation path and isolate
  network or paid-provider tests behind explicit integration conditions.

**Acceptance criteria:**

- A documented command reproduces a live retrieval run against an existing
  graph.
- The harness proves query-only mode does not invoke ingestion.
- At least one recorded run validates evidence provenance and answer/artifact
  alignment.

### 7. Add a controlled VectorRAG-versus-GraphRAG evaluation

**Priority:** High

- Build a fixed, labeled evaluation set from MuSiQue records with usable gold
  answers and supporting facts; map supporting paragraphs to the persisted
  Neo4j chunks without exposing labels to retrieval or generation.
- Run paired VectorRAG and GraphRAG retrievals against the same graph snapshot,
  embedding model, index, question text, vector seed count/top-k, and record
  constraint. Keep record-constrained and corpus-wide results as separate
  evaluation conditions.
- Compare retrieved context before answer-context selection: vector seed
  chunks for VectorRAG versus the complete seed-associated component for
  GraphRAG. Measure supporting-evidence coverage/recall, precision or
  irrelevant-context rate, and whether graph expansion adds supporting
  evidence absent from its seeds.
- Use the same answer model, prompt policy, generation settings, output schema,
  citation validator, and token budget for both answer paths. Preserve the
  backend-specific context representation needed to expose GraphRAG structure.
- Record the complete retrieval output and the exact token-budgeted context sent
  to the answer model as separate fields. Report GraphRAG traversal-incomplete
  outcomes separately from complete traversal followed by context selection.
- Measure answer/alias correctness, citation support, latency, context size,
  and provider cost where available. Keep paired per-question results and
  define how failed, insufficient-evidence, and incomplete-retrieval cases
  affect aggregate metrics.
- Add deterministic evaluation tests with known supporting chunks, distractors,
  multi-hop evidence, and cases where graph expansion adds no useful evidence.

**Acceptance criteria:**

- Both retrievers are evaluated on identical requests and a documented,
  immutable graph/index/model configuration for each paired run.
- Retrieval-context metrics and answer metrics are reported separately.
- The evaluation can demonstrate whether GraphRAG adds gold supporting
  evidence beyond its vector seeds, and whether any added evidence improves
  answer correctness/citation support under the same answer-context budget.
- The exact model-visible context, retrieval output, citations, status, and
  evaluation labels/results are reproducibly associated by trace ID; labels
  are not passed to retrieval or generation.
- Per-query failures and incomplete traversals remain visible in reports and
  are handled by documented aggregate-metric rules.

### 8. Make artifact generation resilient and remove duplicate record loading

**Priority:** Medium

- Remove the unused raw-record load from the per-query answer callback in
  `query_command.py`.
- Load each MuSiQue raw record once, or pass validated records through the
  orchestration boundary instead of scanning JSONL files repeatedly.
- Decide whether missing raw records should fail only correctness reporting or
  fail the entire artifact stage, and encode that decision in the response
  contract.
- Preserve successful query responses when optional evaluation metadata cannot
  be loaded, while recording the artifact limitation explicitly when that is
  the chosen policy.
- Add tests for missing dataset files, missing record IDs, and successful
  answers with unavailable evaluation metadata.

**Acceptance criteria:**

- Each selected raw record is loaded at most once per command execution.
- Query execution does not repeat completed provider work because artifact
  evaluation failed.
- Artifact failures are explicit and do not silently produce incomplete claims.

### 9. Clean up command wiring and fragile retrieval bookkeeping

**Priority:** Low

- Remove duplicate retrieval imports in `query_command.py`.
- Replace `id(request)` retrieval-map keys with stable trace IDs or an explicit
  request-to-retrieval result structure.
- Ensure duplicate or repeated request objects cannot overwrite one another's
  retrieval result.
- Add direct `run_cli` tests for backend selection, prerequisite failures,
  preload behavior, query-only behavior, and exit status.
- Keep changes limited to command wiring; do not mix unrelated refactoring.

**Acceptance criteria:**

- The command module has one authoritative retrieval import block.
- Retrieval results remain associated with the correct response under repeated
  or duplicate requests.
- CLI orchestration tests cover success, insufficient evidence, preload
  failure, prerequisite failure, and provider failure.

### 10. Validate retrieval result quality and numeric safety

**Priority:** Low

- Validate retrieval scores when present, including finite numeric values and
  expected ordering semantics.
- Define behavior for malformed metadata, missing chunk IDs, missing text, and
  oversized context values or graph components.
- Ensure malformed retrieval rows become explicit retrieval failures or are
  excluded with a recorded reason rather than silently becoming evidence.
- Ensure graph-component and answer-context limits report incomplete retrieval
  explicitly; do not silently truncate a component while claiming full
  traversal.
- Add focused tests for malformed rows, non-finite scores, missing provenance,
  graph resource limits, and context-size limits.

**Acceptance criteria:**

- Every returned evidence chunk has a valid stable ID and non-empty text.
- Invalid scores or metadata cannot silently influence ranking or citation.
- Graph resource and context limits are enforced before provider invocation,
  and component incompleteness is explicit.

### 11. Persist resumable MuSiQue ingestion progress

**Priority:** High

- Add a durable local progress manifest for batch ingestion, using a simple
  JSON format unless the implementation provides a justified alternative.
- Identify a run's source unambiguously with the dataset file checksum and
  split, and track each record and its stable paragraph-level `chunk_id`
  requests. Do not treat a record ID alone as globally unique across files.
- Persist progress atomically as chunks finish. Mark a record complete only
  after every expected paragraph request has completed the configured
  `SimpleKGPipeline` stages and its Neo4j provenance has been verified.
- On resume, compare the manifest with the selected source and reconcile it
  with persisted Neo4j chunk/record provenance. Skip verified existing work;
  continue pending or partial records without silently skipping missing
  chunks. Reject or explicitly separate a manifest whose source checksum has
  changed.
- Handle a requested clean stop by stopping before starting more chunks,
  waiting for the current pipeline call to reach a safe boundary when
  possible, and flushing progress before exit. An interrupted in-flight chunk
  must remain pending until Neo4j state is checked.
- Add a configurable manifest path and document the resume behavior and local
  file lifecycle.

**Acceptance criteria:**

- A deterministic test stops a multi-record load after some chunks complete,
  starts a new process with the same source, and proves verified work is
  skipped while pending work resumes.
- A record with any missing or failed paragraph chunk is never reported as
  fully loaded.
- Clean-stop and interrupted-in-flight cases preserve enough state to resume
  without losing failures or falsely marking a record complete.
- A source checksum mismatch cannot cause old progress to be applied silently
  to different dataset contents.

### 12. Persist and drain a failed-chunk retry queue

**Priority:** High

- Add a durable retry queue for every failed application chunk request, keyed
  by its stable `chunk_id` and retaining `record_id`, source fingerprint and
  provenance, attempt count, last error category, and retry status.
- Enqueue the failure before returning or continuing the batch. Resume or an
  explicit retry operation must be able to load queued chunks and send them
  through the same configured `SimpleKGPipeline` and Neo4j verification path.
- Use bounded retry attempts and backoff for retryable failures. Keep
  non-retryable or exhausted failures in a durable dead-letter state rather
  than dropping them from the queue.
- Keep a record partial until every expected chunk has been verified in
  Neo4j and no queued or dead-lettered chunk remains unresolved. Successful
  chunks must not need to repeat provider work when a different chunk fails.
- Automatic replay depends on the idempotent write guarantee in Task 13;
  until that guarantee is verified, do not blindly replay a chunk that may
  already have written graph data.

**Acceptance criteria:**

- A deterministic test injects a chunk failure, restarts the loader, and
  proves the failed chunk is retried from the durable queue while successful
  chunks are not needlessly reprocessed.
- Retry attempts, backoff, terminal errors, and dead-letter state are visible
  in the run summary and persisted state.
- A record with one failed or unresolved chunk remains incomplete until that
  chunk succeeds and its Neo4j provenance is verified.

### 13. Make SimpleKGPipeline chunk ingestion idempotent

**Priority:** High

- Make replaying the same validated MuSiQue paragraph request, including
  retries and resume after interruption, converge on one persisted source
  chunk rather than duplicate graph data.
- Use stable `chunk_id` plus source fingerprint/provenance as the identity
  boundary. Before skipping existing data, verify that the stored text and
  provenance match; report an explicit conflict if the same identity refers
  to different input.
- Inspect and test the actual `SimpleKGPipeline` Neo4j writer behavior. Use
  compatible uniqueness/MERGE semantics or another verified guard for
  plugin-created source documents, chunks, relationships, and entity-resolution
  effects. Do not assume that deterministic IDs or a successful pipeline call
  alone make graph writes idempotent.
- Cover the failure window where Neo4j writes succeed but the local progress
  manifest is not yet updated. A retry must detect and verify that persisted
  work instead of writing duplicate source chunks or source-bound graph
  relationships.
- Keep the supported sequential ingestion contract; if concurrent writers
  are later introduced, enforce duplicate prevention at the database boundary
  rather than relying only on a local preflight check.

**Acceptance criteria:**

- Running the same chunk request twice leaves one logical source chunk and
  does not increase duplicate source-document, chunk, or source-bound
  relationship counts.
- An integration test simulates a stop after the Neo4j write and before the
  progress checkpoint, then proves resume produces no duplicates and
  preserves correct provenance.
- Reusing a `chunk_id` with changed text or source provenance fails explicitly
  instead of silently overwriting or accepting conflicting data.
- Tests cover the configured entity-resolution behavior as well as lexical
  `Document`/`Chunk` writes, so reruns cannot silently duplicate graph effects
  beyond the source-chunk boundary.

## Validation Plan

Before marking this goal complete:

- Run the full deterministic suite and record the result.
- Run focused retrieval, answer, command, prerequisite, and correctness tests.
- Run paired VectorRAG/GraphRAG evaluation and report retrieval-context quality
  separately from answer correctness and citation support.
- Run the live retrieval harness against the configured Neo4j instance when
  credentials and provider services are available.
- Inspect at least one generated artifact and verify that answer context,
  evidence, citations, provenance, and correctness metadata agree.
- Verify a GraphRAG integration fixture returns the complete seed-associated
  component with no hop-depth limit, while respecting the verified component
  boundary and explicit resource safeguards.
- Exercise ingestion interruption and resume with persisted progress, including
  a clean stop and an in-flight chunk whose Neo4j state must be reconciled.
- Exercise a failed-chunk retry from the durable queue and verify its parent
  record stays incomplete until all expected chunks are persisted.
- Run the same chunk through the configured `SimpleKGPipeline` repeatedly,
  including a write-before-checkpoint interruption, and verify graph counts,
  entity-resolution effects, text, and provenance remain idempotent.
- Update `documentation.md` with the final backend limitations, index
  requirements, harness command, and known skipped integration conditions.

## Future Work

- Integrate RAGAS after the paired retrieval/answer artifacts preserve the
  exact contexts supplied to the model, references, and run configuration.
  Use it for semantic context and answer evaluation; keep deterministic Neo4j
  membership, record/component scope, citation-ID, and artifact-integrity
  checks as separate safeguards.
- Extend verification from chunk membership to GraphRAG component provenance:
  confirm returned nodes and relationships belong to the expected component
  and can be traced to source chunks in the selected MuSiQue record.
- MuSiQue supporting-paragraph labels can measure source-evidence coverage but
  are not gold entity/relationship annotations. Add a graph-level reference
  dataset or a separately validated method before claiming extracted graph
  relationships themselves are correct.
- Once these inputs and checks are available, report RAGAS scores alongside
  deterministic evidence coverage, graph provenance, answer correctness, and
  citation support rather than treating any one metric as a complete verdict.

## Deferred Scope

The following remain outside this hardening goal unless separately approved:

- Hybrid search.
- Traversal across unrelated components or the entire database without a
  verified component boundary, relationship policy, and resource safeguards.
- LangGraph reflection cycles.
- Full-dataset benchmarking.
- Autonomous entity resolution.
- Production parallelism and deployment infrastructure.
