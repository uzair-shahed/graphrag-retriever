# Retrieval Pipeline Hardening Goals

**Date:** 17 October 2026

**Last modified:** 17 October 2026

**Purpose of this file:**

Record the remaining retrieval-pipeline hardening tasks identified during the
repository-wide audit on 17 October 2026. The current pipeline is implemented
and its deterministic test suite passes, but these tasks are required before
claiming that GraphRAG answers, evidence, citations, performance, and live
artifacts are production-reliable.

**Files that use this file:**

The active retrieval implementation is in
`src/graphrag_pipeline/retrieval.py`, `src/graphrag_pipeline/answer.py`, and
`src/graphrag_pipeline/query_command.py`. Retrieval contracts and correctness
artifacts are defined in `src/graphrag_pipeline/query.py` and
`src/graphrag_pipeline/musique_verification.py`. The broader project roadmap
is maintained in `agent-coordination/tasks.md` and the earlier vertical-slice
goal is maintained in `agent-coordination/tasks-goals-01-oct-2026.md`.

**Caveats:**

The audit found no current static diagnostics and the deterministic suite
reported 118 passing tests. The live Neo4j preload and query path has also
been exercised, but no dedicated repeatable end-to-end retrieval harness
currently captures that run. Existing worktree changes are user or
automation-owned and must be preserved while completing these tasks.

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
