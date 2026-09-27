# Capstone Implementation Tasks

## Project

**Self-Correcting GraphRAG Knowledge Pipeline with Autonomous Entity Resolution**  
**Program:** RIT M.S. in Computer Science Capstone  
**Term:** Fall 2026  
**Author:** Uzair Shahed Islam

This roadmap converts the weekly milestones in `project-idea-files/milestones.md` into executable tasks, acceptance criteria, evidence requirements, and advisor checkpoints.

`project-idea-files/milestones.md` remains authoritative for schedule, scope, deliverables, and advisor checkpoints. The current implementation will be developed locally with the MuSiQue dataset, local storage, a local Neo4j instance, and a linear processing flow. Docker packaging and deployment are deferred to a stretch goal. `project-idea-files/project-proposal-1.md` supplies architectural detail and motivation; where its phase wording differs, the milestone document controls.

The current architecture uses Neo4j's AI plugin for extraction and will use its GraphRAG and VectorRAG capabilities for retrieval. Custom hand-built extraction, retrieval, vector indexing, and plugin-replacement wrappers are archived rather than active core work. The current verified vertical slice is the local MuSiQue paragraph-to-`SimpleKGPipeline`-to-Neo4j graph path with read-only provenance verification.

## Current Implementation Status

Completed and verified:

- Local Neo4j Enterprise connectivity, bounded transactions, and connection smoke tests.
- Focused MuSiQue fixtures with validated chunk identity and source provenance.
- OpenRouter-backed `SimpleKGPipeline` configuration using `neo4j-graphrag==1.21.0`.
- Plugin input mapping that preserves paragraph text, carries provenance, and excludes gold labels.
- A live focused fixture run that persisted plugin-created nodes and relationships in Neo4j.
- Read-only graph and provenance verification for `Document`, `Chunk`, source links, and extracted relationships.
- Active data-contract and capability documentation, with the superseded task plan preserved in `agent-coordination/archive/tasks_archive.md`.

Still active:

- Idempotent graph reset and rerun checks.
- Neo4j AI plugin VectorRAG and GraphRAG retrieval integration.
- LangGraph orchestration around extraction, retrieval, answer generation, and reflection.
- Evaluation artifacts, benchmark methodology, and broader end-to-end tests.

## Neo4j AI Plugin Architecture Decision

- [x] Use Neo4j's AI plugin as the default implementation for both GraphRAG and VectorRAG in the local project runtime.
- [x] The retrieval layer is delegated to Neo4j AI plugin capabilities.
- [x] Document Neo4j plugin version, configuration, feature flags, limitations, and validation checks before relying on plugin responses in evaluation.
- [x] Treat the plugin-first `SimpleKGPipeline` as the active extraction engine for this project scope, with the archived hand-built extraction models retained only as compatibility/fallback references.

## Definition Of Done

The capstone is complete when:

- [x] The focused extraction vertical slice runs locally from a documented local development workflow.
- [ ] Docker Compose packaging and deployment are documented and implemented only as a stretch goal after the core pipeline is complete.
- [ ] The architecture documents LangGraph, Neo4j, local storage, and a linear processing flow; Ray, Polars, Redis, PostgreSQL, and DuckDB are documented only as deferred stretch options.
- [ ] MuSiQue official JSONL data can be validated, lightly adapted without changing source text, extracted, resolved, and stored in Neo4j with reproducible scripts; the focused fixture extraction and provenance path are complete, while full ingestion and entity resolution remain open.
- [ ] LangGraph orchestrates extraction, entity resolution, retrieval, answer generation, and reflection/self-correction.
- [ ] Users can select Neo4j AI plugin VectorRAG or Neo4j AI plugin GraphRAG retrieval modes, with multi-hop Cypher verification and evidence tracing retained as a required validation path; the interim architecture diagram may show only the GraphRAG path, while Hybrid Search remains an advisor-approved future extension.
- [ ] Core evaluation artifacts are frozen, version-controlled, and reproducible.
- [ ] Entity-resolution, retrieval-quality, latency, throughput, and cost results are reported with methodology and limitations.
- [ ] Core benchmark targets are measured and compared against baselines; any missed targets are explained rather than hidden.
- [ ] Tests cover schema initialization, parsing, validation, graph writes, retrieval, reflection, and end-to-end execution.
- [ ] The final report, public repository documentation, defense deck, poster, demo script, and advisor sign-off are complete.

## Success Criteria And Evidence

| Area | Target | Required evidence |
|---|---:|---|
| Entity resolution precision | >90% | Frozen labeled dataset, evaluation script, precision report and error analysis |
| Entity resolution recall | >85% | Frozen labeled dataset, evaluation script, recall report and error analysis |
| Multi-hop retrieval Recall@K | >80% | 30-50 query suite, gold paths/facts, A/B results against approximately 35% baseline |
| Faithfulness | >90% | RAGAS configuration, raw results, aggregate report, sampled grounded answers |
| Answer relevance | >85% | RAGAS configuration, raw results, aggregate report, sampled answers |
| Query latency P50 | <3 seconds | Repeated-run benchmark data and latency distribution |
| Query latency P95 | <5 seconds | Repeated-run benchmark data and latency distribution |
| API-token cost reduction | approximately 75% | Cost assumptions, token logs, baseline comparison, budget report |
| Ingestion throughput | Measured in docs/sec | Reproducible load benchmark with environment details |

## Project Controls And Prerequisites

### Scope and planning

- [ ] Reconcile this roadmap with `project-idea-files/project-proposal-1.pdf` and record proposal changes in an issue or decision log.
- [ ] Confirm research question, hypotheses, baseline definition, and success criteria with the advisor.
- [ ] Confirm what counts as the Vector RAG baseline and what counts as the graph-enhanced system.
- [ ] Define the minimum viable core scope and the conditions that permit stretch-goal work.
- [ ] Create a weekly review cadence and record advisor decisions, risks, blockers, and metric changes.
- [ ] Create a milestone evidence directory or artifact convention for logs, reports, charts, and reproducibility metadata.

### Repository and runtime

- [ ] Choose and document the Python version, package manager, lockfile strategy, and supported operating systems.
- [ ] Define a minimal repository layout for application code, tests, data manifests, scripts, configuration, evaluation artifacts, and documentation.
- [ ] Add a minimal environment-variable template for Neo4j, LLM provider, and embedding provider settings.
- [ ] Define local MuSiQue storage, including stable file paths, checksums, retention, source-split handling, access policy, and provenance metadata.
- [ ] Record Redis and PostgreSQL as deferred stretch services; do not provision or configure them for the current core implementation.
- [ ] Record Polars and Ray as deferred stretch technologies; use a linear flow unless a milestone-approved stretch implementation is started.
- [ ] Keep secrets out of source control and document local secret provisioning.
- [ ] Establish only the essential formatting and test commands needed during initial development; defer broader CI and pre-commit setup until hardening.
- [ ] Define experiment IDs, random seeds, run metadata, model versions, prompt versions, and artifact naming.
- [ ] Document MuSiQue provenance, licensing, local download location, checksum, and permitted redistribution.
- [ ] Record the project motivation and reporting lens: multi-hop retrieval gaps, entity duplication/graph fragmentation, and lack of deterministic verification.
- [ ] Create a decision log for model choices, thresholds, prompts, schema changes, and benchmark exclusions.

### Cross-cutting engineering quality

- [ ] Add structured logs with run ID, MuSiQue record/query ID, agent, state transition, model, latency, token usage, and error category.
- [ ] Define retry limits, timeouts, backoff, idempotency, dead-letter handling, and partial-failure recovery.
- [ ] Validate all model-generated structured output before it reaches graph writes or answer delivery.
- [ ] Add unit tests for pure transformations and contract tests for Neo4j, LLM, embedding, and RAGAS integrations; use deterministic mocks before paid model calls.
- [ ] Add integration tests for the full graph-write and retrieval paths using small deterministic fixtures.
- [ ] Add resource and load tests for memory, CPU, Neo4j, and the linear processing flow.
- [ ] Record limitations, unavailable services, skipped cases, and known failure modes in the final report.

## Phase 1: Infrastructure And Modern Ingestion

### Weeks 1-3: Preparation and environment design

- [x] Finalize architecture diagram showing ingestion, chunking, extraction, resolution, graph storage, retrieval, reflection, evaluation, and user entry point.
- [x] Specify service boundaries and data contracts between ingestion, agents, storage, and evaluation.
- [ ] Finalize the current technology boundary: local Python execution, local MuSiQue storage, a locally running Neo4j instance for graph/vector storage, and a linear ingestion and processing flow.
- [ ] Define local MuSiQue file paths, checksums, retention behavior, access policy, source-split handling, and provenance links from dataset record to paragraph/chunk, graph entity, and benchmark record.
- [ ] Record the deferred decision for Ray workers; the current implementation passes chunks through a linear flow without worker pools.
- [ ] Define the initial Neo4j labels, relationships, properties, provenance fields, and identifiers needed by ingestion and evaluation.
- [ ] Record basic local resource requirements for the linear flow; defer detailed capacity planning until benchmarking.
- [x] Define the OpenRouter model, token, retry, and concurrency assumptions used by the budget in [openrouter-cost-estimate.md](openrouter/openrouter-cost-estimate.md); replace planning assumptions with a measured usage ledger before the full paid run.
- [x] Create a project-specific minimal fixture based on the official MuSiQue schema, including multi-hop evidence, distractors, alternate mentions, malformed input, missing optional fields, and insufficient/unanswerable cases where needed.
- [x] Define the paragraph-as-chunk contract: initially preserve each MuSiQue paragraph as one chunk with stable IDs, supporting labels, and source provenance; add sub-chunking only if later required by context limits or retrieval quality.

**Acceptance criteria**

- [ ] Architecture and data-flow diagrams are reviewed by the advisor.
- [ ] Service contracts and configuration names are documented.
- [x] A small fixture can be used without downloading the full dataset or calling a paid model for local validation and pipeline initialization.

### Week 4: Local environment and storage layer

- [x] Install and run Neo4j locally with the required APOC and vector-search support.
- [x] Verify local Neo4j startup, persistence, restart behavior, and clean shutdown.
- [x] Implement the simplest working Neo4j connection with bounded transaction timeouts and retry behavior.
- [ ] Create Cypher initialization scripts for constraints, indexes, full-text search, and vector indexes.
- [x] Write a smoke test that creates, reads, updates, and cleans up a fixture graph.
- [ ] Implement one MuSiQue download/verification and local dataset connection path for the official JSONL files. -- (i'm not sure if this is needed since we already have the downloaded dataset)
- [x] Validate and lightly adapt the focused MuSiQue fixture while preserving original IDs, paragraph text, titles, indices, and provenance; full official-split adaptation remains open.
- [ ] Distinguish labeled train/dev records from test records with withheld labels; never evaluate test records as labeled data.
- [x] Expose the focused paragraph as one deterministic application chunk with stable record, document, paragraph, and chunk IDs; defer configurable sub-chunking unless needed.
- [ ] Handle malformed dataset records, missing metadata, duplicate records, and unsupported records explicitly.
- [ ] Create an OpenRouter cost budget using expected input/output tokens, model pricing, retries, concurrency, and dataset volume; keep the initial fixture path free of paid calls.
- [ ] Implement a minimal LangGraph loop with one validated LLM call and observable state transitions.
- [x] Verify that the focused MuSiQue extraction pipeline runs locally without Docker and document the prerequisites needed to reproduce it.

**Deliverables**

- [ ] Documented local development environment.
- [ ] Verified Neo4j connection scripts.
- [ ] MuSiQue ingestion and normalization scripts.
- [x] OpenRouter cost estimate with measured MuSiQue volume, model pricing snapshot, planning scenarios, and a documented measurement plan.
- [ ] LangGraph/LLM smoke test.

**Acceptance criteria**

- [x] The current checkout can install prerequisites, start local Neo4j, and run the focused fixture workflow using documented commands.
- [ ] Initialization is idempotent and creates the expected constraints and indexes.
- [x] The project fixture can be loaded and queried without requiring the original AllenNLP training environment; sampled full-record ingestion remains open.
- [ ] Connection and resource settings are recorded for the advisor.

**Advisor checkpoint**

- [ ] Review database constraints, connection pooling, resource allocations, dataset provenance, and the initial cost budget.

**Optional stretch work**

- [ ] Package the working local application and Neo4j setup with Docker Compose after the core implementation and evaluation are complete.
- [ ] Add Polars or Ray processing only if required by an approved stretch goal.
- [ ] Add DuckDB SQL pipelines for metadata extraction, cleaning, and structural normalization only if required by an approved stretch goal.
- [ ] Add Redis and PostgreSQL services only if the corresponding Week 11 telemetry stretch goal is approved.

## Phase 2: LangGraph Multi-Agent Engine And Parallelization

### Week 5: Plugin extraction and autonomous entity resolution

- [x] Record the verified plugin-created graph vocabulary, source references, and provenance fields for the focused fixture.
- [ ] Implement Cypher drivers for graph initialization, index creation, vector configuration, and fixture reset; plugin-managed extraction writes remain the active graph-write path.
- [x] Define and validate the plugin input boundary for one MuSiQue paragraph, including text preservation and exclusion of gold labels.
- [x] Configure and run `SimpleKGPipeline` through the linear local flow, with plugin failures raised rather than reported as success.
- [ ] Build the LangGraph orchestration around plugin extraction only after retrieval and answer-generation boundaries are defined.
- [ ] Implement extraction through the linear flow; add Ray-parallelized extraction only if an approved stretch goal requires it.
- [ ] Build embedding generation and candidate entity retrieval with configurable similarity thresholds and top-K values.
- [ ] Build the entity-resolution agent that combines vector candidates with contextual LLM reasoning.
- [ ] Define match, non-match, uncertain, and abstain outcomes.
- [ ] Implement atomic Cypher `MERGE` transactions for canonical entities and synonymous mentions.
- [ ] Implement conflict-resolution rules for properties, provenance, confidence, timestamps, and incompatible values.
- [ ] Make plugin fixture reset and rerun behavior idempotent and record the result for later audit.
- [x] Create focused tests for plugin input, configuration, schema boundaries, and graph/provenance verification.

**Deliverables**

- [ ] Functional Neo4j schema and initialization driver.
- [x] Focused plugin-managed extraction vertical slice with graph/provenance verification.
- [x] Linear plugin extraction path with reproducible processing inputs and recorded live-run evidence.
- [ ] Autonomous entity-resolution agent with inline deduplication.

**Acceptance criteria**

- [x] Plugin/provider failures are raised rather than silently reported as successful extraction; deeper output-quality validation remains a future retrieval/evaluation concern.
- [ ] Reprocessing the same chunk does not duplicate graph entities or relationships.
- [ ] Entity-resolution decisions include candidates, rationale/result, threshold/configuration, and provenance.
- [ ] Sample metrics and representative errors are recorded.

**Advisor checkpoint**

- [ ] Review schema enforcement, sample extraction accuracy, resolution behavior, and graph transaction safety.

### Week 6: Configurable retrieval pipeline and entry point

- [ ] Inspect graph subgraphs after ingestion and resolution for duplicate entities, orphan nodes, broken provenance, and unexpected fragmentation.
- [ ] Define the user query request and response contracts, including query text, retrieval mode, hop depth, top-K, citations, evidence, and trace ID.
- [ ] Implement the query entry point with validation, timeouts, error responses, and configuration flags.
- [ ] Configure Neo4j AI plugin for VectorRAG and GraphRAG retrieval so the project uses plugin-managed retrieval and candidate ranking instead of hand-built retrieval code.
- [ ] Validate plugin query output, citations, provenance, and evidence path metadata against the project contract requirements.
- [ ] Implement multi-hop Cypher verification/tracing with bounded depth, relationship filters, candidate ranking, and evidence paths for grounded-answer validation and audit.
- [ ] Define Hybrid Search as a future vector-seed plus graph-expansion extension, retaining Neo4j AI plugin VectorRAG and GraphRAG as the required modes.
- [ ] Implement Hybrid Search only if the advisor approves the added scope; ensure it cannot displace either required mode.
- [ ] Ensure retrieval results preserve source document, chunk, entity, and relationship provenance.
- [ ] Add common answer-generation interface so both modes can be evaluated with the same output contract.
- [ ] Add tests for mode selection, hop limits, ranking, empty results, conflicting evidence, and plugin-response validation.

**Deliverable**

- [ ] Retrieval module that executes Vector RAG or multi-hop Cypher based on incoming configuration.

**Acceptance criteria**

- [ ] The same query can be executed in both modes and produces comparable structured outputs.
- [ ] Cypher traversal depth is bounded and observable.
- [ ] Candidate ranking and citations are deterministic under fixed configuration.

**Advisor checkpoint**

- [ ] Review traversal depth, retrieval ranking, graph connectivity, and the comparison protocol for both modes.

### Week 7: Reflection and self-correction

- [ ] Define LangGraph state for question, retrieval mode, candidate evidence, draft answer, citations, grounding assessment, retry count, and final status.
- [ ] Implement the reflection agent using cyclic state transitions with explicit termination conditions.
- [ ] Cross-check proposed Neo4j paths against MuSiQue supporting facts and expected paths.
- [ ] Use explicit Neo4j path queries to deterministically verify every claim and retain an auditable step-by-step evidence chain.
- [ ] Implement grounding checks for unsupported claims, missing citations, contradictory evidence, and insufficient paths.
- [ ] Route ungrounded or hallucinated drafts back through retrieval and answer-generation states.
- [ ] Suppress unsupported claims and return a transparent insufficient-evidence result when deterministic path verification fails.
- [ ] Enforce a retry budget and return a transparent failure/insufficient-evidence result when the budget is exhausted.
- [ ] Capture an end-to-end execution trace for every state transition.
- [ ] Build deterministic fixtures for grounded answers, hallucinations, missing evidence, contradictory evidence, and retry exhaustion.
- [ ] Run an end-to-end walkthrough from user query through retrieval, generation, reflection, correction, and delivery.

**Deliverable**

- [ ] End-to-end self-correcting multi-agent GraphRAG pipeline in LangGraph.

**Acceptance criteria**

- [ ] A grounded answer completes without unnecessary retries.
- [ ] An ungrounded answer is rejected or corrected before delivery.
- [ ] The graph path and source evidence used for the final answer are inspectable.
- [ ] Every delivered claim has an inspectable Neo4j path or source citation; unsupported claims are not delivered as facts.
- [ ] Cycles terminate within the configured retry budget.

**Advisor checkpoint**

- [ ] Complete the mid-semester review and demonstrate the complete multi-agent execution cycle.

## Phase 3: Benchmark Construction And Quantitative Evaluation

### Week 8: Evaluation datasets and annotation

- [ ] Write annotation guidelines defining entity equivalence, non-equivalence, ambiguity, canonical form, and adjudication rules.
- [ ] Curate 150 entity pairs representative of spelling variation, aliases, abbreviations, context-dependent matches, and hard non-matches.
- [ ] Human-annotate the entity pairs and record annotator rationale, confidence, and adjudication decisions.
- [ ] Freeze the entity-resolution dataset with version, checksum, schema, and provenance metadata.
- [ ] Design 30-50 multi-hop cross-document reasoning queries.
- [ ] Record gold answers, required facts, expected graph paths, hop counts, source documents, and acceptable answer variants.
- [ ] Audit that the query suite requires relational reasoning and is not answerable from one isolated passage.
- [ ] Freeze query datasets as version-controlled JSON/CSV artifacts.
- [ ] Create loaders and schema validators that reject modified or malformed evaluation artifacts.
- [ ] Define train/tuning/evaluation separation to prevent threshold or prompt tuning leakage.

**Deliverables**

- [ ] Annotated 150-pair entity-resolution dataset.
- [ ] 30-50 query multi-hop test suite with facts and graph paths.
- [ ] Annotation guidelines, dataset manifests, checksums, and quality-control record.

**Acceptance criteria**

- [ ] Every evaluation record has a label, rationale or gold evidence, provenance, and schema-valid structure.
- [ ] Ambiguous cases have adjudicated labels.
- [ ] Evaluation artifacts are reproducible from their manifest and cannot be silently changed.

**Advisor checkpoint**

- [ ] Audit annotation quality, dataset leakage controls, and multi-hop query complexity.

### Week 9: RAGAS harness and entity benchmark

- [ ] Integrate RAGAS into automated evaluation scripts with pinned versions and documented metric configuration.
- [ ] Implement entity-resolution evaluation for true positives, false positives, true negatives, false negatives, precision, recall, and threshold curves.
- [ ] Run the agent against the frozen 150-pair dataset and preserve raw predictions and configuration.
- [ ] Report precision and recall with sample counts and error categories.
- [ ] Tune prompts, thresholds, candidate top-K, or abstention rules only on the designated tuning split.
- [ ] Rerun the untouched evaluation split after tuning.
- [ ] Document any failure to reach >90% precision or >85% recall and analyze representative errors.
- [ ] Add regression tests for previously resolved false merges and false non-matches.

**Deliverable**

- [ ] Automated entity-resolution evaluation report with precision/recall curves and error analysis.

**Acceptance criteria**

- [ ] Results can be regenerated from a single documented command and frozen inputs.
- [ ] Metrics are calculated from saved predictions rather than manually copied values.
- [ ] Target comparison and confidence/variation information are included.

**Advisor checkpoint**

- [ ] Verify the entity-resolution evaluation protocol and results against the target metrics.

### Week 10: Retrieval A/B testing and efficiency profiling

- [ ] Define a fixed A/B protocol with identical questions, answer model, generation settings, context budget, and evaluation seeds where possible.
- [ ] Execute Neo4j AI plugin VectorRAG and GraphRAG retrieval on the MuSiQue query suite; keep Cypher verification as the evidence-tracing path used to audit grounded responses.
- [ ] If approved, execute Hybrid Search using the same fixed query suite and protocol; report it as an extension rather than changing the required VectorRAG/GraphRAG comparison.
- [ ] Preserve per-query outputs, citations, paths, latency, token usage, errors, and configuration.
- [ ] Preserve raw-document object references and provenance through ingestion, graph construction, retrieval, reflection, and evaluation artifacts.
- [ ] Calculate Multi-Hop Recall@K and compare against the approximately 35% baseline and >80% target.
- [ ] Calculate Faithfulness and Answer Relevance through the configured RAGAS harness.
- [ ] Measure ingestion throughput in docs/sec across documented dataset sizes and worker configurations.
- [ ] Measure repeated query latency and calculate P50 and P95; investigate outliers and cold-start effects.
- [ ] Estimate API token cost for baseline and graph-enhanced configurations and calculate the reduction.
- [ ] Produce latency distributions, metric tables, confidence/variation summaries, and representative qualitative examples.
- [ ] Write a limitations section covering model variance, dataset size, hardware, API pricing, and evaluation bias.
- [ ] Hold the core-results review before starting stretch goals.

**Deliverable**

- [ ] Comprehensive quantitative benchmark report with performance figures and latency charts.

**Acceptance criteria**

- [ ] A reviewer can reproduce both retrieval modes and inspect raw per-query evidence.
- [ ] Results report >80% Recall@K, >90% Faithfulness, >85% Answer Relevance, P50 <3 seconds, P95 <5 seconds, and approximately 75% cost reduction where achieved.
- [ ] Missed targets have an explicit explanation and remediation or limitation plan.

**Advisor checkpoint**

- [ ] Review core evaluation results and approve or reject transition to stretch goals.

## Phase 4: Stretch Goals, Packaging, And Defense

Stretch work begins only after core functionality, evaluation, and advisor approval are complete. Docker packaging is part of this stretch phase, not a prerequisite for core implementation. If schedule or metrics are at risk, preserve core reproducibility and documentation instead.

### Week 11: Tier 1 stretch goals

- [ ] Define query classes and a labeled routing set for Vector, Cypher, Hybrid, and fallback cases.
- [ ] Implement the retrieval-routing agent at the query entry point.
- [ ] Add a deterministic fallback and allow an explicit user override.
- [ ] Evaluate routing classification accuracy, misroutes, latency overhead, and impact on retrieval metrics.
- [ ] Define an audit event schema for prompts, state transitions, agent decisions, retrieval settings, evidence, errors, and token usage.
- [ ] Defer short-lived Redis traces, durable PostgreSQL telemetry, and Polars-based persistence until the Week 11 stretch goal is approved.
- [ ] If approved, add trace correlation across Redis, PostgreSQL, Neo4j, and benchmark artifacts.
- [ ] Add access controls and redaction rules for prompts, secrets, and sensitive data.

**Deliverables**

- [ ] Intelligent query-routing layer.
- [ ] Persistent audit telemetry logging.

**Acceptance criteria**

- [ ] Routing decisions are inspectable and can be overridden.
- [ ] Telemetry survives the intended service restart and can be queried by run ID.
- [ ] Logging does not expose secrets and does not materially break latency targets.

**Advisor checkpoint**

- [ ] Assess routing accuracy and confirm PostgreSQL telemetry schema integration.

### Week 12: Tier 2 stretch goals and hardening

- [ ] Implement an asynchronous graph-validation agent for orphan subgraphs, duplicate identifiers, missing provenance, invalid labels, invalid relationships, and schema violations.
- [ ] Produce validation findings with severity, node/edge IDs, evidence, and remediation status.
- [ ] Enforce typing across application and evaluation code.
- [ ] Refactor into modular packages with clear dependency boundaries.
- [ ] Complete unit, integration, contract, and end-to-end tests for the supported workflow.
- [ ] Add CI checks for formatting, linting, type checking, tests, security checks, and artifact schema validation.
- [ ] Add migration/version handling for graph schema and evaluation artifacts.
- [ ] Make the complete pipeline runnable with `docker-compose up` or the documented single-command equivalent as an approved packaging stretch goal.
- [ ] Test clean Docker deployment from a fresh environment and verify persistent volumes and initialization if that stretch goal is completed.
- [ ] Re-run core metrics after hardening to detect regressions.

**Deliverables**

- [ ] Graph validation agent.
- [ ] Hardened typed and modular codebase.
- [ ] Complete test suite and CI checks.
- [ ] Single-command local development workflow; Docker deployment package only if the stretch goal is completed.

**Acceptance criteria**

- [ ] Validation findings are actionable and do not silently mutate production data.
- [ ] A clean local setup succeeds from documented prerequisites.
- [ ] Core benchmark results remain comparable after refactoring; Docker comparability is required only if the packaging stretch goal is completed.

**Advisor checkpoint**

- [ ] Perform repository audit, local setup check, and final metric verification; review Docker deployment only if the packaging stretch goal is completed.

### Week 13: Documentation, portfolio, and defense

- [ ] Finalize the capstone report: problem, research question, related work, architecture, methodology, implementation, datasets, experiments, results, limitations, ethics, and future work.
- [ ] Include enough configuration and protocol detail for independent reproduction.
- [ ] Prepare a public GitHub repository with a concise README, prerequisites, setup, configuration, data provenance, usage examples, architecture diagrams, tests, and benchmark reproduction scripts.
- [ ] Separate public code and metadata from secrets, restricted data, and local-only artifacts.
- [ ] Add a release tag or archive containing the evaluated code and configuration.
- [ ] Build the defense presentation with motivation, baseline, system design, key implementation decisions, quantitative results, failure cases, and contributions.
- [ ] Build the required poster and ensure figures match the final report.
- [ ] Prepare a live or recorded demo using a deterministic fixture and a documented fallback recording.
- [ ] Prepare answers for limitations, cost, latency, false merges, hallucinations, data provenance, and alternative designs.
- [ ] Conduct a final advisor rehearsal and incorporate feedback.
- [ ] Obtain final advisor sign-off and archive the final evidence bundle.

**Deliverables**

- [ ] Final MS Capstone report.
- [ ] Polished public repository and reproducibility package.
- [ ] Defense slide deck.
- [ ] Poster.
- [ ] Demo script/recording and final evidence archive.

**Acceptance criteria**

- [ ] Report, repository, figures, deck, poster, and demo use consistent final metrics.
- [ ] The report explains the four evaluation layers: generation reliability, entity-resolution quality, retrieval performance, and operational efficiency.
- [ ] A clean evaluator can follow the setup and reproduce at least the core smoke test and benchmark commands.
- [ ] All known limitations and missed targets are disclosed.

**Advisor checkpoint**

- [ ] Complete final presentation evaluation and obtain sign-off.

### Week 14: Contingency and submission buffer

- [ ] Reserve time for advisor-requested corrections, failed reruns, infrastructure outages, and formatting requirements.
- [ ] Re-run the final smoke test and verify the release archive.
- [ ] Confirm all required submission portals, filenames, deadlines, and file formats.
- [ ] Submit the report, deck, poster, repository link, and any required supplementary artifacts.
- [ ] Archive final logs, benchmark outputs, environment versions, and decision records.

## Traceability Matrix

| Milestone requirement | Implementation location in this roadmap | Evidence |
|---|---|---|
| MuSiQue ingestion | Week 4 ingestion tasks | Dataset scripts, manifest, focused fixture and provenance tests |
| OpenRouter cost budget | Week 4 cost tasks | Budget spreadsheet/report and token assumptions |
| Neo4j, APOC, vector search | Week 4 local environment tasks | Local Neo4j setup, initialization smoke test |
| LangGraph agent loop | Week 4 and Week 7 | Smoke-test trace and end-to-end trace |
| Neo4j schema and Cypher drivers | Week 5 | Schema document, migration/init scripts, tests |
| Plugin-managed extraction | Week 5 | `SimpleKGPipeline`, plugin input adapter, live fixture run, graph verification |
| Linear plugin extraction flow | Weeks 4-5 | Fixture runner, configuration tests, live run summary, provenance verification |
| Autonomous entity resolution | Week 5 and Week 9 | Resolution audit records, precision/recall report |
| Configurable plugin VectorRAG/GraphRAG retrieval | Week 6 and Week 10 | Query API contract, plugin outputs, Cypher evidence checks, ranking tests |
| Reflection and self-correction | Week 7 | State graph, grounding tests, retry traces |
| 150-pair entity dataset | Week 8 | Frozen labeled JSON/CSV, guidelines, manifest |
| 30-50 multi-hop queries | Week 8 | Frozen query set, gold facts, graph paths |
| RAGAS evaluation | Weeks 9-10 | Pinned harness, raw outputs, metric reports |
| Latency, throughput, and cost profiling | Week 10 | Benchmark logs, distributions, cost comparison |
| Retrieval routing and telemetry | Week 11 stretch | Routing labels/results, deferred Redis/PostgreSQL trace queries |
| Graph validation and hardening | Week 12 stretch | Validation findings, test reports, deployment logs |
| Report, repository, deck, and poster | Week 13 | Final release and submission archive |

## Proposal Alignment Matrix

The proposal adds architectural rationale and implementation detail; it does not override the milestone schedule or milestone classification of core versus stretch work.

| Proposal element | Roadmap treatment | Evidence |
|---|---|---|
| Multi-hop retrieval gap | Core motivation and required Vector-versus-Cypher evaluation | Week 6 retrieval tests; Week 10 A/B report |
| Entity duplication and graph fragmentation | Core entity-resolution, idempotent merge, conflict, and graph-integrity work | Week 5 resolution audit; graph integrity report |
| Deterministic verification | Core Reflection Agent path checks and unsupported-claim suppression | Week 7 state traces and grounding tests |
| Linear MuSiQue processing | Current core flow; Polars and Ray are deferred unless stretch work is approved | Ingestion scripts and deterministic processing logs |
| Neo4j graph and vector-index host | Core storage and retrieval implementation | Schema, indexes, Cypher tests, deployment logs |
| Redis session state | Deferred Week 11 stretch service; no current setup | Advisor decision log and future telemetry tests |
| PostgreSQL audit persistence | Deferred Week 11 stretch service; no current setup | Advisor decision log and future telemetry tests |
| Local MuSiQue storage | Current core storage with provenance and retention | Local manifest, checksums, and ingestion tests |
| Hybrid Search | Advisor-approved extension; must not replace required Vector RAG or multi-hop Cypher modes | Week 6 contract and optional Week 10 comparison |
| Four evaluation layers | Report generation reliability, entity resolution, retrieval performance, and operational efficiency separately | Week 9-10 benchmark report |
| Open-source repository | Week 13 package must expose LangGraph orchestration and Neo4j tools; Docker and Ray task interfaces are deferred unless stretch work is approved | README, setup guide, release archive |

## Risk Register

| Risk | Early signal | Mitigation | Owner/evidence |
|---|---|---|---|
| LLM/API cost or availability | Budget burn, rate limits, outages | Use fixture data, cache calls, cap retries, record token usage, maintain a local/mock path | Cost report and failure log |
| Malformed model output | Schema failures or invalid triples | Strict schemas, corrective retries, dead-letter queue, contract tests | Validation metrics |
| False entity merges | Unrelated entities merged | Conservative thresholds, abstention, provenance, rollback-safe transactions, hard-negative tests | Resolution error analysis |
| Graph fragmentation | Orphans, duplicate IDs, disconnected expected paths | Constraints, idempotent writes, graph audits, validation agent | Graph integrity report |
| Evaluation leakage or weak labels | Tuning affects test results, ambiguous annotations | Frozen splits, adjudication, manifests, untouched evaluation set | Dataset QC record |
| Vector index or Neo4j limitations | Missing index support, slow queries, inconsistent results | Verify versions early, document limits, provide bounded fallback, benchmark actual deployment | Compatibility notes |
| Resource exhaustion | Worker crashes, memory pressure, slow P95 | Resource limits, bounded concurrency, profiling, smaller fixtures, backpressure | Load-test report |
| Reflection loops | Retry budget exhausted frequently | Explicit termination, grounding thresholds, trace analysis, fallback response | State-transition metrics |
| Schedule compression | Core metrics or report slip | Gate stretch goals, weekly checkpoints, preserve reproducibility and documentation | Decision log |

## Final Evidence Checklist

- [ ] Source and environment versions are recorded.
- [ ] Local environment startup and clean-run logs are archived; Docker logs are archived only if the packaging stretch goal is completed.
- [ ] Dataset manifests, checksums, annotation guidelines, and frozen artifacts are archived.
- [ ] Schema, prompt, threshold, and model configurations are versioned.
- [ ] Raw predictions and per-query benchmark outputs are preserved.
- [ ] Precision, recall, Recall@K, Faithfulness, Answer Relevance, throughput, latency, and cost calculations are reproducible.
- [ ] Representative successes and failures are included.
- [ ] Tests and CI results are archived.
- [ ] Final report, README, architecture diagrams, deck, poster, and demo are mutually consistent.
- [ ] Advisor feedback and final sign-off are recorded.

## Deferred Tasks (Stretch Goals)

These items were intentionally moved from the core roadmap because they are either custom implementations that Neo4j AI plugin replaces or advanced capabilities that should only be attempted after the core ingestion, extraction, resolution, plugin-based retrieval, and grounded-answer path are working.

### Active plugin retrieval and future extensions

- [ ] Configure and validate Neo4j AI plugin VectorRAG retrieval.
- [ ] Configure and validate Neo4j AI plugin GraphRAG retrieval.
- [ ] Preserve bounded multi-hop Cypher verification and source-evidence tracing for grounded-answer validation.
- [ ] Evaluate plugin VectorRAG and GraphRAG using the same frozen query suite and answer contract.
- [ ] Add retrieval routing only after both required plugin modes have been validated.
- [ ] Implement Hybrid Search only as an advisor-approved extension that does not replace either required plugin mode.

Custom extraction and retrieval replacements are preserved in
`agent-coordination/archive/tasks_archive.md` and are not active core tasks.

### Deferred platform and scaling work

- [ ] Add Ray-based parallelization for ingestion, extraction, or evaluation if a milestone-approved stretch implementation requires it.
- [ ] Add Polars or DuckDB pipelines for heavy metadata normalization, cleaning, and benchmark data preparation if approved as stretch work.
- [ ] Add Redis-backed session or trace state for ephemeral agent telemetry if the project later expands to a multi-user or multi-run environment.
- [ ] Add PostgreSQL-backed durable telemetry, audit logging, and historical run tracking if the telemetry stretch goal is approved.
- [ ] Package the working local application and Neo4j environment with Docker Compose after the core workflow, evaluation, and plugin integration are complete.

### Deferred validation and hardening tasks

- [ ] Implement an asynchronous graph-validation agent for orphan subgraphs, duplicate identifiers, missing provenance, invalid labels, and schema violations as a stretch hardening layer.
- [ ] Produce validation findings with severity, node and edge IDs, evidence, and remediation status.
- [ ] Add more advanced CI, security scans, and artifact schema validation once the local plugin-based workflow is stable.
- [ ] Add additional benchmark suites that compare plugin and custom retrieval implementations under equal conditions.

### Deferred research and extension work

- [ ] Explore automatic retrieval routing, query-classification heuristics, and model-aware retrieval orchestration for future extension.
- [ ] Compare plugin GraphRAG and VectorRAG with ad hoc graph-exploration and keyword retrieval baselines under the final evaluation protocol.
- [ ] Extend the evaluation suite to include failure-mode analysis, plugin-specific latency breakdowns, and tuning trade-offs.
- [ ] Document plugin limitations, unsupported features, and edge cases discovered in local benchmarking.

These deferred items remain valuable, but they are explicitly not required to satisfy the current capstone core implementation once the Neo4j AI plugin is accepted as the primary GraphRAG and VectorRAG mechanism.
