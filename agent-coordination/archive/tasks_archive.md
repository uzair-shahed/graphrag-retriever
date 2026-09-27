# Archived Tasks

## Purpose

This file preserves roadmap items that were superseded by the active Neo4j AI
plugin architecture. They are historical planning context, not current project
requirements.

## Superseded custom extraction work

- Implement a hand-built `ExtractionProvider` as the primary extraction path.
- Implement a custom extraction agent that parses provider output into the
  manual Contract 2 result schema before graph writes.
- Add corrective extraction retries and manual dead-letter handling as the
  required active workflow.
- Add a LangGraph extraction node whose purpose is to replace
  `SimpleKGPipeline`.
- Build custom Cypher write logic for plugin-managed extraction entities and
  relationships.

The validated Pydantic extraction schemas and deterministic provider remain
available as compatibility and fallback material in the source tree. They are
not the active extraction implementation.

## Superseded custom retrieval work

- Build a custom hand-written Vector RAG implementation outside Neo4j.
- Build a custom graph retrieval implementation that replaces Neo4j AI plugin
  GraphRAG behavior.
- Build a custom embedding pipeline, vector-index lifecycle, and chunk-to-vector
  workflow that replaces Neo4j AI plugin VectorRAG.
- Build a custom Hybrid Search stack as part of the core retrieval path.
- Add a manual retrieval fallback that bypasses the Neo4j plugin as a required
  core capability.
- Implement custom candidate generation and ranking that the Neo4j plugin is
  expected to provide.

Neo4j AI plugin VectorRAG and GraphRAG remain the active retrieval direction.
Multi-hop Cypher verification and evidence tracing remain active because they
validate grounded answers and are not replacements for plugin retrieval.

## Superseded platform assumptions

- Treat Ray workers, Polars, Redis, PostgreSQL, or DuckDB as required core
  infrastructure for the linear local workflow.
- Make Docker packaging a prerequisite for the core implementation.
- Rebuild plugin-replaced extraction or retrieval wrappers solely to preserve
  the earlier architecture diagram.

These technologies remain possible stretch work only when explicitly approved
and do not define the current local project boundary.

## Deferred Infrastructure And Stretch Work

The sections below were separated from the active roadmap so each deferred
technology or stretch area can be restored independently after core ingestion,
plugin retrieval, grounded-answer verification, and evaluation are complete.

## Docker Implementation

- Package the working local application and Neo4j environment with Docker Compose.
- Make the complete pipeline runnable with `docker-compose up` or the documented
  single-command equivalent.
- Test a clean Docker deployment from a fresh environment, including persistent
  volumes and initialization.
- Archive Docker startup and deployment logs when this work is completed.
- Confirm Docker results remain comparable with the local core workflow.

## Polars Implementation

- Add Polars processing for approved scaling or data-preparation work.
- Add Polars-based persistence for the Week 11 telemetry stretch if approved.
- Use Polars for heavy metadata normalization, cleaning, or benchmark preparation
  only when the corresponding stretch scope is approved.

## Ray Implementation

- Add Ray workers or parallelized processing for ingestion, extraction, or
  evaluation only after an approved stretch goal requires it.
- Replace the current linear flow with Ray-based parallelization only as a
  deliberate, benchmarked extension.
- Preserve the linear implementation as the reproducible core comparison path.

## Redis Implementation

- Add Redis-backed session or trace state for ephemeral agent telemetry if the
  project expands to a multi-user or multi-run environment.
- Add short-lived Redis traces and correlate them with Neo4j and benchmark
  artifacts as part of the telemetry stretch.
- Verify that telemetry survives the intended service restart and remains
  queryable by run ID.

## DuckDB Implementation

- Add DuckDB SQL pipelines for metadata extraction, cleaning, and structural
  normalization only when approved as stretch work.
- Use DuckDB for heavy benchmark-data preparation only if the core workflow
  requires that extension.

## Other Stretch Goals

Stretch work begins only after core functionality, evaluation, and advisor
approval are complete. If schedule or metrics are at risk, preserve core
reproducibility and documentation instead.

### Week 11: Tier 1 stretch goals

- Define query classes and a labeled routing set for Vector, Cypher, Hybrid, and
  fallback cases.
- Implement the retrieval-routing agent at the query entry point.
- Add a deterministic fallback and allow an explicit user override.
- Evaluate routing classification accuracy, misroutes, latency overhead, and
  impact on retrieval metrics.
- Define an audit event schema for prompts, state transitions, agent decisions,
  retrieval settings, evidence, errors, and token usage.
- Add durable PostgreSQL telemetry and historical run tracking if approved.
- Add trace correlation across Redis, PostgreSQL, Neo4j, and benchmark artifacts
  if the telemetry stretch is approved.
- Add access controls and redaction rules for prompts, secrets, and sensitive
  data.

**Deliverables**

- Intelligent query-routing layer.
- Persistent audit telemetry logging.

**Acceptance criteria**

- Routing decisions are inspectable and can be overridden.
- Telemetry survives the intended service restart and can be queried by run ID.
- Logging does not expose secrets and does not materially break latency targets.

**Advisor checkpoint**

- Assess routing accuracy and confirm the PostgreSQL telemetry schema
  integration.

### Week 12: Tier 2 stretch goals and hardening

- Implement an asynchronous graph-validation agent for orphan subgraphs,
  duplicate identifiers, missing provenance, invalid labels, invalid
  relationships, and schema violations.
- Produce validation findings with severity, node or edge IDs, evidence, and
  remediation status.
- Enforce typing across application and evaluation code.
- Refactor into modular packages with clear dependency boundaries.
- Complete unit, integration, contract, and end-to-end tests for the supported
  workflow.
- Add CI checks for formatting, linting, type checking, tests, security checks,
  and artifact schema validation.
- Add migration and version handling for graph schema and evaluation artifacts.
- Re-run core metrics after hardening to detect regressions.
- Add additional benchmark suites comparing plugin and custom retrieval
  implementations under equal conditions.

**Deliverables**

- Graph validation agent.
- Hardened typed and modular codebase.
- Complete test suite and CI checks.

**Acceptance criteria**

- Validation findings are actionable and do not silently mutate production data.
- A clean local setup succeeds from documented prerequisites.
- Core benchmark results remain comparable after refactoring.

### Week 13: Documentation, portfolio, and defense

- Finalize the capstone report, including the problem, research question,
  related work, architecture, methodology, implementation, datasets,
  experiments, results, limitations, ethics, and future work.
- Include enough configuration and protocol detail for independent reproduction.
- Prepare a public GitHub repository with setup, configuration, provenance,
  usage examples, architecture diagrams, tests, and benchmark scripts.
- Separate public code and metadata from secrets, restricted data, and local-only
  artifacts.
- Add a release tag or archive containing the evaluated code and configuration.
- Build the defense presentation, poster, demo script or recording, and answers
  for limitations, cost, latency, false merges, hallucinations, provenance, and
  alternative designs.
- Conduct a final advisor rehearsal, incorporate feedback, and obtain final
  advisor sign-off.

**Deliverables**

- Final MS Capstone report.
- Polished public repository and reproducibility package.
- Defense slide deck.
- Poster.
- Demo script or recording and final evidence archive.

**Acceptance criteria**

- Report, repository, figures, deck, poster, and demo use consistent final
  metrics.
- The report explains generation reliability, entity-resolution quality,
  retrieval performance, and operational efficiency.
- A clean evaluator can reproduce at least the core smoke test and benchmark
  commands.
- All known limitations and missed targets are disclosed.

### Week 14: Contingency and submission buffer

- Reserve time for advisor-requested corrections, failed reruns, infrastructure
  outages, and formatting requirements.
- Re-run the final smoke test and verify the release archive.
- Confirm submission portals, filenames, deadlines, and file formats.
- Submit the report, deck, poster, repository link, and supplementary artifacts.
- Archive final logs, benchmark outputs, environment versions, and decision
  records.

## Deferred Research Extensions

- Compare plugin GraphRAG and VectorRAG with ad hoc graph-exploration and
  keyword-retrieval baselines under the final evaluation protocol.
- Extend the evaluation suite with failure-mode analysis, plugin-specific
  latency breakdowns, and tuning trade-offs.
- Document plugin limitations, unsupported features, and edge cases discovered
  in local benchmarking.

## Historical planning context

The archived tasks originated in the pre-plugin-first roadmap. They remain here
to explain why the repository contains fallback extraction schemas and provider
tests, but new implementation work should follow `agent-coordination/tasks.md`
and `agent-coordination/data-contracts.md`.
