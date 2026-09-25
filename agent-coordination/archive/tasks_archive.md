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

## Historical planning context

The archived tasks originated in the pre-plugin-first roadmap. They remain here
to explain why the repository contains fallback extraction schemas and provider
tests, but new implementation work should follow `agent-coordination/tasks.md`
and `agent-coordination/data-contracts.md`.
