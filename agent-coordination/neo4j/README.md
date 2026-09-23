# Neo4j GenAI Reference

**Last modified:** 23 September 2026

**Purpose of this file:**
Index the local, token-efficient notes for implementing Neo4j GenAI extraction
and retrieval in this project.

**Files that use this file:**
Developers working on `agent-coordination/tasks-goals-23-sept-2026.md`,
`agent-coordination/data-contracts.md`, Neo4j integration code, and retrieval
benchmarks should start here.

**Caveats:**
These are implementation notes and not a replacement for the official
versions linked in each file. The Knowledge Graph Builder is documented as
experimental; verify the installed `neo4j-graphrag` version before relying on
API details. Secrets and provider-specific configuration are intentionally not
stored here.

## Notes

- [Extraction and Knowledge Graph Builder](extraction.md)
- [Retrieval and GraphRAG](retrieval.md)
- [Compatibility and local verification](compatibility.md)

## Project Baseline

- Local Neo4j server: Enterprise `2026.08.1`.
- Local Python: 3.14.3.
- Installed client package: `neo4j-graphrag==1.21.0`.
- `SimpleKGPipeline` is a Python API, not a server-side Cypher procedure.
- The repository currently uses local Python for input preparation, provenance,
  graph verification, and configuration.
- The first real integration should use the focused MuSiQue fixture before any
  full-dataset or paid-scale run.
