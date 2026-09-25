# Data Contracts

## Purpose

This document records the active, implemented boundary for the current project. It is intentionally narrower than the original design doc and is limited to the verified local workflow:

1. MuSiQue records and paragraphs are normalized into validated chunk-level inputs.
2. The active extraction engine is Neo4j GenAI Plugin `SimpleKGPipeline`.
3. The local Python layer validates the source text and provenance before invoking the plugin.
4. The plugin writes directly to Neo4j, and the graph itself is treated as the source of truth for extraction output.
5. Post-run verification checks the persisted graph against the original MuSiQue chunk provenance.

The original, longer hand-built extraction and answer-generation design has been archived to [agent-coordination/archive/data-contracts_archive.md](archive/data-contracts_archive.md). The live implementation is the plugin-first path that was validated in the 23 September 2026 task work.

### Boundary invariants

- The dataset adapter reads immutable local JSONL input and emits one validated chunk per MuSiQue paragraph.
- The active extraction engine is Neo4j GenAI Plugin `SimpleKGPipeline`; application code passes `text` and `document_metadata` and treats the resulting Neo4j graph as the output surface.
- Gold labels such as `is_supporting` remain in the dataset/chunk store but are not sent as model-visible extraction input.
- Provenance is preserved from the source JSONL file through the chunk, plugin input, and persisted graph state.
- Neo4j access for the live extraction path is write-capable only for the plugin-managed graph write; read-only verification is used after the run.
- The current local implementation was verified against a successful focused MuSiQue fixture run and graph inspection in the local `neo4j` database.

## Architecture Scope

```text
MuSiQue JSONL
  -> MuSiQue adapter
  -> validated chunk
  -> build_plugin_input(text, document_metadata)
  -> SimpleKGPipeline.run_async(...)
  -> persisted Neo4j graph
  -> read-only provenance verification
```

## Contract Handoff Map

```text
Contract 1
MuSiQue record -> validated application chunk

Contract 2
validated chunk -> plugin input metadata contract

Contract 4
plugin-created Neo4j graph -> read-only provenance verification
```

The active system does not rely on the archived manual extraction schema as the source of truth. The earlier, more detailed extraction and answer-generation contracts remain useful for historical reference, but they are not active project doctrine.

## Contract 1: MuSiQue Record To Application Chunk

**Status:** Active and validated.

### Owner

- Producer: MuSiQue adapter
- Consumer: extraction pipeline and graph verification

### Purpose

Convert one MuSiQue JSONL record into one validated application chunk per paragraph while preserving dataset provenance and source text without rewriting it.

### Input

```json
{
  "id": "2hop__460946_294723",
  "question": "<question text>",
  "paragraphs": [
    {
      "idx": 3,
      "title": "<Wikipedia title>",
      "paragraph_text": "<paragraph text>",
      "is_supporting": true
    }
  ]
}
```

### Output

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "document_id": "musique:document:<normalized-title>",
  "paragraph_idx": 3,
  "title": "<Wikipedia title>",
  "text": "<unchanged paragraph text>",
  "is_supporting": true,
  "source_dataset": "musique",
  "source_split": "dev",
  "provenance": {
    "source_dataset": "musique",
    "source_split": "dev",
    "source_file": "musique_ans_v1.0_dev.jsonl",
    "source_sha256": "<file checksum>",
    "source_line": 1842
  }
}
```

### Required fields

- `chunk_id`
- `record_id`
- `document_id`
- `paragraph_idx`
- `title`
- `text`
- `source_dataset`
- `source_split`
- `provenance.source_file`
- `provenance.source_sha256`
- `provenance.source_line`

### Validation rules

- `id` must be present and non-empty.
- Each paragraph must have an integer `idx` and non-empty paragraph text.
- The original paragraph text must be preserved without mutation.
- `chunk_id` must be deterministic for the same record and paragraph index.
- `is_supporting` may be present for labeled records but is not allowed as model-visible extraction input.

### Failure behavior

Malformed records are dead-lettered with the source file name, checksum, source line, record ID, and error category instead of being silently rewritten.

## Contract 2: Chunk To Neo4j Plugin Input

**Status:** Active and validated. The local Python adapter is implemented in [src/graphrag_pipeline/extraction/plugin_input.py](../src/graphrag_pipeline/extraction/plugin_input.py) and the pipeline factory is implemented in [src/graphrag_pipeline/extraction/pipeline.py](../src/graphrag_pipeline/extraction/pipeline.py).

### Owner

- Producer: local Python validation layer
- Consumer: Neo4j GenAI Plugin `SimpleKGPipeline`

### Purpose

Define the actual boundary between a validated MuSiQue chunk and the plugin-managed extraction run. The local app passes the chunk text unchanged and sends source provenance via `document_metadata`.

### Active plugin boundary

```python
text, document_metadata = build_plugin_input(request)
result = await pipeline.run_async(
    text=text,
    document_metadata=document_metadata,
)
```

### Required input envelope

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "document_id": "musique:document:<normalized-title>",
  "paragraph_idx": "3",
  "title": "<Wikipedia title>",
  "source_dataset": "musique",
  "source_split": "dev",
  "source_file": "musique_ans_v1.0_dev.jsonl",
  "source_sha256": "<file checksum>",
  "source_line": "1842"
}
```

The active contract uses strings for the GraphRAG metadata fields that the plugin validates, while the application request itself keeps numeric values in their typed form. Gold labels such as `is_supporting` remain outside the plugin input.

### Validation rules

- `text` must be the unchanged paragraph text from the validated chunk.
- `document_metadata` must include chunk, record, document, and provenance identifiers.
- `source_line` and `paragraph_idx` are normalized to strings before the plugin call because the plugin metadata validator requires that representation.
- `is_supporting`, answer labels, decomposition metadata, and evaluation-only fields are intentionally excluded from the plugin payload.

### Failure behavior

A provider boundary or metadata validation failure is treated as a plugin-integration failure, not a successful extraction. The caller must surface the failure and keep the source provenance attached to the dead-letter or log record.

## Contract 4: Plugin-Generated Neo4j Graph To Provenance Verification

**Status:** Active and verified. The verification helper is implemented in [src/graphrag_pipeline/verification.py](../src/graphrag_pipeline/verification.py).

### Owner

- Producer: Neo4j GenAI Plugin `SimpleKGPipeline`
- Consumer: local verification and later retrieval/query layers

### Purpose

Define the graph shape that the live plugin is known to create and the provenance checks that prove the source chunk is correctly retained in the graph.

### Verified graph vocabulary

The live smoke test validated these graph elements:

- `Document`: source-backed document record carrying the MuSiQue provenance metadata.
- `Chunk`: paragraph-level evidence unit identified by `chunk_id` and containing source text.
- `Entity`: plugin-created extracted entity nodes.
- `FROM_CHUNK` and `FROM_DOCUMENT`: graph links created by the plugin to bind the graph evidence back to the source chunk/doc.
- `ACQUIRED`: the domain relationship created in the smoke-test fixture.

### Provenance contract

The key invariant is that every plugin-created graph fact must map back to the original MuSiQue evidence unit. The verification helper checks for:

- a `Document` node with the source record/chunk metadata;
- a `Chunk` node with the matching `chunk_id` and source `text`;
- the `FROM_CHUNK` and `FROM_DOCUMENT` relationship types in the graph;
- the expected plugin-created relationship vocabulary including `ACQUIRED`.

### Read-only verification behavior

```python
verify_graph_provenance(connection, chunk_id, record_id)
```

This function does not modify Neo4j. It queries the database to confirm that the graph contains a document and chunk for the source record and that the expected relationship types exist.

### Required checks

- `Document.chunk_id` or `Document.record_id` must match the source record.
- `Chunk.chunk_id` must match the chunk under test.
- `Chunk.text` must be present and non-empty.
- The graph must contain the expected plugin-created relationship types, at minimum `FROM_CHUNK` and `FROM_DOCUMENT`.

### Failure behavior

If the graph is missing the expected document/chunk or relationship type, the verifier raises a validation error rather than silently accepting the result as successful.

## Contract 5: Future Retrieval Boundary

**Status:** Planned but not implemented in the active vertical slice.

The archived manual retrieval contract has been moved to [agent-coordination/archive/data-contracts_archive.md](archive/data-contracts_archive.md). The current project scope is extraction proof and provenance validation, not a full answer-generation or retrieval implementation.

## Contract Completion Checklist

- [x] Contract 1 reviewed against a real MuSiQue chunk contract.
- [x] Contract 2 implemented and validated as the active plugin input boundary.
- [x] Contract 4 implemented and verified against the live Neo4j graph produced by `SimpleKGPipeline`.
- [x] Legacy hand-built extraction and answer-generation contract text moved to archive.
- [ ] Contract 5+ are implemented when retrieval, answering, and reflection are added to the active architecture.
- [ ] Every active future contract must name owner, consumer, validation rules, provenance requirements, and failure behavior before being treated as active project doctrine.
