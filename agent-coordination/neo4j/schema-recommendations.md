# MuSiQue schema recommendations

Purpose
-------
This document records a conservative, practical GraphSchema recommendation for MuSiQue ingestion using the `SimpleKGPipeline` (neo4j-graphrag). The intent is to eliminate intermittent `SchemaValidationError` raised by the package's automatic schema extraction step (conflicting `KEY`/`UNIQUENESS`/`EXISTENCE` constraints) while preserving the current pipeline architecture and provenance behavior.

High-level guidance
-------------------
- Prefer a small, stable core ontology rather than allowing unconstrained LLM-driven schema changes per paragraph.
- Be permissive about properties (allow `additional_properties: True`) but strict about identity properties you control (`document_id`, `chunk_id`).
- Avoid `KEY` constraints unless you can guarantee a stable composite key dataset-wide. Use `UNIQUENESS` only for system IDs you mint.
- Do not apply `UNIQUENESS` or `EXISTENCE` to free-text `name` properties for People/Places.

Recommended minimal core schema (conceptual)
--------------------------------------------
- Node types:
  - `Document` — properties: `document_id: STRING` (UNIQUENESS)
  - `Chunk` — properties: `chunk_id: STRING` (UNIQUENESS), `paragraph_idx: INTEGER`
  - Domain labels (conservative set): `Person`, `City`, `Country`, `Region`, `State`, `Province`, `Organization`, `School`, `Event`, `Work`, `Musical`
    - properties: `name: STRING` (no UNIQUENESS/EXISTENCE)
    - `additional_properties: True`
- Relationship types (examples): `FROM_DOCUMENT`, `FROM_CHUNK`, `ACQUIRED` (no strict property constraints)
- Constraints: only `UNIQUENESS` on `document_id` and `chunk_id`.

Why this works
--------------
- Passing a curated `GraphSchema` to `SimpleKGPipeline(schema=...)` prevents the pipeline from invoking the automatic LLM schema-extraction step that produced conflicting `KEY` + `UNIQUENESS`/`EXISTENCE` pairs.
- By limiting strict constraints to identifiers we control, we remove the validation paths in the library that raise `SchemaValidationError` while keeping useful guidance to the extractor (labels and typical properties).

Practical workflow to adopt
--------------------------
1. Create the initial, conservative `GraphSchema` as a Python dict or Pydantic `GraphSchema` object and pass it to `SimpleKGPipeline(schema=...)` in `build_simple_kg_pipeline`.
2. Run small samples (use `--dry-run` or `--records N`) and gather the emitted labels, relationship types, and property names.
3. Curate: add recurring, unambiguous labels/properties to the schema. For any candidate uniqueness or key, verify dataset-wide uniqueness before adding.
4. Re-run a larger sample and add tests that verify idempotent writes, and that `document_id`/`chunk_id` uniqueness holds.

Implementation notes
--------------------
- Keep `perform_entity_resolution=False` for initial ingestion runs; entity resolution can run later against a stable canonical index.
- If you need to remain fully dynamic, implement a small preprocessing step that accepts the LLM-extracted schema but drops `EXISTENCE` and `UNIQUENESS` constraints that conflict with any `KEY` constraints before validating/using the schema. This mirrors an approach if you want to keep `schema=EXTRACTED` but avoid library validation errors.

Example skeleton (illustrative Python dict)
-----------------------------------------
```python
graph_schema = {
    "node_types": [
        {"label": "Document", "properties": [{"name": "document_id", "type": "STRING"}], "additional_properties": True},
        {"label": "Chunk", "properties": [{"name": "chunk_id", "type": "STRING"}, {"name": "paragraph_idx", "type": "INTEGER"}], "additional_properties": True},
        {"label": "Person", "properties": [{"name": "name", "type": "STRING"}], "additional_properties": True},
        # add other conservative domain labels here
    ],
    "relationship_types": [
        {"label": "FROM_DOCUMENT", "properties": []},
        {"label": "FROM_CHUNK", "properties": []},
        {"label": "ACQUIRED", "properties": []},
    ],
    "constraints": [
        {"type": "UNIQUENESS", "node_type": "Document", "property_names": ["document_id"]},
        {"type": "UNIQUENESS", "node_type": "Chunk", "property_names": ["chunk_id"]},
    ],
}
```

Next steps and maintenance
--------------------------
- I can: generate a script to scan a representative MuSiQue sample and propose a schema (labels/properties frequency), or
- Modify `build_simple_kg_pipeline` to accept a default `schema=` loaded from a YAML/JSON file in the repo and wire that into the pipeline constructor.

If you want, I will implement the schema file and the small loader that injects it into `build_simple_kg_pipeline` so ingestion runs are deterministic. Say which option you prefer (script to propose schema, or immediate pipeline change) and I'll implement it.

Live graph inventory (2026-10-17, Task 1 hardening)
----------------------------------------------------
Live inspection of the running Neo4j instance (one ingested MuSiQue record,
`2hop__153573_109006`, 19 paragraphs) to ground the GraphRAG traversal design:

- **Schema actually in use is free-form**, not the conservative core above:
  34 domain labels observed (e.g. `Person`, `WebBrowser`, `Organization`,
  `Zoo`, `ZipLineCourse`, ...) and ~50 relationship types, including
  inconsistent casing (`DEVELOPED_BY` alongside `developed_by`, `developed`,
  `locatedIn`/`LOCATED_IN`). The retrieval traversal must not hardcode a
  relationship allowlist; it discovers types live via `CALL
  db.relationshipTypes()` and only excludes lexical-graph infrastructure
  types (see `graphrag_pipeline.retrieval._discover_relationship_allowlist`).
- **Node/property reality differs from the conceptual schema above**: the
  vector-indexed `Chunk` node only has `text`/`embedding`/`index` — it does
  **not** carry `chunk_id`/`record_id`/`document_id`/`title`/`source_file`.
  Those provenance fields live on the `Document` node, one hop away via
  `(Chunk)-[:FROM_DOCUMENT]->(Document)`. Retrieval code that read these
  properties directly off the vector-matched node was a real, reproduced bug
  (fixed in `retrieval.py`'s `_retrieval_query_vector`/`_vector_cypher`).
- **Connectivity is entity-[:FROM_CHUNK]->Chunk-[:FROM_DOCUMENT]->Document**;
  `Document` has no direct relationship to any entity, and entities have no
  `record_id` property of their own.
- **No cross-chunk or cross-record connectivity exists today**:
  `perform_entity_resolution=False` and per-chunk-scoped extraction IDs mean
  every paragraph's entities and entity-entity relationships are entirely
  self-contained to that one chunk (confirmed: zero entities with more than
  one `FROM_CHUNK` edge; zero entity-entity relationships spanning two
  different chunks). Each paragraph is already its own naturally-bounded
  component; no multi-paragraph/record-root boundary logic is required for
  the current ingestion design.
- **Traversal safety for a hypothetical future shared entity** (e.g. if
  entity resolution is enabled later) was verified with a synthetic fixture
  (`tests/integration/test_retrieval_graph_traversal.py`): restricting the
  `FROM_CHUNK` relationship to incoming-direction-only in the traversal
  filter means a shared entity's *other* chunk/document is never reachable
  (that specific edge is always outgoing from the entity's side), while
  domain relationships a shared entity genuinely participates in correctly
  still traverse. The seed's own `chunk_id`/`document_id` provenance is
  always the seed's, never swapped for another chunk's, regardless of
  entity sharing.
- The `chunk_embedding` vector index was found misconfigured for 1536
  dimensions while the configured Ollama embedding model (and the actually
  stored `Chunk.embedding` values) are 768-dimensional; it was dropped and
  recreated at 768 dimensions (cosine) to unblock live retrieval testing.
