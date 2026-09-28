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
