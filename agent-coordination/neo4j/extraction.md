# Neo4j GenAI Extraction

**Last modified:** 23 September 2026

**Purpose of this file:**
Summarize the official Neo4j GraphRAG Python Knowledge Graph Builder behavior
needed to convert MuSiQue text into a Neo4j knowledge graph.

**Files that use this file:**
`agent-coordination/tasks-goals-23-sept-2026.md` uses these notes for the
`SimpleKGPipeline` capability and fixture-run tasks. Future pipeline setup,
graph verification, and Contract 4 updates should consult this file.

**Caveats:**
Knowledge Graph Builder is experimental. The notes describe package version
`neo4j-graphrag==1.21.0` as installed locally, but the API may change. The
pipeline writes graph data through a Neo4j driver; it does not automatically
satisfy this project's MuSiQue provenance contract. Provenance must be carried
through document metadata or a post-run mapping and then verified.

## Official Entry Point

```python
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

pipeline = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    embedder=embedder,
    from_file=False,
    schema=graph_schema,
    on_error="RAISE",
    perform_entity_resolution=False,
    neo4j_database="neo4j",
)
await pipeline.run_async(
    text=paragraph_text,
    document_metadata=source_metadata,
)
```

Use `from_file=False` when supplying already-loaded MuSiQue paragraph text. The
pipeline requires an LLM, Neo4j driver, and embedder. The project should pass
one paragraph first, not a complete JSONL file.

The project input adapter is
`src/graphrag_pipeline/extraction/plugin_input.py`. It accepts the validated
Contract 2 `ExtractionRequest` and returns `(text, document_metadata)` for the
pipeline. `text` is the original paragraph unchanged. Metadata carries source
identity and provenance; gold supervision fields are never passed to the
plugin.

## Pipeline Components

The documented KG Builder pipeline can include:

1. data loader;
2. text splitter;
3. optional chunk embedder;
4. schema builder;
5. optional lexical graph builder;
6. entity and relation extractor;
7. graph pruning;
8. Neo4j graph writer; and
9. entity resolver.

For this project, the important graph objects are the lexical `Document` and
`Chunk` nodes plus extracted entity and relationship nodes. The default lexical
graph uses document/chunk relationships and can attach chunk embeddings when
configured.

## Schema Guidance

A schema can define:

- node types, labels, descriptions, and properties;
- relationship types, descriptions, and properties; and
- allowed `(source_label, relationship_type, target_label)` patterns.

Prefer an explicit project schema once the MuSiQue domain vocabulary is known.
Automatic schema extraction is available, but it adds another LLM call and
another source of variability. `schema="EXTRACTED"` or `schema=None` derives a
guiding schema from text; `schema="FREE"` disables schema guidance.

Schema guidance is not automatically a strict allow-list by default. Configure
additional node types, relationship types, patterns, and properties explicitly
if the project needs pruning or strict graph vocabulary.

## Structured Output

The package supports structured extraction when the selected LLM advertises
structured-output support. Official documentation identifies OpenAI, Vertex AI,
and Anthropic implementations as supported for this path. Structured output
uses the package's graph Pydantic model and is preferable to prompt-only JSON
parsing when available.

Provider selection must therefore answer:

- Does the provider support the package's structured-output interface?
- Which package extra installs its client?
- Does the model support the required response schema and token budget?
- Are rate limits and retries configured?

## Error Handling

The extractor supports `on_error="IGNORE"` and `on_error="RAISE"` behavior.
The documented default may ignore a failed chunk and save successful chunks;
for the first project smoke test, use `RAISE` so an extraction failure cannot
look like a successful graph build. Record the failed chunk and source
provenance in the run artifact.

## Entity Resolution

`SimpleKGPipeline` performs entity resolution by default, merging entities with
the same label and name. The package also documents exact, spaCy-semantic, and
fuzzy resolvers. Disable built-in resolution during the first extraction
verification if the project needs to inspect raw extraction output separately;
choose and document the resolver explicitly before evaluation.

## Provenance Mapping

Use `document_metadata` for source metadata when the package preserves it on the
Document node. At minimum, attempt to carry:

```text
source_dataset = musique
source_split = fixture/train/dev/test
record_id
chunk_id
paragraph_idx
source_file
source_sha256
source_line
```

Do not assume that a metadata dictionary automatically appears on every entity
or relationship. Query the graph after the run and verify the exact location of
source metadata. If the plugin does not propagate `chunk_id` to extracted facts,
add a documented mapping step or fail the provenance acceptance test.

## Project Smoke-Test Checklist

- load `tests/fixtures/musique/extraction_chunk.json`;
- exclude `is_supporting`, answer, and decomposition labels from the text input;
- pass the original paragraph text unchanged;
- run with an explicit schema and `on_error="RAISE"`;
- use a clean fixture graph;
- capture the `PipelineResult`, run metadata, and provider configuration;
- inspect Document, Chunk, entity, and relationship labels/properties;
- verify source metadata and chunk identity;
- rerun and measure duplicate behavior;
- preserve the result even when the run fails.

## Official Sources

- [Neo4j GraphRAG for Python](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [Knowledge Graph Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)
- [GraphRAG API: SimpleKGPipeline](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html#simplekgpipeline)
- [GraphRAG API: Entity and Relation Extractor](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html#llmentityrelationextractor)

Documentation consulted: 23 September 2026.
