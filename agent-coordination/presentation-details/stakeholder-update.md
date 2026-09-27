# MuSiQue Graph Pipeline

## Completed

- **Local Neo4j environment:** Verified Neo4j Enterprise connectivity, bounded transactions, retries, and fixture graph smoke tests.
- **MuSiQue data boundary:** Created validated paragraph-level chunks with stable record, document, paragraph, and chunk IDs.
- **Provenance preservation:** Preserved source dataset, split, file, checksum, line, record, title, and paragraph metadata.
- **Plugin-first extraction:** Established Neo4j GenAI Plugin `SimpleKGPipeline` as the active extraction engine.
- **OpenRouter integration:** Connected `gpt-4o-mini` for extraction and `text-embedding-3-small` for embeddings through the OpenAI-compatible endpoint.
- **Input protection:** Kept answers, supporting labels, aliases, and question decomposition out of model-visible extraction input.
- **Live graph run:** Successfully processed the focused MuSiQue fixture and persisted source-backed documents, chunks, entities, and relationships in Neo4j.
- **Graph verification:** Added read-only checks for graph existence, source text, chunk identity, provenance, and plugin-created relationships.
- **Focused tests:** Added and passed tests for configuration, schema boundaries, plugin input, provenance, graph verification, Neo4j connectivity, and failure behavior.
- **Cost planning:** Established a stakeholder budget for the approved `MuSiQue-Full` train and dev datasets using measured fixture token usage.

## Current Budget Baseline

- Approved maximum: **44,710 records**
- Estimated full extraction and embedding cost: **$24.16**
- Models: `openai/gpt-4o-mini` and `openai/text-embedding-3-small`

## Next Scope

- Process selected MuSiQue-Full train/dev subsets.
- Add graph reset and rerun idempotence checks.
- Implement Neo4j AI Plugin GraphRAG and VectorRAG retrieval.
- Add LangGraph answering and reflection workflows.
- Build the frozen evaluation and benchmark artifacts.
