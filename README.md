# GraphRAG Pipeline

A Python application that extracts a knowledge graph into Neo4j and answers
questions using Neo4j-backed retrieval and an OpenRouter or Ollama model.

## macOS Quick Start

The verified local baseline is macOS 15 on Apple Silicon with Python 3.14.3.
Intel Macs are not yet verified. Install [Homebrew](https://brew.sh/) if needed,
then install `uv` and the locked project environment:

```sh
brew install uv
git clone <repository-url>
cd project-repo
uv sync --locked --extra dev
uv run --locked python -m pytest tests/unit -v
```

`uv` reads `.python-version`, installs the selected Python if necessary, creates
`.venv`, and installs the exact versions recorded in `uv.lock`. No activation
command is needed. The unit tests need neither Neo4j nor model credentials.

## Live GraphRAG Workflow

A live run additionally needs a Neo4j database, model-provider access, and a
Neo4j graph/vector index. The versions below reflect the locally verified
baseline, not a guarantee for every Neo4j release.

1. Install [Neo4j Desktop](https://neo4j.com/download/) and create/start a local
   Neo4j Enterprise `2026.08.1` database. In Desktop, install APOC and the
   Neo4j GenAI plugin if their procedures are not available, then restart the
   database. Verify the server and procedures using
   [the compatibility notes](agent-coordination/neo4j/compatibility.md).
2. Copy `.env.example` to `.env`. Set the Neo4j Bolt URI, username, and password,
   then provide an OpenRouter API key or configure the Ollama provider. Never
   commit `.env`. Live extraction and answering send text to the selected
   provider and may incur charges.
3. Run the small, tracked extraction fixture against the database:

   ```sh
   uv run --locked python -m graphrag_pipeline.extraction.run_fixture
   ```

   This writes graph data and makes provider calls. It does not reset the
   database.
4. In Neo4j Desktop's query tool, create the vector index using the
   [Neo4j vector-index syntax](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).
   The example matches the checked-in OpenRouter model's 1,536 dimensions and
   cosine similarity; change the dimensions and `.env` settings together if
   using a different embedding model.

   ```cypher
   CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
   FOR (chunk:Chunk)
   ON chunk.embedding
   OPTIONS {indexConfig: {
     `vector.dimensions`: 1536,
     `vector.similarity_function`: 'cosine'
   }};
   ```

   Wait for the index to become `ONLINE`. The pipeline verifies the index; it
   does not create it.
5. Verify the live prerequisites:

   ```sh
   uv run --locked python -m graphrag_pipeline.retrieval_prerequisites
   ```

6. Submit a custom question and write a local artifact:

   ```sh
   uv run --locked graphrag-query \
       --question "What company did Aster Systems acquire in 2021?" \
     --artifact runs/mac-smoke.json
   ```

The query needs an existing graph and online vector index. A successful
retrieval also needs the configured embedding and answer provider. Generated
artifacts under `runs/` are local and ignored by Git.

## Full MuSiQue Data (Optional)

The official dataset is not included in a clone. See
[dataset/README.md](dataset/README.md) for acquisition instructions and the
expected location. Once downloaded, set `GRAPH_RAG_DATASET_PATHS` in `.env` to
the desired JSONL file(s), then select a record with `--record-id`. Dataset
ingestion can make many provider calls; use `--dry-run` before a paid run.

## Further Documentation

- [Operational details and command behavior](documentation.md)
- [Neo4j compatibility and local verification](agent-coordination/neo4j/compatibility.md)
- [Data contracts](agent-coordination/data-contracts.md)
- [MuSiQue dataset description and license](agent-coordination/musique-dataset-description.md)

## License

The application source does not yet declare a project license. Choose and add
one before public distribution. The MuSiQue dataset's CC BY 4.0 license applies
to that dataset, not automatically to this project's source code.