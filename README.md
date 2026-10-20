# GraphRAG Pipeline

A Python application that extracts a knowledge graph into Neo4j and answers
questions using Neo4j-backed retrieval and an OpenRouter or Ollama model.

## VectorRAG vs. GraphRAG evaluation

`graphrag-eval` evaluates the MuSiQue answerable dev file with RAGAS Triad
scores and gold answer/support/hop metrics. Every `run` compares both VectorRAG
and GraphRAG. The supported source is
`dataset/musique/data/data/musique_ans_v1.0_dev.jsonl`.

| Command | What it does | Example selecting records 4–6 |
| --- | --- | --- |
| `prepare` | Ingest missing selected records and reuse verified chunks; may write Neo4j and call extraction/embedding models. | `graphrag-eval prepare --start-record 4 --records 3` |
| `check` | Read-only verification of selected records and graph/index prerequisites; makes no model calls. | `graphrag-eval check --start-record 4 --records 3` |
| `run` | Check readiness, generate paired answers and RAGAS/gold scores, and write evaluation artifacts. Never ingests data. | `graphrag-eval run --start-record 4 --records 3 --output runs/evaluation-4-to-6` |
| `report` | Generate or regenerate offline `report.html` from saved artifacts, without Neo4j, dataset access, or models. | `graphrag-eval report --input runs/evaluation-4-to-6` |

Run the commands from the repository root using the locked environment. Start
with a selection preview, which needs neither Neo4j nor model services:

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3 --dry-run
```

This prints the selected IDs/source lines and planned paragraph count. It does
not establish that the graph or providers are ready. If records 4–6 are already
ingested, evaluate them directly:

```sh
uv run --locked graphrag-eval check --start-record 4 --records 3
uv run --locked graphrag-eval run --start-record 4 --records 3 --answer-provider ollama --eval-provider ollama --output runs/evaluation-4-to-6
```

The run uses Ollama for answers and judging. Its default conditions are both
record-scoped and corpus-wide, producing 12 rows for these three questions.
Retrieval embeddings retain the configured `GRAPH_RAG_PROVIDER`; their model
must match the graph/index. If the check reports missing records, prepare that
same range explicitly, then repeat the check and run:

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3
```

No selectors means the entire dev file. `--records 10` selects the first ten
records; `--max-records 10` is equivalent. Repeated `--record-id` selects exact
IDs and cannot be mixed with range flags. Use `--condition record` for
record-scoped evaluation only, or `--condition corpus` to search the whole
existing index for the selected questions. Repeated evaluations need a new or
empty `--output` directory.

`run` shows a tqdm progress bar on stderr by default, counting saved evaluation
results (12 for the three questions above). It displays the active record,
condition, backend, and answering/scoring phase. Add `--no-progress` to hide it;
the final JSON summary still prints to stdout.

Every evaluation also generates a self-contained **`report.html`**. Open it in
a browser for charts and searchable, side-by-side answers, gold answers/aliases,
gold supporting paragraphs, multi-hop labels, citations, retrieved evidence,
and the exact answer context. Record and corpus conditions remain separate.
Download each chart as PNG or SVG for slides, or use the report's print buttons
and your browser's **Save as PDF** for the overview, selected record, or entire
report. All report content works offline; JavaScript must be enabled.

Create the report for a past run without repeating evaluation:

```sh
uv run --locked graphrag-eval report --input runs/eval-prepared-4-to-6-part-3
```

This atomically replaces only that directory's `report.html`. Large runs produce
larger files; printing an entire run expands all contexts and can take time.
See the [report command reference](vectorrag_vs_graphrag_evaluation_ragas.md#report-reference)
for partial runs, artifact validation, and optional browser tests.

The [complete CLI reference](vectorrag_vs_graphrag_evaluation_ragas.md) lists
**every option, type, default, and applicable subcommand**, with
[use-case examples](vectorrag_vs_graphrag_evaluation_ragas.md#examples-by-use-case),
[provider precedence](vectorrag_vs_graphrag_evaluation_ragas.md#provider-configuration-and-precedence),
and [exit codes](vectorrag_vs_graphrag_evaluation_ragas.md#exit-codes-and-recovery).
Use `uv run --locked graphrag-eval run --help` for command-line help or see the
[operational evaluation reference](documentation.md#paired-vectorraggraphrag-evaluation-command-reference).

## Standalone paragraph extraction benchmark

`graphrag-extraction-benchmark` compares entity and relationship extraction on
**150 individual MuSiQue paragraphs** using OpenRouter or Ollama. First generate
model-produced gold with an explicitly selected OpenRouter reasoning model,
then compare candidate models using deterministic entity/relation precision,
recall and F1. It saves JSONL results and accesses no Neo4j or RAGAS services.
These scores measure extraction quality as a proxy for possible GraphRAG impact.

The [complete benchmark reference and milestone record](agent-coordination/entity-extraction-benchmark/entity-extraction-benchmark.md)
contains artifact formats, scoring rules, recovery details, tests and every
flag. Run these commands from the repository root:

```sh
uv sync --locked --extra dev
uv run --locked graphrag-extraction-benchmark --help
uv run --locked graphrag-extraction-benchmark select
```

`select` previews the first 150 paragraphs in dataset/list order from
`dataset/musique/data/data/musique_ans_v1.0_dev.jsonl`. It writes no artifacts and
makes no service calls. Use `--paragraphs 3` for a smaller preview.

### Generate and resume OpenRouter gold labels

Set `OPENROUTER_API_KEY` in `.env` or your environment. Replace
`YOUR_REASONING_MODEL_ID` with your chosen OpenRouter model ID supporting
structured output and the requested reasoning effort. Model support is checked
before generation. Start with a local preview, then generate the gold:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150 --dry-run
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150
```

Gold uses OpenRouter and requires an explicit enabled reasoning effort. Calls
may incur charges. The default selection is 150 paragraphs; a separate smoke
run can use `--paragraphs 2` and a different output directory.

Resume with the same model, effort, budget and other settings:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150 --resume
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150 --resume --retry-failed
```

Resume skips saved attempts. `--retry-failed` also retries failures while
preserving successful rows. Changed settings or source content require a new
output directory. Candidate evaluation requires complete successful gold.

### Choose Ollama, OpenRouter, or both for candidates

For a local candidate, start Ollama, install the chosen model and use its tag:

```sh
ollama pull qwen3:4b
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate
```

For a hosted candidate, use its OpenRouter model ID:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model openrouter:openai/gpt-4o-mini --output runs/entity-extraction/hosted-candidate
```

Repeat `--model` to compare several models against the same gold in one run:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --model openrouter:openai/gpt-4o-mini --ollama-think false --output runs/entity-extraction/mixed-models
```

Only the first colon separates the provider from the model: `ollama:qwen3:4b`
sends model `qwen3:4b` to Ollama. Models are examples; availability and parameter
support depend on your endpoint. For reasoning-capable hosted candidates, add
`--reasoning-effort medium` or another supported effort. Otherwise omit it to
use the model default. Ollama accepts `--ollama-think default`, `true`, `false`,
or a model-supported string. Use separate runs to compare the same model under
different settings. Candidate runs support `--dry-run`, `--resume` and
`--retry-failed` too.

Each candidate receives title/text and the same structured schema, without gold
labels. Per-paragraph and micro scores include failed predictions as empty sets.
Artifacts include `manifest.json`, `cohort.jsonl`, `gold.jsonl` or
`predictions.jsonl`, and `summary.json`; candidate runs freeze a local gold copy.

### Offline summaries and all flags

Regenerate a saved run's JSON summary without providers or the original dataset:

```sh
uv run --locked graphrag-extraction-benchmark report --input runs/entity-extraction/mixed-models
```

| Flag | Commands | Default / accepted values |
| --- | --- | --- |
| `--dataset-path PATH` | `select`, `gold` | MuSiQue dev JSONL path above. |
| `--paragraphs N` | `select`, `gold` | `150`; positive paragraph count. |
| `--model MODEL` | `gold` | Required OpenRouter model ID. |
| `--model PROVIDER:MODEL` | `run` | Required, repeatable; `ollama` or `openrouter`. |
| `--reasoning-effort LEVEL` | `gold`, `run` | Required for gold: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Optional for candidates, also allowing `none`; omitted uses model default. |
| `--ollama-think VALUE` | `run` | `default`, `true`, `false`, or model-supported string. |
| `--gold DIR` | `run` | Required complete gold run directory. |
| `--output DIR` | `gold`, `run` | Required new/empty directory, or matching existing run with `--resume`. |
| `--max-tokens N` | `gold`, `run` | `16384`; positive output budget including reasoning where applicable. |
| `--timeout SECONDS` | `gold`, `run` | `180`; positive finite per-request timeout. |
| `--retries N` | `gold`, `run` | `2`; additional transient HTTP/transport retries. |
| `--openrouter-base-url URL` | `gold`, `run` | `OPENROUTER_BASE_URL`, then `https://openrouter.ai/api/v1`. |
| `--ollama-host URL` | `run` | `OLLAMA_HOST`, then `http://localhost:11434`; native API host. |
| `--resume` | `gold`, `run` | Off; continue with matching source/configuration. |
| `--retry-failed` | `gold`, `run` | Off; requires `--resume`. |
| `--no-progress` | `gold`, `run` | Off; hide stderr progress. |
| `--dry-run` | `gold`, `run` | Off; local preview without service access or writes. |
| `--input DIR` | `report` | Required existing gold/candidate run. |
| `-h`, `--help` | All | Show command help. |

Provider/model choice is explicit in these commands. Existing
`GRAPH_RAG_PROVIDER` and `RAGAS_PROVIDER` values do not select benchmark models.
The CLI reads `.env` for endpoint/key configuration; no embedding settings are
required. The module entry point is also available before reinstalling scripts:

```sh
.venv/bin/python -m graphrag_pipeline.extraction_benchmark.command --help
uv run --locked python -m pytest tests/unit/test_extraction_benchmark.py tests/unit/test_extraction_benchmark_runtime.py -q
```

## Windows Quick Start

The initial Windows target is Windows 11 x64 with PowerShell and Python 3.14.3.
The Windows CI workflow is configured to check Python setup, unit tests, and
CLI startup. The live Neo4j Desktop, APOC/GenAI plugin, and local Ollama setup
below has not yet been verified on Windows.

Install [Git for Windows](https://git-scm.com/download/win) if needed, then
install `uv` in PowerShell. Close and reopen the terminal so `uv` is on `PATH`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository and install the locked environment from the repository
root:

```powershell
git clone https://github.com/uzair-shahed/graphrag-retriever.git project-repo
Set-Location project-repo
uv python install 3.14.3
uv sync --locked --extra dev
uv run --locked python -m pytest tests/unit -v
uv run --locked graphrag-query --help
```

The unit tests need neither Neo4j nor model credentials. Run the app commands
from the repository root; some fixture and data defaults are relative to it.

## macOS Quick Start

The verified local baseline is macOS 15 on Apple Silicon with Python 3.14.3.
Intel Macs are not yet verified. Install [Homebrew](https://brew.sh/) if needed,
then install `uv` and the locked project environment:

```sh
brew install uv
git clone https://github.com/uzair-shahed/graphrag-retriever.git project-repo
cd project-repo
uv sync --locked --extra dev
uv run --locked python -m pytest tests/unit -v
```

`uv sync` creates `.venv` and installs the exact versions recorded in `uv.lock`;
`uv run` uses that environment without a separate activation command. The unit
tests need neither Neo4j nor model credentials.

## Live GraphRAG Workflow

A live run additionally needs a Neo4j database, model-provider access, and a
Neo4j graph/vector index. The versions below reflect the locally verified
baseline, not a guarantee for every Neo4j release.

1. Install [Neo4j Desktop](https://neo4j.com/download/) and create/start a local
   Neo4j Enterprise `2026.08.1` database. In Desktop, install APOC and the
   Neo4j GenAI plugin if their procedures are not available, then restart the
   database. On Windows, use the Windows installer; this live database/plugin
   path has not yet been verified there. Verify the server and procedures using
   [the compatibility notes](agent-coordination/neo4j/compatibility.md).
2. Copy `.env.example` to `.env`. On Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set the Neo4j Bolt URI, username, and real password (replace the `changeme`
   placeholder), and keep `NEO4J_DATABASE=neo4j`, which is the database currently
   used by extraction. Install the
   [Windows version of Ollama](https://ollama.com/download/windows), start it,
   then pull the configured default models in PowerShell:

   ```powershell
   ollama pull qwen3:4b
   ollama pull nomic-embed-text
   ollama list
   ```

   Set `GRAPH_RAG_PROVIDER=ollama` in `.env`; the OpenRouter key placeholder can
   remain unchanged when Ollama is selected. The default `nomic-embed-text`
   embedding size is 768, so also set `GRAPH_RAG_VECTOR_DIMENSIONS=768`. For
   OpenRouter's default `text-embedding-3-small`, keep it at 1536 and set a valid
   `OPENROUTER_API_KEY`. Never commit `.env`. Live extraction and answering send
   text to the selected provider and may incur charges.
3. Run the small, tracked extraction fixture against the database:

   ```sh
   uv run --locked python -m graphrag_pipeline.extraction.run_fixture
   ```

   This writes graph data and makes provider calls. It does not reset the
   database.
4. In Neo4j Desktop's query tool, create the vector index using the
   [Neo4j vector-index syntax](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).
   The example matches the checked-in OpenRouter model's 1,536 dimensions and
   cosine similarity. For Ollama's default `nomic-embed-text`, change both the
   Cypher dimension below and `GRAPH_RAG_VECTOR_DIMENSIONS` in `.env` to 768.
   Keep the index dimension aligned with the embedding model used to create the
   graph's vectors. If changing embedding models for an existing graph, regenerate
   its embeddings with the selected model before creating or querying the index.

   ```cypher
   CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
   FOR (chunk:Chunk)
   ON chunk.embedding
   OPTIONS {indexConfig: {
      `vector.dimensions`: 1536,
     `vector.similarity_function`: 'cosine'
   }};

   # NOTE: The 1536 embedding should be 768 for Ollama and 1536 for OpenRouter depending on the embedding model being used.
   ```

   Wait for the index to become `ONLINE`. The pipeline verifies the base
   index; it does not create it. This is sufficient for corpus-wide queries.
   For MuSiQue queries constrained with `--record-id`, follow the record-filter
   setup below. You do not need to create a second index or manually add its
   filterable property; the setup command upgrades this index.
5. Verify the live prerequisites:

   ```sh
   uv run --locked python -m graphrag_pipeline.retrieval_prerequisites
   ```

6. Before the first `--record-id` query, ensure the base `chunk_embedding` index
exists, then run this once against the configured Neo4j database:

```sh
uv run --locked python -m graphrag_pipeline.vector_index_setup --apply
```

This command copies `record_id` from `Document` onto existing `Chunk` nodes
when needed, then drops and recreates the existing vector index with `record_id`
declared as filterable while preserving its dimensions and similarity setting.
No separate index or manual Cypher change is required. Recreating the index
temporarily makes vector search unavailable; wait for the command to finish and
verify the index is `ONLINE` before querying. Running the command again is safe
and does not recreate an index that is already configured correctly. Newly
ingested chunks receive `record_id` during ingestion, but the index still needs
the filterable-property upgrade before record-constrained queries.


7. Submit a custom question and write a local artifact:

   ```sh
   uv run --locked graphrag-query --question "What company did Aster Systems acquire in 2021?" --artifact runs/smoke.json
   ```

The query needs an existing graph and online vector index. A successful
retrieval also needs the configured embedding and answer provider. Generated
artifacts under `runs/` are local and ignored by Git.

## Full MuSiQue Data (Optional)

The official dataset is not included in a clone. See
[dataset/README.md](dataset/README.md) for acquisition instructions and the
expected location. Once downloaded, you can extract a bounded set of records
into Neo4j. Start with a dry run to validate the JSONL path and inspect how many
paragraph chunks will be processed. `--records` is required and counts records;
`--start-record` is the 1-indexed starting position (default `1`):

```sh
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --records 1 --dry-run
```

When the summary looks right, repeat the command without `--dry-run` to run
extraction and write the resulting graph data to the configured Neo4j database:

```sh
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --records 1
```

To load three records starting with the fourth record, use `--records 3` and
`--start-record 4`. Dry-run that batch first with:

```sh
uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset --dataset-path dataset/musique/data/data/musique_full_v1.0_dev.jsonl --records 3 --start-record 4 --dry-run
```

The loader also supports `--source-split LABEL` to override the provenance
split inferred from the filename and `--no-progress` to disable the live
progress bar. `--dry-run` validates and counts work without Neo4j or provider
calls. The live command reuses a Neo4j GraphRAG pipeline across paragraphs,
creating chunk embeddings and extracting entities and relationships; this
can make many provider calls and incur charges. The loader reports failed
chunks and exits nonzero if extraction failures or invalid records occurred.
After loading, create or verify the vector index as described in the live
workflow above.

Live ingestion resumes automatically. Progress is saved after each chunk in
`runs/ingestion-state/<split>-<dataset-sha256-prefix>.json`. Repeat the same
command after a stop: the loader checks Neo4j, skips matching complete chunks
without provider calls, and processes pending work. A record is loaded only
after every expected paragraph has been verified.

Each paragraph's source nodes, entities, relationships, embedding, provenance,
and completion record now commit in one Neo4j transaction. Replaying the same
input verifies and reuses that graph. A changed input under the same chunk ID
is a conflict. Older graph data without an atomic completion record is preserved
and reported as `legacy_unverified`; it requires investigation before reuse.
The loader does not automatically repair or certify legacy data.

Press Ctrl-C once to finish the current chunk and exit with code `130`; wait
for the JSON summary. A second Ctrl-C interrupts immediately, leaving the
in-flight chunk for reconciliation on resume. `--manifest-path PATH` overrides
the state file; `--fresh-manifest` moves existing state aside and starts a new
manifest while still reconciling Neo4j. A changed source checksum or split
cannot reuse the old manifest silently. Keep ingestion sequential: concurrent
writer support and a durable failed-chunk retry queue remain separate work.
See [resume and recovery details](documentation.md#resume-stop-and-local-state)
for state lifecycle, partial/conflicting data, limitations, and exit codes.

To query a record from the downloaded file, pass `--dataset-path` and
`--record-id` to `graphrag-query`; add `--preload` only when that selected
record should be ingested immediately before querying. The record ID must be
unique across the supplied dataset files. See the
[operational guide](documentation.md#musique-ingestion-and-query-command-reference)
for the complete query-command flag reference and more examples.

Before the first `--record-id` query, ensure the base `chunk_embedding` index
exists, then run this once against the configured Neo4j database:

```sh
uv run --locked python -m graphrag_pipeline.vector_index_setup --apply
```

This command copies `record_id` from `Document` onto existing `Chunk` nodes
when needed, then drops and recreates the existing vector index with `record_id`
declared as filterable while preserving its dimensions and similarity setting.
No separate index or manual Cypher change is required. Recreating the index
temporarily makes vector search unavailable; wait for the command to finish and
verify the index is `ONLINE` before querying. Running the command again is safe
and does not recreate an index that is already configured correctly. Newly
ingested chunks receive `record_id` during ingestion, but the index still needs
the filterable-property upgrade before record-constrained queries.

## Further Documentation

- [Operational details and command behavior](documentation.md)
- [Neo4j compatibility and local verification](agent-coordination/neo4j/compatibility.md)
- [Data contracts](agent-coordination/data-contracts.md)
- [MuSiQue dataset description and license](agent-coordination/musique-dataset-description.md)

## License

The application source does not yet declare a project license. Choose and add
one before public distribution. The MuSiQue dataset's CC BY 4.0 license applies
to that dataset, not automatically to this project's source code.
