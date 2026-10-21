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

The benchmark compares extracted entities and relationships with saved gold using
precision, recall and F1. Its v2 response schema contains mention IDs, entity
names/types and directed relationship references/types. It requests no confidence
or evidence spans. Paragraph text, identity and provenance are retained locally.
The benchmark accesses no Neo4j or RAGAS services.

The [complete benchmark reference](agent-coordination/entity-extraction-benchmark/entity-extraction-benchmark.md)
covers migration, providers, scoring, resume, append and artifacts.

### Reuse existing gold and candidate responses

Let active runs finish, then convert their saved generations offline:

```sh
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/gold-run-150 --output runs/entity-extraction/gold-run-180-v2 --dry-run
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/gold-run-150 --output runs/entity-extraction/gold-run-180-v2
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/qwen3-4b-instruct-aligned --gold runs/entity-extraction/gold-run-180-v2 --output runs/entity-extraction/qwen3-4b-instruct-v2
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-run-180-v2 --model ollama:qwen3:4b-instruct --ollama-think false --output runs/entity-extraction/qwen3-4b-instruct-v2 --resume
```

Conversion makes zero provider calls. It preserves originals and raw responses,
recovers outputs rejected only for retired fields, and recalculates references
and scores. All 180 saved gold responses pass structural validation after
projection. Invalid references, malformed/incomplete or missing responses remain
failures. The six previously skipped candidate paragraphs become pending; resume
generates missing predictions only. Add --retry-failed to explicitly retry failures.
Migration creates a checksummed audit with original settings/errors and preserves
attempts, timing and usage/cost metadata. Generation-format counts distinguish
legacy responses from new simplified generations. The old --evidence-offsets
option has been removed. Existing v1 directories require migration before use.

### New runs and reports

```sh
uv run --locked graphrag-extraction-benchmark --help
uv run --locked graphrag-extraction-benchmark select --paragraphs 150
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate
uv run --locked graphrag-extraction-benchmark report --input runs/entity-extraction/local-candidate
```

Gold requires OPENROUTER_API_KEY; local Ollama candidates require the model to be
available in Ollama. Process environment overrides .env. Provider/model choices
and budgets are explicit. Source/model/prompt/schema settings must match on
resume. Saved successful rows are never regenerated by resume or retry.
Append supports additional-paragraphs or a contiguous inclusive range; extend
gold first. Summaries report gold skips and pending/failed work separately.

### Visualize model and reasoning comparisons

Create an offline HTML report, four standalone SVG charts, and a machine-readable
`comparison.json` from your saved format-v2 artifacts:

```powershell
uv run --locked graphrag-extraction-benchmark visualize --gold runs/entity-extraction/gold-run-180-v2 --run "Qwen-off=runs/entity-extraction/qwen3-4b-instruct-v2" --output runs/entity-extraction-reports/gpt-sol-qwen
```

Open the output directory's `index.html` in a browser. Repeat `--run "LABEL=PATH"`
for any number of candidate directories; all models within each directory are
included separately. For example, after generating a separate reasoning-on run:

```powershell
uv run --locked graphrag-extraction-benchmark visualize --gold runs/entity-extraction/gold-run-180-v2 --run "Qwen-off=runs/entity-extraction/qwen3-4b-instruct-v2" --run "Qwen-on=runs/entity-extraction/qwen3-4b-instruct-thinking" --output runs/entity-extraction-reports/qwen-reasoning
```

The report shows entity and directed-relation micro precision/recall/F1, exact
TP/FP/FN totals, and paragraph extraction outcomes. Each completed reasoning-on
run is compared with every completed reasoning-off run of the same provider/model;
F1 changes are measured in percentage points. Repeated runs are not averaged.
Reasoning settings come from manifests, regardless of the labels you supply:
Ollama `think=false` and OpenRouter effort `none` mean off; enabled settings show
their level. Omitted settings remain unspecified and are excluded from on/off
pairs. Other configuration differences are listed alongside the comparisons.

Every candidate must match the supplied gold's full cohort, fingerprint, frozen
labels and gold skips. Legacy format-v1 directories must be migrated first.
Metrics are recomputed from validated saved rows, without using `summary.json`,
calling providers, or modifying input artifacts. Failed extractions count as
empty predictions; pending paragraphs and skipped gold are excluded and clearly
reported. Partial runs are charted but excluded from reasoning deltas. Migrated
generation counts are shown so you can distinguish legacy and simplified output
formats when interpreting experiments.

The HTML works offline and embeds its charts and SVG downloads. The four SVG
files can also be used directly in documents or slides. Keep output directories
separate from inputs. Repeating the command replaces only files owned by a prior
visualizer report; unrelated output files are preserved and filename collisions
are rejected. A successful visualization exits 0 even when candidates have saved
failures or pending work; invalid/incompatible artifacts exit 2.

### Flags

| Flag | Commands | Meaning |
| --- | --- | --- |
| `--dataset-path` | `select`, `gold` | Local MuSiQue JSONL input; defaults to the dev dataset. |
| `--paragraphs` | `select`, `gold` | Positive exact prefix count; default 150. |
| `--additional-paragraphs` | `gold`, `run` | Add the next N source positions to an existing fully attempted cohort. |
| `--start-paragraph`, `--end-paragraph` | `gold`, `run` | Inclusive 1-based append range; start must follow the saved cohort. |
| `--model` | `gold`, `run` | Required; gold uses a bare OpenRouter ID, candidates repeat provider:model. |
| `--reasoning-effort` | `gold`, `run` | Required enabled effort for gold; optional for hosted candidates, also accepting none. |
| `--ollama-think` | `run` | default, true, false, or a model-supported string. |
| `--gold` | `run`, `migrate` | Gold directory; migration candidates require already converted v2 gold. |
| `--output` | `gold`, `run`, `migrate` | Destination directory; migration always requires a separate new/empty directory. |
| `--max-tokens` | `gold`, `run` | Positive output budget; default 16384. |
| `--timeout` | `gold`, `run` | Positive finite HTTP timeout in seconds; default 180. |
| `--retries` | `gold`, `run` | Additional transient HTTP/transport retries; default 2. |
| `--openrouter-base-url` | `gold`, `run` | Override OPENROUTER_BASE_URL; default https://openrouter.ai/api/v1. |
| `--ollama-host` | `run` | Override OLLAMA_HOST; default http://localhost:11434. |
| `--resume` | `gold`, `run` | Continue compatible saved work, skipping completed attempts. |
| `--retry-failed` | `gold`, `run` | With --resume, repeat failed attempts; successful rows stay skipped. |
| `--no-progress` | `gold`, `run` | Hide stderr progress while keeping final JSON stdout. |
| `--dry-run` | `gold`, `run`, `migrate` | Preview local validation without provider calls or artifact writes. |
| `--input` | `report`, `migrate` | Saved run directory. |
| `--run` | `visualize` | Repeat `LABEL=PATH` for each candidate run directory; labels and paths must be unique. |
| `--gold` | `visualize` | Matching format-v2 gold directory for all candidate runs. |
| `--output` | `visualize` | Separate HTML/SVG report directory; generated files can be refreshed safely. |
| `-h`, `--help` | All | Command help. |

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
