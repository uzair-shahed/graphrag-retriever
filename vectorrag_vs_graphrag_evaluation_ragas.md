# VectorRAG vs. GraphRAG evaluation: CLI reference

Updated: 20 October 2026.

`graphrag-eval` prepares MuSiQue records, checks an existing Neo4j graph, and
runs paired VectorRAG/GraphRAG evaluation with RAGAS. Every evaluation compares
both backends on the selected questions; it is not necessary to run each
backend separately. RAGAS supplies the RAG Triad scores, and MuSiQue gold labels
supply answer, supporting-paragraph and decomposition-hop checks.

Currently supported source:
`dataset/musique/data/data/musique_ans_v1.0_dev.jsonl`.
The file contains 2,417 records, all currently marked answerable. The evaluation
CLI does not use `GRAPH_RAG_DATASET_PATHS` or accept the full/test splits.

## Contents

- [Installation and invocation](#installation-and-invocation)
- [Choose a command](#choose-a-command)
- [Shared options and selection rules](#shared-options-and-selection-rules)
- [`prepare` reference](#prepare-reference)
- [`check` reference](#check-reference)
- [`run` reference](#run-reference)
- [`report` reference](#report-reference)
- [Examples by use case](#examples-by-use-case)
- [Provider configuration and precedence](#provider-configuration-and-precedence)
- [Graph prerequisites](#graph-prerequisites)
- [Scores and output](#scores-and-output)
- [Exit codes and recovery](#exit-codes-and-recovery)
- [Implementation and validation history](#implementation-and-validation-history)
- [Standalone paragraph extraction benchmark](#standalone-paragraph-extraction-benchmark)

## Installation and invocation

Run commands from the repository root after installing the locked environment:

```sh
uv sync --locked --extra dev
uv run --locked graphrag-eval --help
```

General syntax:

```text
graphrag-eval {prepare,check,run,report} [OPTIONS]
```

All examples use `uv run --locked graphrag-eval`. In an activated project
environment, `graphrag-eval` is equivalent. The module entry point is also
available:

```sh
uv run --locked python -m graphrag_pipeline.evaluation.command run --help
```

The evaluation CLI examples use single lines compatible with POSIX shells and
PowerShell. Historical validation commands use POSIX environment-variable
assignments. Paths are relative to the repository root unless absolute paths are supplied.
Model and database settings are read from the environment and local `.env`.

Dependencies pin `ragas==0.4.3` and `langchain-community==0.4.1`; the latter
avoids a RAGAS import failure with 0.4.2. A token-budgeted run may download the
`tiktoken` `cl100k_base` table on its first use. That download is not a model call.

## Choose a command

| Command | Purpose | Neo4j access | Model calls | Result |
| --- | --- | --- | --- | --- |
| `prepare` | Ingest the selected dataset records, or reuse matching completed chunks. | Reads and may write graph data and ingestion checkpoints. | Extraction and embedding calls for missing chunks, using `GRAPH_RAG_PROVIDER`. | JSON ingestion summary on stdout and resumable ingestion state. |
| `check` | Verify that the selected records and retrieval prerequisites are ready. | Read-only checks of selected chunks, provenance, completion inventory and index configuration. | None; configured model settings are read, but model services are not contacted. | JSON selection/preflight/prerequisite report on stdout. |
| `run` | Retrieve evidence, generate answers for both backends, and score them with RAGAS and MuSiQue gold labels. | Read-only preflight, retrieval and graph fingerprints. | Retrieval embeddings, answers and RAGAS judging/embeddings. | JSON summary on stdout and files in an evaluation output directory. |
| `report` | Generate offline HTML visualizations and record comparisons from an existing run. | None. | None. | `report.html` in the supplied directory; JSON path, row count, and warnings on stdout. |
| `prepare`, `check`, or `run` with `--dry-run` | Validate and preview the dataset selection and planned paragraph count. | None. | None. | JSON selection and chunk count on stdout; no preparation or evaluation artifacts. |

Use `check` then `run` directly when the selected records are already ingested.
Use `prepare` explicitly when records are missing. `check` and `run` never
perform ingestion, create an index, or repair conflicting data. A successful
`check` establishes graph readiness; it does not test answer/evaluator service
availability or generate RAGAS scores.

## Shared options and selection rules

These options are accepted by **`prepare`, `check`, and `run`**. None is required.
`report` has its own artifact-only options below.

| Option | Type / default | Description |
| --- | --- | --- |
| `--dataset-path PATH` | Path; `dataset/musique/data/data/musique_ans_v1.0_dev.jsonl` | Read this dev JSONL file. A relocated copy must retain the filename `musique_ans_v1.0_dev.jsonl`. Supply this option once. |
| `--record-id ID` | String; unset; repeatable | Select exact record IDs in the supplied order. IDs must exist in the file and must not be repeated. |
| `--records N` | Positive integer; no count limit | Select N consecutive records beginning at `--start-record`. Counts records, not paragraph chunks. |
| `--max-records N` | Positive integer; unset | Alias for `--records N`, matching query-command terminology. Use one count flag, not both. |
| `--start-record S` | Positive integer; effective default `1` | Start at the 1-indexed, inclusive nonblank-record position S. With no count limit, select from S through EOF. |
| `--dry-run` | Switch; off | Preview selection and paragraph counts only. It does not check graph readiness, provider availability or retrieval execution. |
| `-h`, `--help` | Switch | Print help for the current command and exit. |

Selection rules:

- With no selectors, use **every record** in the configured dev file.
- `--start-record 4 --records 3` selects records **4, 5 and 6**, exactly as the ingestion command does.
- Positions count nonblank records; reported `source_line` retains the actual JSONL line number.
- A range reaching EOF returns the available remaining records. A range starting beyond EOF fails because the selection is empty.
- `--record-id` cannot be combined with `--records`, `--max-records`, or an explicitly supplied `--start-record`, even `--start-record 1`.
- Unknown IDs, duplicate IDs, duplicate source IDs, empty selections and nonpositive bounds produce an error.
- Use the same selector for `prepare`, `check` and `run` when working with the same cohort.

## `prepare` reference

**Purpose:** ensure that every paragraph of each selected record has been
ingested through the existing resumable ingestion path. Preparation uses
`GRAPH_RAG_PROVIDER` for extraction and embeddings; answer/evaluator flags
belong to `run` and are not accepted here.

```text
graphrag-eval prepare [SHARED_OPTIONS] [--manifest-path PATH] [--no-progress]
```

In addition to the [shared options](#shared-options-and-selection-rules):

| Option | Type / default | Description |
| --- | --- | --- |
| `--manifest-path PATH` | Path; automatic ingestion-state path | Override the resumable ingestion manifest. Normally state is stored under `runs/ingestion-state/<split>-<checksum-prefix>.json`. Reuse the same source identity when resuming. |
| `--no-progress` | Switch; off | Hide the preparation progress bar. The final JSON summary is still printed. |

Example:

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3
```

This prepares all paragraph chunks belonging to records 4–6. Matching verified
chunks are reused without extraction or embedding calls; missing chunks invoke
the configured ingestion provider. Partial, legacy or conflicting data is
reported for investigation and is preserved. Repeating the command reconciles
Neo4j and resumes pending work. Preparation does not create or upgrade the
retrieval vector index.

**Output:** ingestion JSON includes selected-record information, planned
chunks, successful/failed/skipped chunks, investigation cases, record-completion
counts and manifest paths. A record is complete only after every expected
paragraph has been verified.

## `check` reference

**Purpose:** validate existing graph data for the chosen questions and retrieval
conditions before generating answers or RAGAS scores.

```text
graphrag-eval check [SHARED_OPTIONS] [--condition {record,corpus} ...]
```

In addition to the [shared options](#shared-options-and-selection-rules):

| Option | Type / default | Description |
| --- | --- | --- |
| `--condition {record,corpus}` | Choice; repeatable; both when omitted | Check prerequisites for the condition(s) that will be used by `run`. `record` also requires index-supported `record_id` filtering. Each condition may appear once. |

Example:

```sh
uv run --locked graphrag-eval check --start-record 4 --records 3 --condition record
```

This checks records 4–6 for record-scoped evaluation. It verifies all selected
paragraphs against their source text, provenance and atomic completion
inventory, then checks Neo4j/retrieval/index compatibility. Only the selected
records must be ingested. Missing or conflicting records are reported before
answer or judge clients are created.

**Output:** successful JSON contains `records_selected`, selected IDs/source
lines, `preflight.ready: true`, and `prerequisites`. If records are missing,
`preflight.ready` is false and `preflight.issues` identifies the affected chunks;
`missing_record_ids` lists records with absent chunks. Setup errors instead
return a diagnostic JSON object. `check` writes no evaluation output directory.

Use the same `--condition` choices on `check` and `run`. For example, a
corpus-only check does not establish that record filtering is configured for a
subsequent run that defaults to both conditions.

## `run` reference

**Purpose:** perform a fresh paired retrieval, answer generation and scoring run
on the selected, already-ingested records. Preflight is built in, so a separate
`check` invocation is useful but optional. `run` does not rescore previously
saved answer artifacts.

```text
graphrag-eval run [SHARED_OPTIONS] [CONDITION_OPTIONS] [RETRIEVAL_OPTIONS] [PROVIDER_OPTIONS] [--output PATH] [--no-progress]
```

In addition to the [shared options](#shared-options-and-selection-rules), the
complete set of `run` options is:

| Option | Type / default | Description |
| --- | --- | --- |
| `--condition {record,corpus}` | Choice; repeatable; both when omitted | `record` filters retrieval to the question's record. `corpus` searches the entire existing graph/index. Repeating the flag with different values runs both and reports them separately. |
| `--output PATH` | Directory; generated `runs/evaluation/<UTC-timestamp>-<unique-suffix>` | Save evaluation artifacts here. A supplied directory must be new or empty. Reuse of a populated directory fails. |
| `--no-progress` | Switch; off | Hide the default tqdm evaluation bar on stderr. The final JSON summary still prints to stdout. |
| `--top-k N` | Positive integer; `5` | Request N vector seed chunks for each backend, with the same embedded question and no overfetching. |
| `--max-context-chunks N` | Positive integer; `5` | Maximum whole evidence chunks retained for answering after retrieval. Does not change `--top-k`. |
| `--max-context-tokens N` | Positive integer; `4096` | Maximum rendered evidence-context size using the shared `cl100k_base` tokenizer proxy. Fixed instructions and question tokens are additional. |
| `--max-graph-nodes N` | Positive integer; `200` | Graph traversal resource budget per seed. This is not a hop-depth limit. Budget exhaustion is recorded as an incomplete component. |
| `--answer-provider {ollama,openrouter}` | Choice; `GRAPH_RAG_PROVIDER` | Select the answer provider for both backends. Does not switch retrieval embeddings or change the index. |
| `--answer-model MODEL` | String; selected answer provider's LLM model | Override the answer model for both backends. Use a model identifier valid for the selected answer provider. |
| `--eval-provider {ollama,openrouter}` | Choice; `RAGAS_PROVIDER`, then `GRAPH_RAG_PROVIDER` | Select the RAGAS judge provider independently of the answer provider. |
| `--eval-model MODEL` | String; `RAGAS_LLM_MODEL`, then selected evaluator provider's LLM model | Override the RAGAS judge model. |
| `--eval-embedding-model MODEL` | String; `RAGAS_EMBEDDING_MODEL`, then selected evaluator provider's embedding model | Override evaluator embeddings used by answer relevance. Does not alter ingestion/retrieval embeddings. |

Example:

```sh
uv run --locked graphrag-eval run --start-record 4 --records 3 --answer-provider ollama --eval-provider ollama --output runs/evaluation-4-to-6
```

This evaluates records 4–6 with Ollama answers and judging. Since `--condition`
is omitted, both `record` and `corpus` are evaluated for both VectorRAG and
GraphRAG: **3 questions × 2 conditions × 2 backends = 12 result rows**.
Retrieval embeddings still use `GRAPH_RAG_PROVIDER`; set that configuration to
match the model used to build the graph vectors.

The progress bar counts completed result rows, so this example runs from 0/12
to 12/12. Each increment follows scoring, flushing the row to `records.jsonl`,
and updating `summary.json`. Failed answers, abstentions, and metric errors
also count as completed results once saved. The bar shows the active record
position, retrieval condition, backend, and answering/scoring phase. It starts
after preflight and provider setup. A model call can take time without moving
the counter; the bar does not estimate progress inside individual model calls.
On interruption, the bar closes and completed artifacts remain available.

To suppress the bar for redirected output or automation:

```sh
uv run --locked graphrag-eval run --start-record 4 --records 3 --output runs/eval-quiet-4-to-6 --no-progress
```

This performs the same evaluation with only the final JSON summary on stdout
(provider diagnostics may still appear on stderr).

For each question, both backends receive the same original question embedding
and vector seed count. GraphRAG expands each seed's associated entity component;
VectorRAG keeps its vector-only evidence. Answering uses the shared prompt
policy, generation settings and citation validator. The benchmark records full
retrieval and the context selected for answering separately. There is no
`--retrieval-backend` option on this command because both backends always run.

**Output:** a JSON summary on stdout and `preflight.json`, `manifest.json`,
`records.jsonl`, `summary.json`, and `report.html` under the output directory. Details are in
[Scores and output](#scores-and-output). The default full-file run produces
**2,417 × 2 × 2 = 9,668 rows** when every selected query completes.

## `report` reference

Every started `run` automatically builds a self-contained `report.html` after
saving final JSON artifacts. The `report` command generates or regenerates this
file for a past or interrupted run without re-running evaluation.

```text
graphrag-eval report --input PATH
```

| Option | Type / default | Description |
| --- | --- | --- |
| `--input PATH` | Directory; required | Existing evaluation directory containing `manifest.json` and `records.jsonl`. `summary.json` is optional. Atomically replaces only `PATH/report.html`. |
| `-h`, `--help` | Switch | Show help. |

```sh
uv run --locked graphrag-eval report --input runs/eval-prepared-4-to-6-part-3
```

**Output:** JSON containing `report`, `completed_rows`, and `warnings`. This
command does not load the dataset, connect to Neo4j, or call answer/judge models.
It accepts no selection, provider, or `--dry-run` options. A successful command
can describe a partial or failed evaluation; inspect the report's run status.

### Viewing and exporting

Open `PATH/report.html` directly in a browser with JavaScript enabled. No server,
CDN, external assets, or network access is required. The overview includes:

- Run status, completion counts, graph consistency, and expandable configuration.
- Separate record-scoped and corpus-wide panels comparing Triad scores, gold
  answer accuracy/F1, supporting-paragraph and hop coverage, answer outcomes,
  context tokens, and pipeline latency excluding RAGAS judging.
- Tables with exact displayed chart values, scored-case counts, metric failures,
  incomplete traversals, and paired differences.

Search records by ID or question and choose a retrieval condition. The two
backends appear side by side with the gold answer/aliases, answerability,
supporting paragraphs, decomposition hops, finalized answers, citations, metric
outcomes, full retrieved chunks and graph evidence, and exact bounded answer
context/prompt. Search affects record navigation only; overview charts continue
to summarize every saved result. Expand long contexts and diagnostics as needed.

| Presentation action | Result |
| --- | --- |
| **Download SVG** on a chart | Standalone vector image with title, condition, legend, values, and sample counts; suitable for scaling in slides. |
| **Download PNG** on a chart | Raster image at twice the chart's base resolution, with a white background. |
| **Print overview / Save PDF** | Run overview and all condition charts, without record details. Select Save as PDF in the browser print dialog. |
| **Print selected record / Save PDF** | Selected question and condition, both backends, expanded gold/context details. |
| **Print entire report / Save PDF** | Overview and every selected record/condition, with expanded context. Large runs may take time and produce many pages. |

Gold labels and evidence are deduplicated within each record. Details are parsed
on demand. A full-dev report can still be large because the exact saved evidence
is embedded. Browser PDF printing is supported; automatic PDF generation and
PowerPoint files are not provided.

### Validation and recovery

Charts preserve existing scoring rules: answer metrics use answered cases only,
abstentions/failures remain separate, N/A never becomes zero, and conditions are
never pooled. Tables and charts are recomputed from complete saved rows; missing
or inconsistent `summary.json` produces a visible warning. Source JSON is not
modified.

Artifact checks reject unknown schema versions, malformed complete rows,
duplicate results, inconsistent labels/questions, and evidence/citation
mismatches. An incomplete final JSONL line may be ignored with a warning.
Missing results and absent labels for questions that never started are explicit.
Partial reports are attempted after interruption. Hard process termination or
storage errors may prevent automatic reporting; rerun `report` after recovery.
Report failures preserve evaluation artifacts and the original evaluation
exception; otherwise they cause a nonzero exit. A failed regeneration keeps the
previous HTML file intact.

The report adds no Python runtime dependencies. Optional browser validation:

```sh
uv sync --locked --extra dev --extra browser-tests
uv run --locked --extra browser-tests python -m playwright install chromium
RUN_REPORT_BROWSER_TESTS=1 uv run --locked --extra browser-tests python -m pytest tests/browser -q -s
```

For PowerShell, set `$env:RUN_REPORT_BROWSER_TESTS="1"` before the final command
and omit its POSIX environment-variable prefix. These checks use synthetic
artifacts, local Chromium, and no model/Neo4j calls.

## Examples by use case

Each example assumes the configured Neo4j database/index and selected provider
models are available, except examples explicitly marked as selection previews.
Use a new output directory for each run.

### Preview a range without contacting services

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3 --dry-run
```

Prints the IDs/source lines for records 4–6 and the total paragraph chunk count.
It does not ingest data, contact Neo4j/models, or establish graph readiness.
`check --dry-run` and `run --dry-run` perform the same selection-only preview.

### Evaluate already-ingested records directly

```sh
uv run --locked graphrag-eval check --start-record 4 --records 3
uv run --locked graphrag-eval run --start-record 4 --records 3 --output runs/eval-existing-4-to-6
```

The first command validates the existing records and both retrieval conditions.
The second generates fresh paired answers and scores using the configured
providers. Neither command performs ingestion; the rest of the dataset need
not be loaded.

### Prepare missing records, then evaluate them

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3
uv run --locked graphrag-eval check --start-record 4 --records 3
uv run --locked graphrag-eval run --start-record 4 --records 3 --output runs/eval-prepared-4-to-6
```

Preparation loads missing paragraphs and reuses verified work. The check
confirms graph/index readiness; the run writes the paired evaluation. Inspect
reported conflicts before proceeding rather than treating them as missing data.

### Select arbitrary record IDs

```sh
uv run --locked graphrag-eval run --record-id 2hop__460946_294723 --record-id 2hop__259228_793698 --condition record --output runs/eval-two-ids
```

Evaluates those two IDs, in the supplied order, with both backends under the
record condition only. These are actual IDs in the current dev file. Both
records must already be ingested. This produces four rows; it does not select
all records between their file positions.

### Evaluate only the first N records

```sh
uv run --locked graphrag-eval run --records 10 --condition record --output runs/eval-first-10
```

Evaluates the first ten records with record filtering and both backends,
producing 20 rows. `--max-records 10` is equivalent to `--records 10`.

### Evaluate from a starting position to EOF

```sh
uv run --locked graphrag-eval run --start-record 2400 --condition record --output runs/eval-from-2400
```

With no count limit, selects records 2400–2417 of the current file. Each must be
ready in Neo4j. Future file lengths are determined from the input, not hardcoded.

### Compare corpus-wide retrieval only

```sh
uv run --locked graphrag-eval check --records 10 --condition corpus
uv run --locked graphrag-eval run --records 10 --condition corpus --output runs/eval-corpus-first-10
```

Uses the first ten records as questions, but searches the **entire existing
index**, including graph data outside those ten records. It produces 20 rows.
Selecting a question subset does not create a restricted retrieval corpus.
To request both conditions explicitly, repeat
`--condition record --condition corpus`.

### Use Ollama for both answers and RAGAS

```sh
uv run --locked graphrag-eval run --records 1 --answer-provider ollama --answer-model qwen3:4b-instruct --eval-provider ollama --eval-model qwen3:4b-instruct --eval-embedding-model nomic-embed-text --output runs/eval-ollama
```

Uses the named installed Ollama models for answers, judging and evaluator
embeddings. The selected model identifiers are examples; install them in the
configured Ollama service first. Retrieval embeddings still follow
`GRAPH_RAG_PROVIDER` and must match the graph/index.

### Use OpenRouter answers with Ollama judging

```sh
uv run --locked graphrag-eval run --records 1 --answer-provider openrouter --answer-model openai/gpt-4o-mini --eval-provider ollama --eval-model qwen3:4b-instruct --eval-embedding-model nomic-embed-text --output runs/eval-openrouter-answer
```

Both backends answer through OpenRouter; RAGAS uses Ollama. Configure the
OpenRouter API key/settings and the local judge models first. This command
makes OpenRouter answer calls when executed; it is a usage example and was not
executed during documentation validation.

### Use Ollama answers with OpenRouter judging

```sh
uv run --locked graphrag-eval run --records 1 --answer-provider ollama --eval-provider openrouter --eval-model openai/gpt-4o-mini --eval-embedding-model openai/text-embedding-3-small --output runs/eval-openrouter-judge
```

Answers use Ollama while RAGAS judge and evaluator-embedding calls use
OpenRouter. To use OpenRouter for both roles, set both provider flags to
`openrouter`. Model names are examples; availability and credentials must be
configured for the chosen service. This example was not executed during
documentation validation.

### Change retrieval and context budgets

```sh
uv run --locked graphrag-eval run --records 5 --top-k 8 --max-context-chunks 4 --max-context-tokens 2048 --max-graph-nodes 300 --output runs/eval-budgeted
```

Retrieves eight vector seeds for each backend. Answering retains a ranked
prefix of at most four whole evidence blocks within the 2,048 proxy-token
context budget. Graph expansion uses the per-seed 300-node resource budget.
The full retrieval remains in artifacts even when answer-context selection
keeps fewer chunks. An oversized first block can leave no answer context and
cause abstention without an answer-model call.

### Resume preparation using a custom manifest

```sh
uv run --locked graphrag-eval prepare --start-record 4 --records 3 --manifest-path runs/ingestion-state/eval-4-to-6.json --no-progress
```

Uses the specified manifest and suppresses its progress bar. Repeat the same
command after a stop to reconcile and reuse completed chunks. This resumes
preparation, not evaluation. `run` has no automatic resume option.

### Use the dataset through an explicit path

```sh
uv run --locked graphrag-eval check --dataset-path dataset/musique/data/data/musique_ans_v1.0_dev.jsonl --records 1
```

Selects the first record from that path. Replace it with an absolute path if
needed, retaining the required filename. For already-ingested data, preserve
the source path spelling used during ingestion because it forms part of the
stored source identity.

### Evaluate the full dev file

```sh
uv run --locked graphrag-eval check
uv run --locked graphrag-eval run --output runs/eval-full-dev
```

Checks and evaluates every record; both conditions and both backends are the
default. All selected records must already be ready. The run does not trigger
preparation if any are missing.

## Provider configuration and precedence

CLI options override the corresponding environment defaults. `.env.example`
contains the configurable provider settings. The three roles are independent:

| Role | Provider selection | Model / endpoint configuration |
| --- | --- | --- |
| Preparation and retrieval embeddings | `GRAPH_RAG_PROVIDER`; defaults to `openrouter` if unset | Selected provider's existing `OLLAMA_*` or `OPENROUTER_*` settings. There is no retrieval-provider CLI override. |
| Answering, shared by both backends | `--answer-provider`, otherwise `GRAPH_RAG_PROVIDER` | `--answer-model`, otherwise the selected provider's configured LLM. |
| RAGAS judging and evaluator embeddings | `--eval-provider`, otherwise `RAGAS_PROVIDER`, otherwise `GRAPH_RAG_PROVIDER` | CLI evaluator-model overrides, then `RAGAS_*` model overrides, then selected provider's model settings. |

Changing only `--answer-provider` or `--eval-provider` leaves retrieval embeddings
unchanged. Changing the ingestion/retrieval embedding model requires matching
stored graph embeddings and vector-index dimensions.

Evaluator settings:

| Environment variable | Default / behavior |
| --- | --- |
| `RAGAS_PROVIDER` | Inherits `GRAPH_RAG_PROVIDER` when unset. Allowed values: `ollama`, `openrouter`. The checked-in `.env.example` explicitly selects `ollama`. |
| `RAGAS_LLM_MODEL` | Selected evaluator provider's LLM when unset or empty; overridden by `--eval-model`. |
| `RAGAS_EMBEDDING_MODEL` | Selected evaluator provider's embedding model when unset or empty; overridden by `--eval-embedding-model`. |
| `RAGAS_BASE_URL` | Selected provider's endpoint when unset or empty: `OLLAMA_HOST` plus `/v1`, or `OPENROUTER_BASE_URL`. A nonempty override remains in effect even when changing `--eval-provider`; keep it aligned with that provider. |
| `RAGAS_MAX_TOKENS` | `4096`; positive judge output-token limit. |
| `RAGAS_TIMEOUT_SECONDS` | `180`; positive timeout for HTTP requests and each complete metric evaluation. |
| `RAGAS_RETRIES` | `2`; nonnegative retry allowance for provider/structured-output handling. |

Ollama settings are `OLLAMA_HOST`, `OLLAMA_LLM_MODEL`,
`OLLAMA_EMBEDDING_MODEL` and `OLLAMA_LLM_MAX_TOKENS` (answer output limit,
default 4096). OpenRouter uses `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`,
`OPENROUTER_LLM_MODEL`, `OPENROUTER_EMBEDDING_MODEL`,
`OPENROUTER_EMBEDDING_DIMENSIONS` and `OPENROUTER_LLM_MAX_TOKENS`, plus its
configured parameter/fallback policy. Neo4j connection and index settings are
listed in `.env.example` and the [operational documentation](documentation.md).

Both answer providers use temperature zero. RAGAS telemetry is disabled before
metric imports. OpenRouter evaluator routing requires requested parameters and
disables provider fallbacks. Provider credentials are excluded from saved
configuration.

Ollama structured extraction and answering explicitly send `think: false`.
The RAGAS Ollama judge uses the OpenAI-compatible `reasoning_effort: "none"`
control and sends the metric's actual JSON Schema through `response_format`.
This keeps thinking disabled for the configured model and constrains judge
output to the metric's fields; JSON-object syntax alone does not enforce those
fields. Answer requests also honor `OLLAMA_REQUEST_TIMEOUT_SECONDS`, as native
extraction and embedding requests do. The default remains 120 seconds; evaluator
metric/request bounds continue to use `RAGAS_TIMEOUT_SECONDS`.

The context budget uses `cl100k_base` as a shared comparison proxy, rather than
each answer model's native tokenizer. It counts rendered evidence including
graph lines; the question and fixed instructions consume additional tokens.

## Graph prerequisites

Both `check` and `run` require the selected records' paragraphs, source
provenance and intact atomic completion inventory. Source identity includes
path spelling, checksum, source line and `ans_dev` split. Identical record IDs
from a different dataset file or split do not establish a valid match.
Legacy/partial/conflicting graph data remains unchanged for investigation.

The vector index must already exist, be online, and match the ingestion
embedding dimensions/model. The record condition additionally requires index
support for `record_id` filtering. Where appropriate, use the existing explicit
index-upgrade command after creating the base index as described in the
[README live workflow](README.md#live-graphrag-workflow):

```sh
uv run --locked python -m graphrag_pipeline.vector_index_setup --apply
```

Preparation does not create or upgrade the index. Index setup can recreate an
existing index, so run it as a separate operation before evaluation.

Keep the database quiet during a run. The manifest fingerprints node and
relationship properties and index definitions before and after evaluation.
Persistent changes mark the run inconsistent. These checks do not supply
transaction snapshot isolation across searches or detect every temporary
change that is subsequently reverted.

## Scores and output

The default conditions are `record` and `corpus`; choose one with
`--condition record` or `--condition corpus`. Corpus-wide retrieval searches
the entire existing graph/index, not just the selected question records. The
graph snapshot records which data was present. Results remain separate.

RAGAS Triad scores use `Faithfulness`, `AnswerRelevancy`, and `ContextRelevance`
from the pinned release. The exact rendered answer contexts, including graph
structure, are supplied for faithfulness/context scoring. Full retrieval
context relevance is also retained when selection changes the answer context.
RAGAS Triad metrics are reference-free: MuSiQue gold is evaluated alongside
them, without leaking labels into either retrieval or answering.

Gold checks include normalized exact answer/alias match and token F1, support
recall/precision, per-hop support coverage, answerability accuracy, and citation
gold-support precision/recall. A support match requires both the stable chunk
ID and unchanged paragraph text. Hop coverage measures retrieved source
coverage; it does not prove that the model followed a particular reasoning
chain. Citation gold support is a paragraph-label proxy, not a semantic proof
that a cited sentence entails an answer.

Answer metrics are scored for successful answers. Context relevance is scored
where context exists, including failed/abstained answer runs. Undefined values
are null with explicit status, not fabricated zeros. Judge exceptions/timeouts
remain per-metric errors; other metrics and query rows continue. Reports include
score denominators, error counts, abstention and failure rates. Gold answer
accuracy uses all completed query rows, counting failures/wrong abstentions
as incorrect; future `answerable:false` records require explicit abstention.
Paired deltas use only matching seeds and scores available for both backends.
Traversal-incomplete counts remain visible even when scoring succeeds.

Current graph components do not join separate source paragraphs. Consequently,
graph expansion may add entities/relationships but no additional gold source
paragraphs; the report preserves this result and does not claim graph-level
factual correctness from paragraph labels.

Each output directory contains:

- `preflight.json`: selected IDs/source lines and graph/index checks.
- `manifest.json`: configuration without credentials, versions, dataset checksum,
  selection, run status and graph fingerprints.
- `records.jsonl`: durable completed rows with trace/pair IDs, labels, full
  retrieval, exact answer context/prompt, response/citations, scores and timings.
- `summary.json`: per-condition/backend metrics, denominators and paired deltas.
- `report.html`: standalone visual summary and searchable record comparisons,
  with PNG/SVG chart downloads and browser PDF print layouts.

During evaluation, context/label payloads stay in JSONL rather than accumulating
in memory. Report generation reads the saved rows after evaluation and embeds
deduplicated contexts and labels in HTML for offline viewing.
Token usage is stored when the answer wrapper exposes it. Provider price is
null when unavailable; no price is invented. Existing artifact directories are
not overwritten. Interrupted runs retain completed rows; rerun a remaining
subset into a new directory (automatic evaluation resume is not implemented).

## Exit codes and recovery

All commands print JSON results or diagnostics. The evaluation CLI uses these
exit codes; a wrong gold answer alone is an evaluation outcome, not a command
failure.

| Code | Meaning | Next action |
| --- | --- | --- |
| `0` | Selection preview, preparation, readiness check, evaluation, or offline report generation completed successfully. | Inspect the report; a successful run may still contain incorrect answers or abstentions. Report generation can succeed for partial evaluation artifacts. |
| `1` | Preparation has failed/incomplete work, or evaluation has query/metric failures, incomplete graph traversal, seed mismatch or graph-snapshot inconsistency. | Inspect ingestion details or per-row scores/statuses; address the cause before repeating the affected work. |
| `2` | Invalid arguments/configuration, missing or conflicting selected graph data, unavailable prerequisites, setup error, invalid report artifacts, or HTML generation failure. | Read the JSON diagnostic/preflight issues. Preserve saved evaluation artifacts; resolve the cause and regenerate HTML with `report --input PATH` when applicable. |
| `130` | User interrupted the operation. | Preparation retains checkpoints; evaluation retains completed rows if artifact writing had begun. |

Evaluation output directories must be new or empty. To repeat an evaluation,
choose a new `--output` path. To continue after interruption, select the
remaining questions and write a new run directory; automatic evaluation resume
and saved-artifact rescoring are not implemented.

## Implementation and validation history

The five approved subtasks were completed together following the later user
instruction. The table and evidence below preserve the implementation handoff;
the command reference above describes current behavior.

| Subtask | Implementation | Validation |
| --- | --- | --- |
| 1. Dependency, data, providers | RAGAS 0.4.3; validated gold records; independent evaluator LLM and embeddings | All 2,417 dev records validated; both providers tested through actual RAGAS/Instructor/SDK code with mocked HTTP |
| 2. Abstention | Shared answered/insufficient-evidence output and validation | Both answer backends, invalid combinations, retained evidence and future false labels tested |
| 3. Selection and graph preparation | Compatible ranges, exact IDs, read-only readiness, optional resumable preparation | Boundary/alias/ID tests; live atomic replay skips extraction and embedding |
| 4. Paired evaluation | Shared vector, same seeds, separate record/corpus conditions, common answer boundary, gold and Triad scores | Deterministic failures/budgets/label isolation; live Neo4j + Ollama paired evaluation passed |
| 5. Reports and operations | Incremental JSONL, summaries, manifests, snapshots and documentation | Artifact alignment and graph-change tests; final test counts recorded below |

### Recorded implementation validation

Initial validation: **308 unit tests passed** (277 existing plus 31 new), and
the opt-in end-to-end fixture passed against live Neo4j and Ollama in 74.96s.
That fixture verifies missing/ready preflight, atomic replay with no repeated
provider work, both retrieval conditions, unchanged paired vector seeds,
graph structure, correct cited answers and all three actual RAGAS metrics.
It uses deterministic extraction/ingestion embeddings and live Ollama answers
and judging. It is a synthetic test fixture, not an additional benchmark
dataset. Its UUID-scoped graph/index is cleaned up and the pre-existing graph
hash is checked. Artifacts are under ignored `runs/evaluation-live/`.

```sh
uv run --locked python -m pytest tests/unit -q
RUN_EVALUATION_LIVE=1 EVALUATION_EVIDENCE_DIR=runs/evaluation-live uv run --locked python -m pytest tests/integration/test_evaluation_live.py -q
```

OpenRouter clients are covered with mocked HTTP through real RAGAS metric code.
A paid live answer/Triad test is written in
`tests/integration/test_evaluation_openrouter_live.py` and requires the explicit
`RUN_OPENROUTER_EVALUATION_LIVE=1` opt-in; it was not executed. No live OpenRouter
call was made. The entire 2,417-record live benchmark is not
run automatically: first select/check already-ingested records or explicitly
prepare the intended cohort.

Final validation: **347 tests passed in 101.74s**, using the full unit suite and
all opted-in Neo4j/Ollama fixtures (Task 13 stock/atomic checks and the new paired
evaluation included). The separate paid OpenRouter test is gated and its
default skip is checked without provider calls. `uv lock --check --offline`,
`pip check`, module compilation, CLI help/dry runs, and `git diff --check` pass.
The final live paired run contains four successful rows, zero Triad metric
failures and matching before/after graph snapshots; artifacts are under
`runs/evaluation-live-final/`. Cleanup preserved the original graph data.

A real read-only `graphrag-eval check --records 1` exited 2 and identified all
20 missing chunks for first dev record `2hop__460946_294723`, as intended. The
existing graph has not been repopulated with the dev set automatically.
Implementation is complete; full benchmark execution and user acceptance
remain operational steps, not claimed benchmark results.

The full non-paid regression command used was:

```sh
RUN_TASK13_ATOMIC_TESTS=1 RUN_TASK13_STOCK_WRITER_TESTS=1 RUN_EVALUATION_LIVE=1 EVALUATION_EVIDENCE_DIR=runs/evaluation-live-final uv run --locked python -m pytest tests -q --tb=short
```

Documentation update (20 October 2026): expanded this guide into a complete CLI
reference, including option applicability/defaults, subcommand effects and
outputs, provider precedence, and examples for existing graphs, preparation,
record selection, retrieval conditions and budgets. README and the operational
reference link to the same command definitions.

Documentation validation: all CLI option strings were checked against the
installed parser, 38 evaluation example commands parsed successfully, seven
offline previews selected the expected records, and 40 local file/section links
resolved. `git diff --check` passed. Documentation validation contacted no
database or model service and made no ingestion/evaluation writes.

### Troubleshooting records 4–6 (20 October 2026)

The requested `prepare`, `check`, and `run` commands were exercised against the
actual dev records and configured local services. The tool shell did not have
`uv` on PATH, so the installed `.venv/bin/uv` executable was used with the same
`run --locked graphrag-eval ...` arguments. No global PATH or `.env` values
were changed.

The selected records already had 58 verified paragraphs. Paragraphs 4 and 15
of record `2hop__481349_302087` had repeated `ReadTimeout` failures and were
absent from Neo4j. This made both `check` and `run` correctly reject the cohort.
The configured `qwen3:4b` model reported thinking support, while extraction's
explicit `think: false` setting had been commented out. The existing pipeline
regression test also failed on that missing field.

The user explicitly chose to retain `qwen3:4b` and disable thinking for
structured extraction, answering and RAGAS judging. The fix restores the
native extraction flag, adds it to answers, applies the configured native
answer timeout, and sets `reasoning_effort: "none"` for the Ollama judge.
Model identifiers, 120-second native request timeout and generation-token
limits were retained. With that fix, the two missing paragraphs completed
during a 6.343-second preparation run.

A subsequent evaluation exposed a separate judge-output defect: RAGAS's default
JSON-object mode allowed `qwen3:4b` to echo `user_input`/`context` instead of
returning required score fields such as `rating`. The Ollama evaluator now
uses Instructor's JSON Schema mode to enforce each metric's response schema;
OpenRouter's evaluator path retains its existing behavior. Offline HTTP tests
verify the actual non-thinking and schema fields sent through the SDK.

The intermediate evaluation was interrupted and its artifacts preserved at
`runs/eval-prepared-4-to-6-json-object-attempt-20261020T144039Z`. The completed
rerun occupies the originally requested `runs/eval-prepared-4-to-6` directory.

| Requested command | Final result |
| --- | --- |
| `uv run --locked graphrag-eval prepare --start-record 4 --records 3` | Exit 0; three records loaded, all 60 paragraphs verified, 58 reused and two newly ingested; no failures/investigation cases. |
| `uv run --locked graphrag-eval check --start-record 4 --records 3` | Exit 0; selected graph data, 768-dimensional index and `record_id` filtering verified. |
| `uv run --locked graphrag-eval run --start-record 4 --records 3 --output runs/eval-prepared-4-to-6` | Exit 0; all 12 expected rows completed, no query/metric failures, complete traversal and matching graph fingerprints. |

The measured answer outcomes are four answers and eight abstentions, with zero
normalized exact gold-answer matches. These are retained evaluation findings.
Under the agreed answered-only policy, faithfulness and answer relevance are
scored for the four answered rows; context relevance is scored for all twelve.
Abstained rows correctly have null/not-applicable answer metrics.

Validation: **318 unit tests passed**. Artifact checks verified twelve unique
trace IDs, six paired question/condition IDs, unchanged original questions,
matching response/answer evidence, citations within the supplied context,
consistent vector seeds and complete graph traversal. Detailed local evidence
is in `runs/evaluation-troubleshooting/records-4-to-6.json`; the evaluation
scores and full traces are in `runs/eval-prepared-4-to-6/summary.json` and
`records.jsonl`. No live OpenRouter calls were made.

### Evaluation progress bar — October 20, 2026

Implemented default tqdm progress for `run`, with `--no-progress` suppression,
active record/condition/backend/phase display, and increments after durable
result and summary writes. The bar uses stderr, preserving JSON on stdout,
and closes on interruption. Existing tqdm dependency reused.

Validation: **325 unit tests passed**. Added coverage for one or three records,
one or both conditions (including the requested 12-result total), saved query
and metric failures, stderr output, disabled progress, CLI option forwarding,
and partial artifacts on interruption. CLI help and `git diff --check` also
passed. These checks used fake retrieval and provider clients; no additional
live model calls were needed for this display change. No unexpected runtime
or dependency problems were found.

### Offline HTML reports — October 20, 2026

Completed the approved report implementation:

1. Artifact-only validation and per-record deduplication preserve gold labels,
   exact retrieved evidence, bounded context, prompts, score statuses, and counts.
2. Self-contained HTML provides separate condition charts, searchable record
   comparisons, supporting paragraphs/decomposition, citations, graph evidence,
   diagnostics, standalone PNG/SVG chart downloads, and three PDF print scopes.
3. `run` generates reports after finalizing JSON, including partial runs where
   possible. `report --input PATH` regenerates HTML offline and replaces no source
   JSON. Original evaluation errors survive report failures.
4. README, operational documentation, CLI reference, tests, and package-data
   configuration have been updated. Playwright is a locked optional test extra;
   browser viewing and report generation require no new runtime dependency.

Validation evidence:

- Full unit suite: **346 passed**. Report-specific tests cover source alignment,
  zero versus unavailable scores, abstentions/failures, incomplete/truncated
  artifacts, duplicate/corrupt rows, safe embedding, offline regeneration,
  interruption, and report-generation failures.
- Opt-in Chromium suite: **6 passed**. Checked offline search, condition changes,
  exact context display, PNG/SVG downloads, PDF scopes, and script-injection text.
- Synthetic full-dev-sized report: **2,417 records / 9,668 rows**, approximately
  **42.2 MB HTML**, generated in **1.31 s**, opened in **0.13 s**, searched in
  **0.009 s** on the local test machine. These measurements use synthetic fixture
  evidence; actual report size and timings depend on saved graph/context sizes.
- Existing `eval-prepared-4-to-6-part-3` run: generated `report.html` (**202,855
  bytes**) with no warnings; browser validation checked all twelve row contexts
  and gold supporting paragraphs, and overview values match `summary.json`.
  Details are in that run's `report-validation.json`; sample PNG/SVG/PDF exports
  are in its `presentation/` directory.
- Wheel build includes `evaluation/report_template.html`; CLI help and
  `git diff --check` passed. No live Neo4j or model calls were needed.

No unexpected report runtime problems remain. Full-report printing deliberately
expands all record/condition contexts and can be expensive for large runs. No
implementation subtasks remain for the approved report plan.

## Standalone paragraph extraction benchmark

The separate entity/relation extraction benchmark is documented in
[its own reference and milestone record](agent-coordination/entity-extraction-benchmark/entity-extraction-benchmark.md).
That document contains all commands, flags, provider/model selection, JSONL
artifact formats, resume behavior, deterministic scoring rules, tests and
implementation progress. [README examples](README.md#standalone-paragraph-extraction-benchmark)
provide the quick start.

All four implementation subtasks are now implemented. The earlier cohort/schema
checkpoint and README-only documentation history have moved to that dedicated
record. The benchmark has its own `graphrag-extraction-benchmark` command and
artifacts; the RAGAS runtime and `graphrag-eval` behavior are unchanged.
