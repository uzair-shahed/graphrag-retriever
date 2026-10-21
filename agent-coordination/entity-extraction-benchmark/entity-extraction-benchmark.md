# MuSiQue entity and relationship extraction benchmark

Updated: 21 October 2026. Artifact format v2 removes confidence and evidence spans
from the extraction contract. This standalone benchmark measures extraction
quality; it does not ingest Neo4j graphs or measure end-to-end answer quality.

## Output contract and scoring

Both providers receive the same extraction instructions, title and paragraph
body. They receive no gold labels, question, answer or support flags. Each mention
has only `mention_id`, `mention`, and `entity_type`. Each directed relationship
has only `source_mention_id`, `relationship_type`, and `target_mention_id`.

Identity, original paragraph text, and provenance are attached locally. JSON,
scalar types, unknown/extra fields, duplicate mention IDs, and missing endpoint
references remain strictly validated. Empty arrays are valid. Confidence and
`evidence_span` are neither requested nor accepted in newly generated output.
The former `--evidence-offsets` option has been removed.

Scoring remains `normalized-exact-v1`: normalize Unicode (NFKC/casefold), replace
punctuation with spaces, and collapse whitespace. Entity keys are normalized
mention/type pairs. Relationship keys include both endpoint keys, relationship
type and direction. Deduplicate within each paragraph. TP counts matches, FP
counts unexpected predictions, FN counts missed gold items. Precision is
TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean; zero denominators
yield zero. Overall micro metrics sum TP/FP/FN before calculating scores.

There is no synonym or alias matching. Semantic gold errors are possible:
model-generated gold has no human review stage. Removing span validation removes
verification of quote locations; the original paragraph and source provenance
remain available for inspection. Failed predictions count as empty sets. Pending
rows and unusable gold are excluded from the scoring population.

## Convert existing runs without generating gold again

Format-v1 gold and predictions are reusable through offline Python conversion.
Let any active writer finish first. An existing lock, including an unverified
stale lock, blocks conversion. Finish interrupted append transactions using the
v1 implementation before converting. Preserve the original directories.

Preview gold conversion, then write a separate v2 directory:

```sh
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/gold-run-150 --output runs/entity-extraction/gold-run-180-v2 --dry-run
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/gold-run-150 --output runs/entity-extraction/gold-run-180-v2
```

Then convert the completed candidate, using the converted gold:

```sh
uv run --locked graphrag-extraction-benchmark migrate --input runs/entity-extraction/qwen3-4b-instruct-aligned --gold runs/entity-extraction/gold-run-180-v2 --output runs/entity-extraction/qwen3-4b-instruct-v2
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-run-180-v2 --model ollama:qwen3:4b-instruct --ollama-think false --output runs/entity-extraction/qwen3-4b-instruct-v2 --resume
```

Conversion calls no providers and needs no API credentials or source dataset.
It strips only the retired fields from saved raw responses, validates all other
content and recovers eligible failures. It never repairs mention references,
invents items, or silently discards relationships. Malformed, missing or incomplete
responses remain failed; absent attempts remain pending. All 180 saved responses
in gold-run-150 passed a read-only structural projection, including its six span
failures. This is not a new generation or a semantic correctness guarantee.

Candidate conversion checks the old frozen reference against converted labels,
uses the entire saved cohort, and recomputes reference checksums, gold fingerprints,
skips, per-row scores and summaries. Recovered gold makes previously skipped
paragraphs pending: a 174-comparison candidate gains six missing predictions when
all 180 gold labels become usable. Resume generates only missing work. To retry
remaining failed predictions too, add --retry-failed explicitly.

The original raw response, attempts, generation timestamps, elapsed time and
usage/cost metadata are preserved. `migration.json` stores the original manifest,
source file hashes and original row statuses/errors; its checksum is frozen in
the new manifest. Each converted row has `response_format_version: 1`; new
responses have version 2. Offline reporting projects archived v1 raw responses,
while new v2 responses reject the retired fields. Summaries expose each model's
`generation_formats` counts (`legacy_v1` and `simplified_v2`): reused generations
still came from the older prompt. Scores after conversion describe a different
validation policy and possibly a larger scoring population than original scores.

Converted directories have new run IDs. Old files are never rewritten. The output
is validated in a temporary sibling directory and published only when complete.
A failed conversion publishes no partial run. Ordinary run/report commands on v1
directories explain that migration is required. Never edit only gold.jsonl: the
saved schema, references and checksums must remain consistent.

## Installation, selection and generation

```sh
uv sync --locked --extra dev
uv run --locked graphrag-extraction-benchmark --help
uv run --locked graphrag-extraction-benchmark select --paragraphs 150
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150 --dry-run
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150
```

Selection follows record and paragraph source order. Counts mean paragraphs,
not dataset records or entities. Identical text at different source positions
is retained. Duplicate IDs/indices, malformed input and insufficient paragraphs
are explicit errors. The default dataset is
`dataset/musique/data/data/musique_ans_v1.0_dev.jsonl`.

Gold requires one explicit OpenRouter model and an enabled reasoning effort.
OPENROUTER_API_KEY comes from the process environment or .env (process wins).
Preflight checks advertised structured-output and reasoning controls. Provider
requests enforce supported parameters without fallback. Ollama candidates use
native /api/chat schema-constrained generation, temperature zero, and explicit
think settings when supplied. Models/budgets do not inherit from GRAPH_RAG_PROVIDER
or RAGAS_PROVIDER. Endpoint credentials, queries and fragments are rejected.

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model openrouter:openai/gpt-4o-mini --output runs/entity-extraction/hosted-candidate
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --model openrouter:openai/gpt-4o-mini --output runs/entity-extraction/mixed-models --dry-run
```

Only the first colon separates provider and model. Duplicate model specifications
are rejected. Missing or failed gold labels are skipped for every candidate.
Summaries include requested/comparable/skipped counts, skipped_gold identities,
and expected/attempted/succeeded/failed/pending counts per model. If no gold is
usable, the run succeeds with zero comparisons and no provider calls.

## Resume, append and offline reports

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150 --resume --retry-failed
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150 --additional-paragraphs 30
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --max-tokens 32768 --output runs/entity-extraction/gold-150 --start-paragraph 151 --end-paragraph 180 --dry-run
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate --additional-paragraphs 30
uv run --locked graphrag-extraction-benchmark report --input runs/entity-extraction/local-candidate
```

Resume skips successes and saved failures by default; --retry-failed requires
--resume and retries failures only. Gold resume must specify its saved paragraph
count; candidate resume reads the saved count. Prompt/schema/source/model/budget/
endpoint/timeout/retry settings must match. Candidate references and skip sets
are frozen. Start a new run if previously frozen gold changes.

Append requires an existing fully attempted cohort (saved failures are allowed).
Choose either positive additional-paragraphs or an inclusive range starting at
saved count + 1. Do not combine with --paragraphs. Extend gold before candidates;
positions include skipped paragraphs. Dry-run validates without writes or calls.
Append journals metadata replacement; resume/report recover interrupted commits.
After interrupted generation, resume the expanded count rather than adding N again.

Reports validate rows/raw responses and recompute deterministic scores offline,
without original gold, dataset, credentials or model access. Each attempt is saved
atomically with a summary. Concurrent writers are rejected by .benchmark.lock.
A crash before summary replacement is recoverable from saved rows. Interrupted
requests not yet saved may need another provider call. Verify that a process has
stopped before manually removing a stale lock.

HTTP 408/429, 5xx and transport failures use bounded retries. Invalid outputs,
other HTTP errors and truncated generations are failures. No repair LLM is used.
Generation exit codes: 0 successful, 1 saved failures or incomplete report,
2 setup/artifact errors, 130 interruption. Migration exits 0 after a valid
conversion even if converted rows remain failed/pending; inspect its summary.
JSON stdout escapes Unicode so redirected Windows output remains valid.

## Visual comparison reports

Use `visualize` to compare any number of explicitly labeled format-v2 candidate
directories with the same gold cohort and fingerprint:

```powershell
uv run --locked graphrag-extraction-benchmark visualize --gold runs/entity-extraction/gold-run-180-v2 --run "Qwen-off=runs/entity-extraction/qwen3-4b-instruct-v2" --output runs/entity-extraction-reports/gpt-sol-qwen
```

Repeat `--run LABEL=PATH` to include more runs; all models in each directory are
included separately. The command validates frozen references and recomputes
metrics from saved rows. Input files and saved summaries are never rewritten.
It needs no provider, dataset, credentials or network access. Locked runs and
unfinished append transactions must be finished before visualization.

Open `index.html` for entity/relation precision, recall and F1 charts, exact
TP/FP/FN tables, extraction outcomes, and reasoning-on minus reasoning-off F1
changes. The report embeds its SVGs and download links; four standalone SVGs and
`comparison.json` are also written. Reasoning settings are read from manifests;
unspecified settings remain unclassified. Each complete on/off pair of the same
provider/model is shown separately, with other configuration differences listed.
No averaging or statistical significance claims are made. Failed predictions
count as empty predictions; gold skips and pending rows are excluded. Partial
runs remain visible but do not participate in reasoning deltas. Legacy generation
counts are shown even after migration.

`--output` must be separate from all input directories, without nesting. Existing
visualizer-owned files may be refreshed atomically; unrelated files are preserved
and filename collisions are rejected. Exit 0 means the report was generated,
including reports containing saved failures or pending work. Invalid or
incompatible artifacts exit 2; format-v1 artifacts require migration first.

## Flag reference

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
| `--run` | `visualize` | Repeat `LABEL=PATH` for each candidate directory; labels and directories must be unique. |
| `--gold` | `visualize` | Matching format-v2 gold directory. |
| `--output` | `visualize` | Separate directory for HTML, SVG and comparison JSON files. |
| `-h`, `--help` | All | Command help. |

## Verification

```sh
uv run --locked python -m pytest tests/unit/test_extraction_schemas.py tests/unit/test_extraction_benchmark.py tests/unit/test_extraction_benchmark_runtime.py tests/unit/test_extraction_benchmark_migration.py -q
uv run --locked python -m pytest tests/unit -q
```

Provider-wire, conversion, recovery, corruption, append/resume and isolation tests
use mocked providers. Optional live smoke tests remain in
`tests/integration/test_extraction_benchmark_live.py`; they require explicit
RUN_EXTRACTION_BENCHMARK_OLLAMA_LIVE or RUN_EXTRACTION_BENCHMARK_OPENROUTER_LIVE,
BENCHMARK_OLLAMA_MODEL or BENCHMARK_OPENROUTER_MODEL, and a hosted reasoning effort
where appropriate. Integration collection skips them by default.

On Windows, `.venv\Scripts\python.exe -m graphrag_pipeline.extraction_benchmark.command`
is an equivalent entry point; POSIX uses `.venv/bin/python`.
