# MuSiQue paragraph entity and relationship extraction benchmark

Updated: 20 October 2026. This is the complete usage reference and implementation
handoff for the standalone benchmark. RAGAS evaluation keeps its reference in
`vectorrag_vs_graphrag_evaluation_ragas.md`.

## Purpose and current state

Implementation is complete for cohort selection, structured gold generation,
resumable artifacts, append/range extension, candidate comparison, deterministic
scoring and offline summaries. Generating the real 150-paragraph gold set remains
an operational step: choose an OpenRouter reasoning model and run the gold command below.
Model-generated gold is authoritative for this experiment; it has no human
annotation/review stage.

This experiment measures extraction quality as a proxy for possible GraphRAG
impact. It does not establish the effect on end-to-end answer accuracy. Its
commands do not access Neo4j, ingest graphs, create embeddings, retrieve evidence,
or call the RAGAS evaluator. Both providers use the same entity/relation schema
and extraction instructions. Generation can vary between calls; scoring is
deterministic.

## Installation and help

Run from the repository root, with the dataset available locally:

```sh
uv sync --locked --extra dev
uv run --locked graphrag-extraction-benchmark --help
uv run --locked graphrag-extraction-benchmark select --help
uv run --locked graphrag-extraction-benchmark gold --help
uv run --locked graphrag-extraction-benchmark run --help
uv run --locked graphrag-extraction-benchmark report --help
```

The equivalent module entry point works in the existing environment without
reinstalling the project script:

```sh
.venv/bin/python -m graphrag_pipeline.extraction_benchmark.command --help
```

On Windows, use `.venv\Scripts\python.exe` for this alternative. All `uv`
examples below are single lines usable in POSIX shells and PowerShell.

## Commands and examples

### 1. Preview exactly 150 paragraphs

```sh
uv run --locked graphrag-extraction-benchmark select
uv run --locked graphrag-extraction-benchmark select --paragraphs 3
uv run --locked graphrag-extraction-benchmark select --dataset-path dataset/musique/data/data/musique_ans_v1.0_dev.jsonl --paragraphs 150
```

Selection flattens records and each record's paragraph list in source order.
Indices are stable identifiers, not sort keys. The final record can be selected
in part, and identical text at different source positions is retained. Invalid
records, duplicate IDs/indices and insufficient paragraphs fail explicitly.

For the verified local dev dataset, the first 150 paragraphs span eight records:

- First: `musique:2hop__460946_294723:paragraph:0`.
- Last: `musique:2hop__390947_232243:paragraph:9`, source line 8.
- Source SHA-256: `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.

`select` prints selection metadata to stdout, makes no service calls and writes
no files. Counts always mean paragraphs, not records or extracted entities.

### 2. Generate gold with OpenRouter reasoning enabled

Set `OPENROUTER_API_KEY` in your local `.env` or environment. Replace
`YOUR_REASONING_MODEL_ID` with your chosen OpenRouter model ID. No model is
selected implicitly. The model must advertise structured output and reasoning
control; the command checks advertised effort levels when the model supplies
them. Endpoint support is also enforced in the generation request.

Preview locally, without credentials or model-service access:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150 --dry-run
```

Generate the actual 150 gold labels (provider calls incur charges):

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150
```

Start with a separate two-paragraph smoke run if desired:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 2 --max-tokens 32768 --output runs/entity-extraction/gold-smoke
```

Reasoning effort is mandatory for gold and cannot be `none`. The token budget
may include internal reasoning as well as visible JSON, depending on the
provider. A completed but truncated response is a failure, even if its text
happens to be parseable JSON. No LLM repair call silently changes the result.

### 3. Resume or retry failed gold calls

Repeat the same configuration and output path with `--resume`:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150 --resume
```

By default, resume skips both successful and failed saved attempts and performs
only missing work. To retry failures as well:

```sh
uv run --locked graphrag-extraction-benchmark gold --model YOUR_REASONING_MODEL_ID --reasoning-effort high --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-150 --resume --retry-failed
```

Successful rows are always skipped. Retries increment the saved attempt count
and replace that paragraph's latest outcome; earlier raw failure text is not an
attempt-history log. Model/schema/prompt/source/budget/endpoint/timeout/retry
settings must match the saved run. Changing them requires a new output directory.

The paragraph count must also match `configuration.paragraphs` in the saved
`manifest.json`. For example, retrying the 12 failures in the existing
`gold-run-150` run requires `--paragraphs 150`, even if you now want 180:

```sh
uv run --locked graphrag-extraction-benchmark gold --model openai/gpt-6.1-sol --reasoning-effort medium --paragraphs 150 --max-tokens 32768 --output runs/entity-extraction/gold-run-150 --resume --retry-failed
```

Requesting 180 with ordinary resume stops before provider calls. Use the append
selectors below to extend the saved cohort. A retried extraction can still fail
validation; check saved attempt counts to distinguish a retry from skipped work.

### 3a. Append more paragraphs to an existing run

To extend a saved 150-paragraph gold run to 180, repeat its exact model and
generation settings and choose either of these equivalent selectors:

```sh
uv run --locked graphrag-extraction-benchmark gold --model openai/gpt-6.1-sol --reasoning-effort medium --max-tokens 32768 --output runs/entity-extraction/gold-run-150 --additional-paragraphs 30
uv run --locked graphrag-extraction-benchmark gold --model openai/gpt-6.1-sol --reasoning-effort medium --max-tokens 32768 --output runs/entity-extraction/gold-run-150 --start-paragraph 151 --end-paragraph 180
```

Run **one** command. Counts mean paragraphs, and range endpoints are inclusive,
1-based positions in the flattened source order, not dataset records, source
lines or per-record `idx` values. A range must start immediately after the saved
cohort. Overlaps, gaps, reversed ranges and insufficient input are rejected.
Append selectors cannot be combined with `--paragraphs`.

Append requires an existing run with an attempt saved for every current
paragraph/model. Finish pending work with ordinary resume first. Saved failures
are allowed and stay unchanged by default. To retry them while adding new
paragraphs, also pass `--resume --retry-failed`. Successful rows, attempt counts,
raw responses and scores are preserved. Only newly added paragraphs generate
calls unless retries are explicitly requested. The output directory name does
not change; `gold-run-150` can contain 180 paragraphs after append.

The original source file (including checksum and saved prefix), models, prompt,
schema, budgets, endpoints and retry/timeout settings must match. Add `--dry-run`
to validate compatibility and preview the old count, added count and inclusive
range without provider calls or artifact writes.

Extend the gold cohort before extending candidate comparisons. Candidate
append uses the same selectors, takes inputs from saved gold, and validates
that previously frozen labels are unchanged:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-run-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate --additional-paragraphs 30
```

Use the settings and output path of your existing candidate run. Gold must
cover the target range; a larger gold cohort is allowed. Every candidate model
processes the same added paragraphs with usable gold, and the summary includes
the full requested cohort and its skipped paragraphs. Append positions count
all gold cohort paragraphs, including skips, rather than only comparable rows.

If append is interrupted, resume the expanded cohort instead of adding another
N. For gold, specify the new total; candidate resume reads its saved count:

```sh
uv run --locked graphrag-extraction-benchmark gold --model openai/gpt-6.1-sol --reasoning-effort medium --paragraphs 180 --max-tokens 32768 --output runs/entity-extraction/gold-run-150 --resume
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-run-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/local-candidate --resume
```

### 4. Compare an Ollama candidate

Start Ollama and install your chosen model, for example:

```sh
ollama pull qwen3:4b
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --ollama-think false --output runs/entity-extraction/qwen3-4b
```

`ollama:qwen3:4b` means provider `ollama`, model tag `qwen3:4b`; only the first
colon separates the provider. `--ollama-think false` requests no thinking on a
model that supports this control. Omit the flag to use the model's default.

### 5. Compare an OpenRouter candidate

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model openrouter:openai/gpt-4o-mini --output runs/entity-extraction/hosted-candidate
```

This is example syntax, not a guarantee that a particular hosted model/endpoint
supports all requested parameters. Preflight and the provider validate support.
The candidate model ID is explicit and independent of the gold model.

For reasoning-capable hosted candidates, add a supported effort:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model openrouter:YOUR_REASONING_MODEL_ID --reasoning-effort medium --output runs/entity-extraction/reasoning-candidate
```

### 6. Compare hosted and local models in one run

Repeat `--model` with each provider/model pair:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --model openrouter:openai/gpt-4o-mini --ollama-think false --output runs/entity-extraction/mixed-models
```

Each model receives the same paragraphs, title/text inputs and extraction
instructions. The candidate command uses successful validated gold labels and
skips paragraphs whose gold extraction failed or has no saved attempt. Candidate
models never see gold labels in their requests. No original dataset access is
required after gold generation; paragraph inputs come from the saved cohort.

Skipping is automatic. With 180 requested paragraphs and 130 failed gold
extractions, each candidate receives only the remaining 50 paragraphs. The JSON
summary includes `requested` (180), `comparable` (50), `skipped` (130), and
`skipped_gold`, an ordered list of paragraph `chunk_id`, `reason` and `error_type`.
Reasons are `gold_failed` with the saved exception type, or `gold_missing` with
null error type when no gold attempt exists. Full error messages are not copied
into skip records. Corrupt gold artifacts still fail validation.

Each model reports `expected` as the comparable count and `compared` as the
number of completed, scored candidate attempts, including candidate failures.
Gold skips count as neither candidate failures nor pending work and contribute
no TP/FP/FN. Scores therefore describe only the comparable subset. If all gold
labels are unavailable, the run completes successfully with zero comparisons,
zero metrics and no provider checks or generation calls. A zero metric in this
case describes an empty scoring population. Dry-run previews include coverage
counts and skipped IDs without calls or artifact writes.

Duplicate provider/model entries are rejected. Candidate reasoning settings
apply to all candidates of the matching provider in this run. Use separate
output directories to compare the same model with different budgets/efforts.

Preview the matrix without model calls or artifact writes:

```sh
uv run --locked graphrag-extraction-benchmark run --gold runs/entity-extraction/gold-150 --model ollama:qwen3:4b --model openrouter:openai/gpt-4o-mini --output runs/entity-extraction/mixed-models --dry-run
```

Resume a candidate run using the same arguments plus `--resume`, and optionally
`--retry-failed`, just as for gold generation.
The gold fingerprint and skip set are frozen with the candidate run. Resume
rejects a changed gold snapshot; append preserves existing labels and skip
records and applies eligibility to the new source positions. To include labels
repaired for previously skipped paragraphs, start a new candidate output
directory. Candidate retries repeat candidate failures, not gold skips.

### 7. Inspect or regenerate scores offline

```sh
uv run --locked graphrag-extraction-benchmark report --input runs/entity-extraction/mixed-models
uv run --locked graphrag-extraction-benchmark report --input runs/entity-extraction/gold-150
```

`report` validates saved rows and regenerates `summary.json`. It also completes
an interrupted append metadata commit when needed. It checks
stored per-paragraph scores against deterministic recomputation. It needs no
dataset, original gold directory, credentials, models, or graph. A partial run
is reportable: pending paragraphs are shown explicitly and are not scored yet.
This command prints JSON; there is no HTML report for this standalone benchmark.

## Complete flag reference

All subcommands support `-h` / `--help`. Arguments have no implicit inheritance
from the answer pipeline or RAGAS model settings.

| Flag | Commands | Type / default | Meaning |
| --- | --- | --- | --- |
| `--dataset-path` | `select`, `gold` | Path; MuSiQue dev file above | Local input JSONL. |
| `--paragraphs` | `select`, `gold` | Positive integer; `150` | Exact paragraph prefix size; insufficient input fails. |
| `--additional-paragraphs` | `gold`, `run` | Positive integer; omitted | Append the next N paragraphs to an existing fully attempted cohort; excludes range selectors and `--paragraphs`. |
| `--start-paragraph` | `gold`, `run` | Positive integer; omitted | First new 1-based paragraph position; must be saved count + 1 and requires `--end-paragraph`. |
| `--end-paragraph` | `gold`, `run` | Positive integer; omitted | Inclusive last new paragraph position; must be >= start. |
| `--model` | `gold` | Required string | OpenRouter model ID without a provider prefix. |
| `--model` | `run` | Required repeatable string | `openrouter:MODEL_ID` or `ollama:MODEL_TAG`. |
| `--reasoning-effort` | `gold` | Required choice | `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; must be supported by the selected model. |
| `--reasoning-effort` | `run` | Optional choice; omitted | OpenRouter candidates: the choices above plus `none`; omitted uses model defaults. Unsupported control is rejected. |
| `--ollama-think` | `run` | String; `default` | `default` omits the field; `true`/`false` send booleans; other strings such as `high` are passed to Ollama for model-specific validation. |
| `--gold` | `run` | Required directory | Gold run containing manifest and cohort; failed/missing labels are skipped and reported. |
| `--output` | `gold`, `run` | Required directory | New/empty output or an existing matching run with `--resume` or append selectors. |
| `--max-tokens` | `gold`, `run` | Positive integer; `16384` | Output budget; maps to OpenRouter `max_tokens` or Ollama `options.num_predict`. |
| `--timeout` | `gold`, `run` | Positive finite seconds; `180` | Per-HTTP-request timeout, not a whole-run deadline. |
| `--retries` | `gold`, `run` | Nonnegative integer; `2` | Additional transient HTTP/transport retries. |
| `--openrouter-base-url` | `gold`, `run` | URL; environment/default below | Override OpenRouter API base, including `/api/v1`. |
| `--ollama-host` | `run` | URL; environment/default below | Native Ollama host, normally without `/v1` or `/api`. |
| `--resume` | `gold`, `run` | Switch; off | Validate and continue an existing compatible run; skip saved attempts. |
| `--retry-failed` | `gold`, `run` | Switch; off | Requires `--resume`; repeat saved failures, preserving successful rows. |
| `--no-progress` | `gold`, `run` | Switch; off | Hide stderr progress; retain final JSON stdout. |
| `--dry-run` | `gold`, `run` | Switch; off | Validate local inputs/model settings and print selection; no service calls or file writes. Does not check remote model support. |
| `--input` | `report` | Required directory | Existing gold/candidate run to validate and summarize offline. |

Runtime/configuration failures print JSON diagnostics. Argparse usage errors
print standard CLI help/errors to stderr. Exit codes are `0` for a successful
operation, `1` for saved failures or an incomplete offline report, `2` for setup /
configuration/artifact errors, and `130` for keyboard interruption. Low scores
alone do not make an otherwise completed run exit with an error.

## Provider configuration and schema

The CLI loads `.env` with process environment values taking precedence. Its
environment inputs are limited to:

| Variable | Default | Used for |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | None | Required only when an actual OpenRouter call is needed; excluded from saved settings and error headers. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint; CLI override takes precedence. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint; CLI override takes precedence. |

Model IDs and budgets always come from this benchmark's flags. It does not read
`GRAPH_RAG_PROVIDER`, `RAGAS_PROVIDER`, existing model-name/embedding settings or
Neo4j credentials. URL credentials, query parameters and fragments are rejected
so secret-bearing endpoints cannot enter artifacts. Credentials can change on
resume without changing the saved public configuration.

Both providers use `ParagraphExtraction.model_json_schema()` and the existing
`Mention`, `Triple`, `EvidenceSpan` types. The wire object has only `mentions`
and `triples`. Locally, `ExtractionResult` binds source identity/provenance and
validates unique mention IDs, relationship references, confidence `[0,1]`, and
exact body substrings at Python Unicode offsets `[start:end]`. Empty arrays are
valid results; malformed/extra fields or invalid spans produce failed rows.

OpenRouter receives `response_format.type=json_schema`, strict schema mode,
`provider.require_parameters=true` and `allow_fallbacks=false`. Gold reasoning
uses `reasoning.effort`; reasoning text is excluded from the returned content.
OpenRouter temperature is omitted because some reasoning models reject it.
Ollama receives native `/api/chat` `format=<schema>`, `stream=false`, temperature
zero and the requested `think` setting. JSON validation remains mandatory for
both providers. See the official [OpenRouter structured output guide](https://openrouter.ai/docs/guides/features/structured-outputs),
[OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
and [Ollama chat API](https://docs.ollama.com/api/chat).

### Python interfaces

For scripts, `extraction_benchmark.records.load_paragraphs(path, count=150)`
returns a list of shared `ExtractionRequest` objects. `path` defaults to the dev
file shown above; `count` is a positive integer. The returned objects include
local provenance. Pass only `contracts.model_input(request)` to a provider.
`contracts.ParagraphExtraction.model_json_schema()` returns the response schema;
`contracts.validate_extraction_response(request, content)` accepts raw JSON text
and returns a fully validated shared `ExtractionResult`, or raises a validation
error. These utilities make no service calls themselves.

`runner.run_extraction(..., append_start=S)` takes the complete target prefix,
validates that S is immediately after the saved cohort, and extends it while
reusing saved rows. Candidate calls also supply the complete target reference
and current gold fingerprint. `dry_run=True` performs local validation and
returns a prospective summary without creating files or clients.
For partial-gold candidates, pass `reference` containing only successful labels
and `skipped_gold` containing the ordered skip records. Every cohort paragraph
must have exactly one label or skip record. `artifacts.load_gold(path)` returns
`(requests, reference, fingerprint, skipped_gold)` with the full source cohort.

## Artifacts and recovery

All run artifacts live under the requested output directory. Examples under
`runs/` are ignored by Git.

| File | Content |
| --- | --- |
| `manifest.json` | Format version, run ID/time, frozen public model settings, full prompt/schema, scoring version, cohort checksum and gold/reference fingerprints where applicable; candidate configuration also freezes nonempty `skipped_gold` records. |
| `cohort.jsonl` | One label-free `ExtractionRequest` per paragraph, including title/body, stable IDs and source provenance/checksum. |
| `gold.jsonl` | Gold runs: latest saved attempt per paragraph, including validated extraction or error and raw returned content. |
| `predictions.jsonl` | Candidate runs: latest attempt per model/paragraph, with extraction/error and deterministic scores plus missing/unexpected sets. |
| `reference.jsonl` | Candidate runs: frozen successful gold extractions in cohort order, excluding skipped paragraphs; permits independent offline reporting. |
| `preflight.json` | Last invocation's model availability/capability checks for models needing calls. |
| `summary.json` | Per-model expected/attempted/success/failed/pending counts; candidate coverage, skipped IDs/failure types, per-model compared counts and entity/relation micro metrics. |
| `.append-transaction.json` | Temporary full target metadata snapshot used to recover an interrupted append commit; automatically removed after recovery. |

Saved attempts also include model specification, chunk ID, attempt number,
timestamp, elapsed time, provider response/model/usage metadata when returned,
and error details. Full source provenance for failed rows remains in the cohort.
Successful rows contain the full shared `ExtractionResult`.

Each completed attempt replaces the complete JSONL snapshot atomically after
flushing/fsync, then updates the summary. Corrupt, duplicate, unknown or
inconsistent saved rows fail validation. A crash between those two writes is
recoverable by resume/report, which derives the summary from the saved rows.
Source/prompt/schema/configuration/gold changes cannot be silently mixed with
existing results. Concurrent writers are rejected by `.benchmark.lock`.

Append journals the target cohort, manifest and candidate reference before
replacing those files, keeping the run ID and creation time. Results remain
intact. If metadata replacement is interrupted, readers validate the journal's
target snapshot; resume or report finishes the replacements under the run lock.
Dry-run reads that snapshot without changing files. Keep the transaction file
until recovery completes. Existing format-v1 runs can be extended directly;
ordinary runs require no migration.

On a normal keyboard interruption, the lock is released and completed rows are
retained. If a process is forcibly killed, confirm it has stopped before
manually removing its stale `.benchmark.lock`. A request that completed remotely
before the process persisted it may be billed again on resume; the workflow
cannot guarantee exactly-once execution of an external model call. If a process
dies during initial directory setup before a valid manifest/cohort is saved,
preserve that directory for inspection and choose a new output directory.

Transport errors, HTTP 408/429 and HTTP 5xx retry with bounded exponential waits;
other HTTP errors and JSON/schema/span failures do not retry automatically.
No model fallback or judge/repair model is invoked. Retrying failed candidates
changes their final measured outcome, so retain separate run directories when
you need the original first-attempt measurement.

## Deterministic scoring

Scoring version: `normalized-exact-v1`.

1. Apply Unicode NFKC normalization and case folding, replace Unicode punctuation
   with spaces, and collapse whitespace. Do not remove articles, stem words,
   resolve aliases, or infer synonymous relationships.
2. Entity keys are `(normalized mention, normalized entity type)`.
3. Relation keys are `(source entity key, normalized relationship type, target
   entity key)`; direction and endpoint types count. Mention IDs are resolved
   locally and need not match between models. Evidence spans/confidence validate
   outputs but do not enter the matching key.
4. Deduplicate entities and relationships into sets within each paragraph.
   Calculate TP/FP/FN, precision, recall and F1 independently for entities and
   relations. Zero denominators yield zero. Save missing/unexpected keys.
5. Micro scores sum TP/FP/FN across completed paragraph attempts per model,
   then calculate precision/recall/F1. Paragraphs remain separate: matches cannot
   cross paragraph boundaries. A failed candidate predicts empty sets and thus
   contributes every gold item as a false negative. Pending work is unscored and
   the summary marks the run incomplete. Paragraphs without usable gold are
   skipped for every candidate and excluded from scores and pending counts.

Free entity/relation vocabularies can produce semantic equivalences that exact
matching marks different (for example, `LOCATED_IN` versus `IS_IN`). This is a
known limitation of the chosen deterministic metric. Gold-model errors and
evidence-span failures also affect results. Report provider failures alongside
quality scores rather than describing this benchmark as a direct measurement
of GraphRAG answer accuracy.

## Milestones and validation record

| Milestone | State | Evidence / handoff |
| --- | --- | --- |
| 1. Cohort and extraction contract | Complete | First 150 paragraphs checked against actual source; 45 initial schema/adapter/benchmark tests passed. |
| README checkpoint reference | Complete, superseded by full CLI reference | Initially documented Python previews while provider execution was pending. |
| 2. Gold generation and resume | Complete | Explicit reasoning/model controls, common schema, atomic JSONL and resume/retry validation; interruption and provider-wire tests. |
| 3. Candidate matrix and scoring | Complete | Both providers, frozen gold, strict row validation, micro and per-paragraph scores, visible failed requests and offline reporting. |
| 4. Documentation and compatibility | Complete | Standalone reference here, README commands/flags and link from the RAGAS document; all four answer/evaluator provider combinations verified. |

The user subsequently authorized completion of all remaining subtasks, replacing
the original per-subtask review pauses. The benchmark adds only its own command
and package. `httpx` was already locked transitively; it is now an explicit
dependency using the same locked version. No changes were required to the answer
or RAGAS runtime implementation.

Run the focused tests:

```sh
uv run --locked python -m pytest tests/unit/test_extraction_benchmark.py tests/unit/test_extraction_benchmark_runtime.py -q
```

Run the full unit suite, including existing answer and RAGAS provider tests:

```sh
uv run --locked python -m pytest tests/unit -q
```

Original implementation evidence: **58 benchmark tests passed; 404 full unit
tests passed**. Coverage includes real HTTP payloads via mocked transports for both
providers, advertised reasoning support, truncation, retry bounds, label
isolation, interruption/resume skipping, explicit retries, artifact corruption,
file locks, atomic replacement, type/direction/duplicate matching, failed
candidate denominators, multiple candidates and offline reporting without the
source dataset or original gold directory.

Current focused benchmark suite: **89 tests passed**. Append/range coverage
additionally exercises both 150-to-180 selectors, source
and configuration rejection, saved failures and explicit retries, candidate
reference preservation and cumulative scoring, larger-gold prefixes, interrupted
generation, and recovery at each metadata replacement boundary. All generation
in these unit tests uses mocked providers.

Additional completion checks passed: locked project installation and the actual
console entry point, 150-paragraph console selection, gold dry-run without
artifact writes, parsing every README/reference benchmark command and documenting
every parser flag, `uv lock --check --offline`, and `git diff --check`. A fresh
benchmark-command import loads neither the Neo4j driver nor the RAGAS runtime.
The full unit suite also exercises the existing answer factories and real
RAGAS metric code with mocked HTTP for both providers; a new compatibility
matrix covers all four independent answer/evaluator provider combinations.

Opt-in live smoke tests use the same schema/provider client and a small tracked
paragraph fixture; no graph writes are involved. Set the following variables
in your shell (PowerShell uses `$env:NAME='value'` instead of POSIX `export`):

```sh
export RUN_EXTRACTION_BENCHMARK_OLLAMA_LIVE=1
export BENCHMARK_OLLAMA_MODEL=qwen3:4b
uv run --locked python -m pytest tests/integration/test_extraction_benchmark_live.py -q -rs
```

For a paid OpenRouter smoke test, explicitly select the model and effort:

```sh
export RUN_EXTRACTION_BENCHMARK_OPENROUTER_LIVE=1
export BENCHMARK_OPENROUTER_MODEL=YOUR_REASONING_MODEL_ID
export BENCHMARK_REASONING_EFFORT=high
uv run --locked python -m pytest tests/integration/test_extraction_benchmark_live.py -q -rs
```

Default integration collection skips these tests. Availability skips are
reported explicitly; malformed live extraction is a failure rather than a
successful smoke test. No real 150-paragraph gold labels or candidate quality
results are claimed by mocked tests.

### Live Ollama findings — 20 October 2026

The configured local `qwen3:4b-instruct` was reached successfully. It returned
schema-shaped JSON with three mentions and two relationships and a normal
completion status, but its evidence offsets did not match the source paragraph.
Both attempted one-paragraph checks failed the shared span validator. This is
an observed model-output limitation, not a successful live extraction or a
passing smoke test. Validation was kept strict and no span repair was added.

The diagnostic-enabled run saved its complete raw output, model settings, source
request, usage metadata and validation error at
`runs/entity-extraction/live-smoke/ollama.json` (ignored by Git). In a benchmark
candidate run, the same invalid output is saved as a failed extraction and
contributes empty predictions to scoring; this path is covered by the unit tests.
Gold generation retains it as a failure. Candidate evaluation now skips such
failed gold paragraphs and reports their IDs and failure types.

The live helper accepts `BENCHMARK_LIVE_EVIDENCE_DIR` to change its diagnostic
output directory. It saves `<provider>.json`; a later smoke run replaces that
provider's previous diagnostic. Default integration collection was also checked:
both live tests skip unless explicitly enabled. No live OpenRouter request was
made because an experiment gold model has not been selected. The implementation
is ready; the next operational step is choosing that model/effort and generating
the desired gold cohort with the commands above.
