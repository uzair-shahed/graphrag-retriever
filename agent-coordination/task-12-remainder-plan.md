# Task 12: durable retries and accurate ingestion progress

Date: 18 October 2026.

## Implementation sequence and user review gates

The user initially required a verification stop after each subtask. The agent
stopped after 12.1 and 12.2. The user then explicitly authorized **all remaining
tasks**, superseding the remaining per-subtask stops. Subtasks 12.3–12.5 were
implemented iteratively with passing gates before progressing. Required live
checks used real Neo4j with deterministic providers; no required check skipped.

**Final status:** all five subtasks are implemented and validated, ready for user
acceptance. Final results: **429 unit tests and 41 live tests passed**.

| Subtask | Deliverable | Status |
| --- | --- | --- |
| 12.1 | Source-scoped durable queue, request inventories, retry budgets, classification and deterministic storage/policy tests. | Complete. |
| 12.2 | Shared batch/preload/fixture failure capture, interrupted-attempt reconciliation, manifest synchronization and live safety checks. | Complete. |
| 12.3 | Selected-range resume, due retries, truthful chunk/record progress and real-process failed-chunk recovery. | Complete. |
| 12.4 | Inspect/drain/requeue command, source-independent replay, bounded selection and live recovery tests. | Complete. |
| 12.5 | Combined live acceptance/regressions, graph/provider audits and final operational documentation. | Complete. |

## Confirmed contract

- No same-run retries. Due selected failures retry on loader resume or through
  an explicit drain command. Ordinary resume stays within the record selection.
- Three total attempts by default; delays before attempts two and three are
  five and thirty seconds. Chunk budgets are separate from existing SDK and
  database-transaction retry behavior.
- Atomic source-scoped queues are separate from progress manifests. Fresh
  manifests must not erase failures. Requests and expected sibling inventories
  are retained so draining does not depend on the original dataset file.
- Retry transient service/database errors and malformed generated output.
  Authentication, bad input/configuration, unsafe graph state, unknown and
  exhausted failures are terminal. Keep all dead letters and history.
- Explicit targeted requeue with a reason starts a new bounded cycle without
  removing lifetime history. Use current configured models and record non-secret
  settings for each attempt.
- Always reconcile authoritative Neo4j state before reuse/replay. Successful
  siblings must not repeat provider calls. A parent record remains incomplete
  until every expected chunk verifies and no unresolved queue entry remains.
- Preserve sequential ingestion, disabled entity resolution and existing graph
  data. Paid-provider tests and unrelated retrieval changes remain out of scope.

## 12.1 implementation

`ingestion/retry_queue.py` provides versioned Pydantic-validated queue state,
copy-on-write mutations and atomic JSON persistence. Queue paths under
`runs/ingestion-retries/` include the full file checksum and full split-name
hash, avoiding truncated checksum and sanitized split-name collisions. The
queue additionally verifies full source identity internally.

Each record retains its exact expected IDs and validated requests; entries
retain canonical Task 13 fingerprints, source provenance, cycle policies,
non-secret configuration, attempt numbers, outcomes, error categories, retry
eligibility and terminal reasons. States are pending, in_flight, retry_wait,
dead_letter and resolved. Resolution methods require their caller to verify
Neo4j; local state itself is never evidence of database completion.

The writer flushes and fsyncs a temporary file, atomically replaces the target,
then fsyncs its directory. A persistence failure stops use of that queue instance
until it is reopened, including failures after replacement. Corruption, unknown
versions, duplicate JSON keys, incompatible sources, changed requests and
inconsistent inventories/history are rejected without replacing existing files.
Interrupted entries remain in_flight on open; no automatic demotion authorizes
unsafe replay.

`ingestion/retry_policy.py` supports optional `INGESTION_RETRY_MAX_ATTEMPTS`
and `INGESTION_RETRY_DELAYS_SECONDS` (JSON array) settings. A policy requires
exactly one finite nonnegative delay per possible retry and is captured per
cycle. Existing cycles do not change when environment settings change.

Classification examines stages and typed exception causes, contexts and
exception-valued arguments, including the installed GraphRAG wrappers. It does
not infer retryability from error-message substrings. Generated schema/JSON and
graph-validation failures are distinguished from invalid application input;
unsafe graph and permanent authorization/client failures override transient
nested causes. Arbitrary diagnostic/provider bodies are not stored in queues.

## 12.1 validation evidence

Commands:

```sh
.venv/bin/python -m pytest tests/unit/test_retry_queue.py tests/unit/test_retry_policy.py -q
.venv/bin/python -m pytest tests/unit -q
git diff --check
```

Results: **83 focused tests; 360 total unit tests passed** (277-test original
baseline plus 83 new tests). Tests cover complete source-independent request
round trips, full source/path isolation, terminal/retryable wrapped exceptions,
backoff boundaries, exhaustion, requeue history, invalid configuration, unsafe
transitions, corrupt files/history, filesystem failures before and after atomic
replacement, and independent subprocess reload of durable in-flight state.

At the 12.1 review gate, production ingestion was unchanged and the queue was
not wired into loaders. No Neo4j mutations or provider calls occurred in 12.1.
The user then authorized 12.2. The retry CLI and final operational documentation
were subsequently completed in 12.4 and 12.5.

## 12.2 implementation and recovery contract

Batch loading (including callers without manifests), query preload and fixture
commands now persist full requests through the shared `ingest_paragraph`
boundary. Each pipeline attempt is journaled before provider execution. Typed
component failures retain their extraction, graph-validation, embedding or write
stage when available, with cause-aware classification and non-secret model/
database configuration. Settings, connection and reconciliation failures are
persisted separately without inventing provider attempts.

The shared boundary reconciles Neo4j before replay: complete atomic inventories
resolve the queue without extraction/embedding calls; partial, conflicting and
legacy graph states are preserved and dead-lettered. Interrupted in-flight work
must first be proven absent before its consumed attempt is closed and bounded
retry eligibility is scheduled. An intact commit after a verification timeout
or local queue-write failure is reused on the next invocation. Existing terminal
entries are never reactivated by a transient reconciliation failure.

Queue resolution happens before progress verification. Queue persistence errors
and manifest checkpoint errors stop processing instead of becoming ordinary
chunk/provider failures. Optional invalid-response logging cannot suppress the
original failure or prevent it from being durable. Previously verified progress
entries require fresh reconciliation before contributing to record completion;
final completion also requires every expected queue entry to be resolved.

Legacy manifest failures retain their recorded lifetime attempt count, including
counts already greater than the new budget. Their untyped historical diagnostics
are not sufficient to authorize automatic replay: they become durable
`legacy_unclassified` dead letters until explicit review/requeue. No fake
individual attempt histories are manufactured. Requeue preserves lifetime
numbering and starts a separate bounded cycle. Existing manifests are retained;
legacy failure import must succeed **before** `--fresh-manifest` archives old
progress. An import persistence failure leaves the original manifest intact.

The source queue directory can be overridden with
`INGESTION_RETRY_STATE_DIR` for isolated runs/tests; changing or replacing a
progress manifest does not change the default retry directory. Historical
synthetic fixtures use an opaque checksum token (`fixture-sha256`). Queue paths
hash opaque tokens safely with an `opaque-` prefix while retaining exact request
provenance internally; ingestion never rewrites that historical identity. Real
dataset sources continue to use their complete file SHA-256.

At the 12.2 handoff, the shared boundary already enforced eligibility to avoid
immediate/unbounded replay. Rich resume/progress fields and queue operations were
still pending; they were subsequently completed below. No paid provider service
was used during this subtask.

## 12.2 validation evidence

- Full deterministic suite: **392 unit tests passed**, including 31 focused
  shared-capture cases and one additional queue chronology regression beyond
  the 360-test 12.1 baseline.
- Focused retry queue, policy and capture suite: **115 passed**.
- Required live suite: **29 passed, zero skips** (10 new Task 12.2 checks and
  all 19 Task 13 atomic-ingestion regressions), using Neo4j Enterprise 2026.08.1,
  installed GraphRAG 1.21.0 and deterministic provider doubles.
- Live tests prove transactional rollback after graph/completion writes;
  durable failure capture before successful siblings continue; incomplete parent
  records; captured malformed output and real preload/fixture failures;
  queue-resolution and manifest-checkpoint recovery without repeating providers;
  and preservation/dead-lettering of partial, legacy and conflicting graph data.
- Existing atomic regressions still cover real subprocess death before commit
  and after commit before checkpoint, missing/fresh manifests, unique identities,
  transaction retries, lost commit acknowledgment and independent graph audits.
  Unknown injected errors now require explicit reviewed requeue before replay;
  identity changes may fail at queue validation before the database boundary.
- Atomicity-focused tests explicitly use zero retry delays; policy/capture tests
  retain fake-clock checks of the real five/thirty-second default schedule.
- Every one of the 29 live reports has matching before/after graph snapshot
  hashes. Existing graph data was preserved. No graph reset or legacy adoption
  occurred. `git diff --check` passed.

Commands:

```sh
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest tests/unit/test_retry_queue.py tests/unit/test_retry_policy.py tests/unit/test_retry_capture.py -q
RUN_TASK12_RETRY_TESTS=1 RUN_TASK13_ATOMIC_TESTS=1 TASK12_EVIDENCE_DIR=runs/task12-2/retry-capture TASK13_EVIDENCE_DIR=runs/task12-2/atomic-regressions .venv/bin/python -m pytest tests/integration/test_retry_capture_integration.py tests/integration/test_atomic_ingestion.py -q
```

Evidence is retained locally under `runs/task12-2/`: per-test sanitized reports
in `retry-capture/` and `atomic-regressions/`, and `validation-summary.json`.
These are gitignored runtime artifacts; this tracked report preserves the
verified results. The agent stopped at 12.2; the user then authorized all remaining
work together.

## Verified progress behavior in 12.3

The first ten records of musique_ans_v1.0_dev.jsonl each contain twenty
paragraphs (200 planned chunks). The current tqdm bar counts processed chunks,
including failures; records_loaded counts fully completed records. Processed or
successful chunks divided by twenty is not a complete-record count because
successes can be spread across incomplete records. The current progress record
counter also updates at record boundaries and can remain visually stale on
reuse. The final summary instead consults manifest completion.

12.3 preserves strict completion and shows processed/newly verified/reused/
failed/investigation/deferred counts separately. Complete/partial/unstarted
records derive from exact expected source/record/paragraph identities. Tests
cover the 145-processed/five-failed case, variable paragraph counts, interleaved
records, duplicate record IDs across sources, reused last chunks and dry runs.
Completion was not weakened to match aggregate chunk arithmetic.


## 12.3 — Bounded resume and truthful progress

- Due selected failures execute once per invocation; no same-run retry loop or
  wait exists. Out-of-range queued failures remain untouched by ordinary resume.
- Exact expected paragraph identities drive the postfix after every processed
  chunk, including reuse and the final chunk. JSON separates processed, newly
  verified, reused, failed, investigation, deferred, retry-attempted, unresolved
  and dead-letter counts, with queue paths and per-entry policy/status metadata.
- A fake clock proves the default five/thirty-second boundaries, preserved
  exhaustion and fresh-manifest behavior. Dry runs report planned records as
  unstarted without creating retry files or contacting services.
- A real new-process restart proves a failed chunk retries from its durable
  queue while its successful sibling preserves its inventory and provider-call
  counts. The record stays partial until both source chunks verify.
- Gate evidence at 12.3: **399 unit tests and 30 live tests passed**.

## 12.4 — Queue operations CLI

- Added `python -m graphrag_pipeline.ingestion.retry_chunks` with offline
  `inspect` and atomic targeted `requeue`, plus a source-independent `drain`.
- Explicit queue paths are required. Repeated record/chunk filters intersect;
  unknown selections fail before provider work. Requeue requires terminal IDs
  and a nonblank reason, preserving history in a new bounded cycle.
- Drain snapshots due/pending/interrupted candidates once and respects
  `--max-chunks`. Newly due work is left for the next invocation. It never runs
  providers for unselected siblings, but independently verifies every stored
  sibling before reporting an affected record complete.
- Queues retain validated absolute progress-manifest references. Existing
  manifests synchronize after source/inventory checks; explicit overrides can
  attach older custom manifests. Missing linked manifests are reported without
  blocking queue/Neo4j verification. Original datasets can be unavailable.
- Structured output includes per-entry budgets/categories/timestamps, outcomes,
  errors, graph-verified record status and progress linkage. Exit codes are
  0/1/2/130 for successful selected work, unresolved work, invalid state/arguments
  and clean stop. A filtered successful drain can still report a partial parent.
- Gate evidence at 12.4: **421 unit tests and 36 live tests passed**. The live
  default-verifier regression caught a missing assignment during extraction of
  the shared read-only reconciliation helper; it was fixed and covered before
  completing this gate.

## 12.5 — Final acceptance and handoff

- Final deterministic command: `.venv/bin/python -m pytest tests/unit -q`
  → **429 passed** (152 additional tests over the original 277-test baseline).
- Final required live command:

  ```sh
  RUN_TASK12_RETRY_TESTS=1 RUN_TASK13_ATOMIC_TESTS=1 RUN_TASK13_STOCK_WRITER_TESTS=1 TASK12_EVIDENCE_DIR=runs/task12-final/retry TASK13_EVIDENCE_DIR=runs/task12-final/atomic .venv/bin/python -m pytest tests/integration/test_retry_capture_integration.py tests/integration/test_atomic_ingestion.py tests/integration/test_stock_writer_behavior.py -q
  ```

  → **41 passed, zero skips**: 20 retry/capture/CLI checks, 19 atomic-ingestion
  regressions and two stock-writer characterization checks. Real Neo4j Enterprise
  2026.08.1 and installed GraphRAG 1.21.0 were used with provider doubles.
- New real-process checks cover exhaustion, reviewed requeue, pre-commit death,
  post-commit/pre-queue-checkpoint death, failure-after-journal death, missing
  source files, and actual SIGINT/SIGTERM clean stops. Provider event ledgers
  prove current work finishes and successful chunks do not repeat model calls.
- Final upgrade audit covers manifests interrupted before retry queues existed:
  recorded attempts remain lifetime counts; matching committed work can resolve
  even an exhausted old count, while absent exhausted work cannot get a fresh
  budget. No individual historical provider events are fabricated.
- All **41** per-test cleanup hashes match their original graph snapshots. The
  stock writer's separate report format was independently checked using its
  baseline/cleanup SHA-256 fields. No existing graph reset, repair or legacy
  certification occurred. No paid provider calls were made.
- Updated README, operational documentation, `.env.example`, active contracts
  and task status. CLI help commands and full test collection pass; whitespace
  checks include new files. Sequential ingestion and disabled entity resolution
  remain the supported contract. Optional real-provider/retrieval suites are
  outside this ingestion acceptance; no required live check was skipped.

Acceptance mapping:

| Task 12 criterion | Verified evidence |
| --- | --- |
| Failed chunk restarts from durable state; successes avoid provider replay | `test_failed_chunk_resume_in_new_process_skips_successful_sibling`, source-independent drain and provider event ledgers. |
| Attempts, backoff, terminal/exhausted errors and dead letters remain visible | Fake-clock queue/resume tests, inspect output, CLI exhaustion and explicit requeue history (four lifetime attempts over two cycles). |
| A failed/unresolved sibling keeps its record incomplete until verified | Distributed-failure progress tests, live rollback partial-record check, independent sibling verification and manifest synchronization. |
| Replay depends on verified atomic idempotency | All Task 13 atomic regressions, independent graph inventories, lost-acknowledgment tests and preserved cleanup snapshots. |

Local gitignored evidence: `runs/task12-final/validation-summary.json`,
`runs/task12-final/retry/test_*.json` and
`runs/task12-final/atomic/test_*.json`. Reports retain statuses, request identity,
provider events, relevant graph inventories, version/configuration names and
cleanup hashes. Source queues grow as full JSON request/history stores; one
writer must operate at a time. Other platforms/database deployments and paid
provider/full-dataset performance remain unverified, as documented.

Implementation and required validation are complete. User acceptance is not
presumed. No Task 12 subtask remains outstanding.
