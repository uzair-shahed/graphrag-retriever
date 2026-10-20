# Harsh review: the RAGAS implementation and the latest VectorRAG/GraphRAG run

**Review date:** 20 October 2026  
**Scope:** the evaluation implementation, stated evaluation protocol, selected MuSiQue data, saved run artifacts, and project handoff documents.  
**Read-only review:** no application code, source data, or existing run artifacts were changed. This report is the only file added for this request.

## Executive verdict

**The engineering work is substantially more careful than a typical first RAGAS integration. The experimental result is not yet strong enough to answer the project's central research question.**

The evaluation has a real CLI, preflight checks, paired retrievals, a shared answer boundary, gold-label checks, durable per-query evidence, graph snapshot fingerprints, explicit metric failures, and an offline report. Those are meaningful strengths. The report is professional and auditable as an artifact.

But the latest run contains only **three distinct questions** (repeated under two retrieval conditions), and only **one of those questions produced an answer**. The selected file is the answerable MuSiQue dev split. The persisted graph currently has no cross-paragraph entity-resolution connections, so the GraphRAG path is not being tested on the graph capability that motivates the project: joining evidence across paragraphs for multi-hop answers. The paired backends retrieve the same five seed chunk IDs in all six question/condition pairs; the graph path adds structured graph facts around those chunks but adds **zero gold supporting paragraphs**. That is a useful diagnostic observation, not evidence that GraphRAG is better or worse in general.

My recommendation is to **keep the custom evaluation harness and RAGAS as one judge layer**, while treating the current run as a smoke test and diagnostic slice only. The next decision-quality evaluation should combine: (1) deterministic MuSiQue gold metrics for retrieval and answers, (2) a graph-specific benchmark that can detect added multi-hop evidence, (3) a small human-labeled judge-calibration set, (4) paired per-question statistics and uncertainty, and (5) repeated, versioned runs. Do not select a winner from this three-question report.

## What I reviewed, and what counts as an instruction

The request asked for critique, verification, a beginner-friendly comparison of options, and a saved report; it explicitly prohibited changing or deleting project material except for saving this report. The linked project files were treated as **documents to analyze**, not as new instructions governing this review. In particular, workflow or acceptance directions embedded in the task plans describe the original implementation process; they do not authorize changing code or asking for acceptance here.

Reviewed files include the three supplied coordination plans, `documentation.md`, `README.md`, `vectorrag_vs_graphrag_evaluation_ragas.md`, the `src/graphrag_pipeline/evaluation/` implementation, `pyproject.toml`, and the complete saved run inputs/results in `runs/eval-prepared-4-to-6-part-3/` (`manifest.json`, `records.jsonl`, `summary.json`, and `report.html`). I compared the preceding part-2 `summary.json` only as historical context; the findings below refer to part 3 unless stated otherwise.

The project documents say the earlier evaluation implementation passed 346 unit tests and six opt-in browser tests, and that the part-3 report was checked against its 12 rows. Those are **documented validation claims**. I did not rerun tests or live services for this review. I independently read the source and recomputed/inspected the saved per-row results described below.

## What the implementation was supposed to do

The October 17 protocol sets a good target: use the same questions, graph/index snapshot, question embedding, vector seed count, answer model and answer policy; keep record-scoped and corpus-wide retrieval separate; hide gold answers and support labels from retrieval and answering; compare vector-only evidence with GraphRAG's seed-associated graph component; score retrieval separately from answer quality; preserve the exact context shown to the model; report added supporting evidence, citations, failures, incomplete traversals, cost/latency, and paired per-question outputs.

The code mostly implements that *experimental harness* well. `EvaluationCase.query()` carries the question and source metadata, while gold answers/support labels remain on a separate field. `PairedRetriever` embeds each question once and requests the same top-k for both backends. `run_evaluation()` stores retrieval evidence, selected answer evidence, actual prompt/context, response, citations, scores and trace IDs in each JSONL row. The same answer generator/validator is used for both paths. The report validates saved rows and recomputes aggregates rather than trusting a possibly stale `summary.json`.

The important qualification is that a controlled harness does not automatically create a valid or sufficiently powered experiment. Here, protocol mechanics are stronger than dataset coverage and graph-treatment validity.

## Headline result from the latest run

The manifest identifies the dataset checksum, three selected records, RAGAS `0.4.3`, Ollama `qwen3:4b-instruct` for generation and judging, `nomic-embed-text` for retrieval and answer-relevance embeddings, top-k 5, two conditions, and the same before/after graph fingerprint (713 nodes, 979 relationships, 269 indexes). There are 12 saved rows: 3 questions × 2 conditions (`record`, `corpus`) × 2 backends.

| Measure | VectorRAG | GraphRAG | What the artifact actually establishes |
|---|---:|---:|---|
| Saved rows per condition | 3 | 3 | These are the same three questions in each condition, not six independent questions. |
| `succeeded` / `insufficient_evidence` per condition | 1 / 2 | 1 / 2 | Both answer paths abstain on two of three questions. No row is marked `failed`. |
| Exact normalized answer accuracy | 1/3 (0.333) | 0/3 (0.000) | On the one question answered, VectorRAG matches the gold answer; GraphRAG does not. The same answer outcomes repeat under both conditions. |
| Token F1 | 0.333 | 0.083 | A tiny descriptive sample; not a stable system estimate. |
| MuSiQue support recall / hop coverage | 0.50 / 0.50 | 0.50 / 0.50 | Graph expansion does not improve gold paragraph/hop coverage in this sample. |
| MuSiQue support precision | 0.20 | 0.20 | Only one fifth of the five retrieved seed paragraphs are labeled supporting on average. |
| Graph-added gold supporting chunks | — | 0 in each paired sample | GraphRAG added graph nodes/relationships, but no additional labeled support paragraphs. |
| RAGAS faithfulness, scored count 1 | 1.00 | 0.00 | One answered example per condition/backend; do not treat this as a population estimate. |
| RAGAS answer relevancy, scored count 1 | 0.00 | 0.982 | The higher GraphRAG score belongs to the answer that failed exact gold matching. Relevance is not correctness. |
| RAGAS context relevance, scored count 3 | 0.333 | 0.333 | Same aggregate for both systems; includes abstention rows by design. |
| Mean answer-context token proxy | 546 record / 518 corpus | 1,146 record / 1,100 corpus | GraphRAG's rendered context is roughly twice as large in this run. `cl100k_base` is documented as a proxy, not the answer model's tokenizer. |
| Mean latency (excluding RAGAS judging) | 3.97s record / 3.26s corpus | 4.14s record / 1.71s corpus | Three rows and one local run are not a defensible performance comparison. |

An example makes the evaluation concrete. For **“Which company owns the manufacturer of Learjet 60?”**, gold is “Bombardier Inc.” VectorRAG answers “Bombardier Inc.” and passes exact match; GraphRAG answers “Bombardier Aerospace owns the manufacturer of the Learjet 60.” and fails exact match. Its RAGAS answer-relevancy is 0.982 while faithfulness is 0.0. That is not a contradiction: answer relevance asks whether the response addresses the question; faithfulness asks whether its claims are grounded in the supplied context; MuSiQue exact match asks whether it gives the expected answer. These measure different things. It is, however, a clear warning against presenting one RAGAS score as a quality verdict.

Two questions are abstained on by both backends, so faithfulness and answer relevance have `not_applicable` statuses for them. This is a reasonable policy, and the report displays denominators. Its consequence is that the headline RAGAS answer metrics are effectively **one scored case** each. The report does not conceal that count, but a casual reader could still over-read the plotted average.

## Harsh findings by area

### 1. The sample cannot support a winner

Three questions are a debugging sample, not a benchmark result. Repeating those questions across record/corpus conditions is useful for checking condition wiring, but it does not increase the independent question count. There are no confidence intervals, paired significance tests, repeated generation runs, or sensitivity analysis. One answer changes a per-condition accuracy by 33 percentage points. With one answerable outcome, a model choice based on the chart would be essentially a choice based on one case.

The measured answer outcomes are also sensitive to the generation/judge model, prompt, graph extraction state and local serving behavior. `temperature=0` reduces one source of randomness; it does not establish determinism across providers, model revisions, hardware, decoding implementations, or repeated executions.

### 2. The treatment does not test cross-paragraph GraphRAG

The saved rows show identical vector seed IDs for VectorRAG and GraphRAG in every matched record/corpus pair. That is appropriate for an ablation where the treatment is graph expansion after a shared seed search. The issue is what “expansion” means in the current data: project documentation explicitly says current graph components do not join separate source paragraphs and entity resolution remains disabled. GraphRAG enriches each retrieved paragraph with extracted nodes and relationships; it does not bridge the chain of separate supporting paragraphs required by many MuSiQue multi-hop questions.

Therefore this implementation can test whether **within-paragraph graph facts** affect answer generation. It cannot yet answer whether **cross-paragraph graph connectivity** improves multi-hop retrieval/answers. The fact that graph-added supporting chunk IDs are empty is consistent with the documented graph design, not proof that traversal code is broken. But it means the experiment does not test the project's primary GraphRAG thesis.

There is a second attribution issue: the two backends use the same answer model and policy, but GraphRAG's model-visible context includes graph node/relationship lines while VectorRAG's does not. This is a fair end-to-end comparison of two context strategies, but it combines **retrieval expansion and context representation**. If GraphRAG answers change, the current design cannot attribute the change to discovering additional source evidence versus changing how the same source text is represented. Report both as a system comparison, then add a staged ablation to isolate them.

### 3. The selected dataset split cannot evaluate abstention safety

The configured input is `musique_ans_v1.0_dev.jsonl`; the docs say it contains 2,417 answerable records. An answerable-only split can test how often the system finds an answer, but it cannot establish whether it correctly refuses unanswerable questions. The two observed abstentions are misses on answerable examples, not correct safety abstentions. `answerability_accuracy` is consequently a coverage-like measure on this split, not evidence of calibrated answerability detection.

The MuSiQue paper describes MuSiQue-Ans as answerable multi-hop questions and MuSiQue-Full as adding unanswerable contrast questions. If the product goal includes “know when evidence is insufficient,” add a labeled unanswerable cohort (or appropriate MuSiQue-Full data) and report false-answer rate, correct-abstention rate, and answerable-case coverage separately. Do not silently call the current answerable-only result abstention evaluation.

### 4. RAGAS scores are not calibrated to this project yet

The evaluator is `qwen3:4b-instruct`, while the answer-relevancy embedding is `nomic-embed-text`. That is reproducible enough to identify the run configuration, but no artifact demonstrates that this judge agrees with a human on this domain. The current output itself is a calibration warning: the incorrect GraphRAG answer receives a very high answer-relevance score, while VectorRAG's correct concise entity answer receives 0.0. Given RAGAS's answer-relevancy construction (generate likely questions from the answer, then compare embeddings), short entity answers are especially worth calibrating rather than assuming the score is intuitive.

Faithfulness and answer relevance are useful diagnostic views; neither replaces a reference answer, direct support labels, citation review, or human judgment. ContextRelevance is an LLM judge over contexts; project-specific gold supporting paragraph recall/precision is a stronger deterministic retrieval anchor for this dataset. The custom implementation already calculates that anchor, which is a major strength.

There is no judge agreement study, annotated adjudication sample, inter-rater agreement, or evaluator stability/repeatability result in the supplied artifacts. RAGAS is an established library, not a certificate that its scores are valid for every model, language, task shape or dataset.

### 5. Retrieval scores do not fully expose graph quality or ranking quality

The custom supporting precision/recall and hop coverage are valuable, and the full evidence is retained. However, the paragraph metrics are set-based. They do not tell whether a supporting paragraph was ranked first or fifth, although rank can materially affect a bounded context. The LLM ContextRelevance adds a ranking-agnostic judged signal but is not a substitute for stable labels. Add a rank-sensitive metric such as MRR or nDCG using explicit per-question relevance judgments (or carefully map MuSiQue support labels to binary relevance and state that simplification).

The graph nodes/edges themselves have no gold relation/path labels in this evaluation. A system may return many plausible-looking entities and edges without being scored on whether those facts are correct, complete, or useful to the question. That is particularly important when graph extraction is LLM-produced. Add graph-specific path/evidence tests or annotated relation/path judgments, and show graph node/edge counts, budget exhaustion, and how many graph facts are actually cited/used. Do not equate a complete traversal with a correct graph.

### 6. “Corpus” is a scope label, not proof of full-corpus coverage

The code's corpus condition searches the entire graph/index that exists at run time; it does not ingest the rest of the dev file. The manifest fingerprints 713 nodes and 979 relationships but does not provide an inventory proving that all 2,417 dev records (or the intended corpus) are loaded and searchable. Thus, `corpus` means “all current indexed graph data,” not “the full MuSiQue dataset.” The record/corpus result equality here is consistent with the same seeds, and says little about behavior on a fully populated corpus. Before interpreting the corpus condition, publish a corpus inventory: record count, paragraph/chunk count, split/source checksums, index state and graph snapshot identifier.

### 7. Latency and cost are incomplete for decision making

The report explicitly says total pipeline timing excludes RAGAS judging. That is honest, but it means latency is not end-to-end evaluation cost, and it is not necessarily user-facing latency if the benchmark invokes the judge sequentially after answers. `provider_cost` is always `null` in `runner.py`; token usage is captured from the answer model only, and judge token/call usage is not recorded in row artifacts. With one local run and three examples, timing is descriptive only. For a cost/performance decision, record per-provider call count, prompt/completion tokens for generator and judge, model identifiers/revisions, retries/timeouts, and repeated latency percentiles after warmup.

### 8. Reproducibility is good but not complete

Good: source checksum and source lines, package versions, model/provider names, retrieval budgets, exact prompts/contexts, response/citations, trace IDs, saved per-row status, and before/after graph fingerprints are recorded. The before/after fingerprints match, so the check found no persistent graph/index change during the run.

Limits: a before/after fingerprint is not transaction-level snapshot isolation, and the code comment says temporary changes that are later reverted can escape detection. Model names can be mutable tags; the manifest does not capture model weight hashes, server/runtime versions, prompt-template hashes as named versions, hardware, or judge seeds. The paired row costs are missing. These gaps do not invalidate the run, but they matter before presenting it as independently reproducible research.

### 9. The report is polished; polish can make weak evidence look stronger

The offline HTML report is a real strength: searchable per-record views, side-by-side backends, explicit score counts, gold contexts, citations, exact prompts/contexts, graph evidence, downloads and warnings. The JSONL records are a better audit source than the chart. The presentation is materially more professional than a notebook screenshot.

The risk is visual authority: clean charts may suggest stable comparative evidence when the RAGAS answer metrics each have `n=1` and the whole run has only three unique questions. The report correctly includes counts, but the overview should foreground unique question count and evidence limitations as strongly as the means. Its title/description should call this a **three-question diagnostic run**, not a performance benchmark result.

## What is right, and what I would change

| Decision already made | Verdict | Why / impact |
|---|---|---|
| Use RAGAS for faithfulness, answer relevance and context relevance | **Agree as one diagnostic layer** | Established and useful for aspects not captured by exact labels. Keep scores separate and show denominators; calibrate the local judge. |
| Add custom MuSiQue exact/alias answer, F1, support, hop and citation checks | **Strong decision; retain** | Gold gives deterministic dataset-grounded evidence that RAGAS alone does not. Expand interpretation and rank/graph diagnostics. |
| Use one paired evaluation command and same answer policy | **Agree** | Reduces obvious confounders and captures row-level traces. Add an ablation because graph context representation also changes. |
| Keep strict evidence/context boundaries and answer validation | **Agree** | This makes the observed comparison auditable and avoids hidden evidence. Retain. |
| Score faithfulness/relevancy only when an answer exists | **Agree with clear reporting** | Undefined answer metrics should stay N/A, not become zero. Always put denominator next to score and pair it with answer coverage/correctness. |
| Run only records 4–6 and use the result to choose VectorRAG vs GraphRAG | **Disagree** | Three records are appropriate for smoke tests. They are too few and currently do not exercise cross-paragraph graph retrieval. |
| Treat record and corpus conditions as two larger samples | **Disagree** | They reuse the same questions and may search the same limited graph. They are separate conditions, not six independent questions. |
| Call the current test a complete GraphRAG-vs-VectorRAG evaluation | **Disagree** | It is a controlled but tiny diagnostic comparison with no graph-added gold support. It cannot validate the primary graph hypothesis. |

## Options for the long-term evaluation solution

No single framework is an industry-wide gold standard for all RAG evaluation. Research benchmarks commonly define a dataset, relevance/reference labels, reproducible systems and per-query comparisons; production teams commonly add curated internal test sets, failure analysis, human review and ongoing telemetry. “Industry standard used by prominent companies” is not a verifiable single product choice, so it would be misleading to name one vendor as the answer.

| Option | Pros | Cons / risks | Fit and impact here |
|---|---|---|---|
| **A. Keep the existing custom harness + RAGAS only** | Smallest change; already fits Neo4j, MuSiQue, labels and JSON/HTML traces; local judge support; low migration risk. | Current judge uncalibrated; no uncertainty/statistical testing; no full graph-specific correctness; can encourage over-reading RAGAS charts. | Good for smoke tests and iteration. **Insufficient alone** for the thesis comparison. |
| **B. Recommended hybrid: existing harness + deterministic benchmark metrics + calibrated human set + paired statistics** | Preserves strong bespoke controls; gold labels answer the known task; human checks tell whether the judge is trustworthy; uncertainty supports defensible claims; vendor-neutral and long-lived. | Requires annotation time and careful metric definitions; larger evaluation costs more; confidence intervals do not repair a biased or unrepresentative dataset. | **Best fit and most robust path for this project.** Keeps RAGAS as one signal, not the arbiter. |
| **C. Adopt RAGChecker as the primary replacement** | Research-backed, fine-grained retriever/generator diagnosis; can reveal failure patterns hidden by one aggregate score. | Does not remove need for MuSiQue-specific gold checks, graph-path labels, judge calibration or custom Neo4j conditions; adds dependency/metric compatibility work. | Useful as a **secondary diagnostic cross-check** after validating its metric assumptions, not a drop-in replacement. |
| **D. Adopt ARES-style evaluation** | Research approach explicitly combines specialized judges with a small human-labeled set and prediction-powered confidence intervals; directly addresses evaluator uncertainty. | Requires setup/data/annotation and model-training or judge infrastructure; heavier than this capstone needs immediately; confidence guarantees depend on assumptions and a suitable representative human validation sample. | Strong future research path if calibrated aggregate judge scores are a core contribution. Borrow its calibration/uncertainty principle even if not its whole stack. |
| **E. Move everything to a hosted eval/observability platform** | Convenient dashboards, traces, collaboration and potentially production monitoring. | Data egress/privacy, recurring cost, lock-in, custom graph/gold logic remains; dashboard does not make a weak benchmark valid. No evidence this is required by the current project. | Consider only if team collaboration or production tracing becomes a real requirement. Not the next research investment. |

### Recommendation, in plain terms

**Do not throw away RAGAS; stop asking it to answer questions it cannot answer alone.** The best long-term setup is a hybrid evaluation stack:

1. Use MuSiQue's known answer aliases and paragraph support labels for deterministic answer and retrieval measures.
2. Keep RAGAS faithfulness/relevance as judge-based diagnostics, but compare a stratified sample of judge results with human ratings before trusting aggregate scores.
3. Evaluate GraphRAG on graph-relevant cases where a correct path can add evidence across distinct chunks; separately test graph extraction/path correctness.
4. Report paired per-question differences with confidence intervals and an appropriate paired test; use the same query items for both systems and preserve abstentions/errors in stated denominators.
5. Repeat model runs where generation/judging variation matters; pin model versions and prompts, record all tokens/calls/costs, and publish corpus inventory/checksums.

For the academic claim, use the **full appropriate labeled development cohort** for iteration and a held-out set for final reporting where labels are available. The answerable-only MuSiQue-Ans split cannot substantiate abstention performance; add unanswerable examples. If runtime cost makes the whole set impractical, draw and document a stratified sample by hop count, retrieval success, support coverage and answerability, then report uncertainty and do not call it full-dataset performance.

## Suggested staged plan

| Stage | Work | Exit evidence |
|---|---|---|
| 1. Define the claim | Decide whether the question is “does graph expansion improve paragraph retrieval?”, “does graph representation improve answers?”, or both. Define the primary metric and acceptable abstention policy before looking at more results. | One-page protocol with primary outcome, cohorts, exclusions, denominators and analysis plan. |
| 2. Fix benchmark scope | Inventory loaded graph records/chunks and index; select a sufficiently broad, representative MuSiQue cohort; include unanswerable examples if abstention is in scope. | Dataset and graph manifests reconcile selected gold paragraphs to persisted chunks; corpus scope is explicit. |
| 3. Separate components | Compare vector seed retrieval first; then graph expansion; then answer generation using matched contexts. Add a graph-off/graph-lines-only ablation to isolate graph facts from text changes. | Per-query retrieval and answer artifacts show which stage changed outcomes. |
| 4. Add diagnostic labels | Use paragraph recall/precision plus rank-sensitive retrieval metrics; annotate graph relations/paths for a targeted multi-hop subset; inspect citations against support. | Graph-added evidence has a correctness/usefulness score, not only node/edge counts. |
| 5. Validate the judge | Have humans independently rate a stratified sample for support, relevance and answer correctness; resolve disagreements; compare judge-human agreement and disagreement cases. | Calibration report with rubric, sample makeup, agreement, uncertainty and examples. |
| 6. Analyze paired outcomes | Use paired per-question deltas, confidence intervals and paired significance analysis appropriate to the sample/metric; account for repeated conditions and multiple metrics. | Effect size and uncertainty accompany each claim; no winner selected from the score mean alone. |
| 7. Harden reproducibility | Pin model revision/runtime and prompts; capture all provider usage, retries, costs and warm/cold timing; retain snapshot and corpus inventory. | Another researcher can reconstruct inputs/settings and explain discrepancies. |

## Assumptions and how they were checked

| Assumption | Meaning | Verification / confidence |
|---|---|---|
| Part 3 is the run to critique | The user pointed to `runs/eval-prepared-4-to-6-part-3/report.html` as the last run. | Read its manifest, summary, records and report. High. |
| The source dataset is the answerable dev JSONL named in the request | It defines the current evaluation population. | `records.py` default and eval guide point to that file; the guide says 2,417 records and all answerable. Official MuSiQue literature distinguishes MuSiQue-Ans from MuSiQue-Full. High for run configuration; exact current file row distribution was not independently rescanned in this review. |
| Current graph structure has no cross-paragraph joins | GraphRAG cannot use entity links to bridge different source chunks in the evaluated graph. | Explicit in `documentation.md`, the October task completion log, and task plan: entity resolution disabled and paragraph components remain separate. High for documented project state. |
| The run itself has only three questions | It cannot provide population-level confidence. | Manifest has three `selected_records`; records JSONL has 12 rows over those IDs. High. |
| Graph “no added gold support” is an observed outcome, not a retrieval bug | It follows from row-level evidence and the documented current component design. | Paired seeds match; `graph_added_supporting_chunk_ids` is empty; support/hop metrics are identical. High for these rows. |
| The graph snapshot proves no persistent graph change, not perfect isolation | Persistent before/after hash equality is a useful guard but not transactional isolation. | Manifest hashes/counts match; `runtime.py` documents the limitation directly. High. |
| RAGAS scores need local calibration | Judge metrics can disagree with task-specific gold or human judgments. | One concrete disagreement pattern appears in the run; no human labels/analysis are present in the supplied artifacts. High that calibration is missing; no claim made about the judge's true accuracy beyond this sample. |
| “Industry standard” is not one universal tool | There is no single verified product that makes the benchmark scientifically valid by itself. | RAGAS, ARES and RAGChecker are distinct published approaches with differing tradeoffs; their papers/docs do not establish one universal standard. High. |
| Timing is not a production performance estimate | Three local rows and no repeat/warmup distribution cannot estimate typical service latency. | `runner.py` timing excludes RAGAS evaluation; report labels it; no repeated trials. High. |

## What is built in already?

There are two senses of “built in” here:

- **In this repository:** yes. `graphrag-eval` is already an end-to-end custom command for prepare/check/run/report, with a RAGAS integration and MuSiQue-specific gold checks. The HTML and JSONL artifacts are project-specific infrastructure, not a notebook prototype.
- **As an established external solution:** RAGAS supplies reusable metrics, but it does not automatically know this project's Neo4j snapshot, paired seed constraints, MuSiQue paragraph identity, graph traversal semantics, or graph-added support labels. ARES and RAGChecker are established alternatives/companions for different parts of the problem, but neither makes the local dataset and graph-specific protocol disappear.

So there is no need to replace everything to get a “professional” evaluation. The robust solution is to preserve the existing domain-specific harness and improve the validity of its test population, graph treatment, judge calibration, uncertainty analysis and evidence coverage.

## Source and code evidence map

- [Evaluation protocol and completion log](tasks-goals-17-oct-2026.md): original requirements, acceptance criteria and completed implementation records.
- [Task 11 handoff](task-11-remainder-plan.md) and [Task 13 handoff](task-13-remainder-plan.md): ingestion/recovery scope; these validate upstream engineering work but do not validate the statistical power of the RAGAS experiment.
- [Operational evaluation guide](../vectorrag_vs_graphrag_evaluation_ragas.md): dataset scope, CLI behavior, provider setup, score interpretation and limitations.
- [Project documentation](../documentation.md) and [README](../README.md): architecture, current graph-component limitation, and user-facing workflows.
- [Runner](../src/graphrag_pipeline/evaluation/runner.py): paired execution, saved traces, aggregation and timing.
- [Scoring](../src/graphrag_pipeline/evaluation/scoring.py): custom answer/support/hop/citation metrics and RAGAS scoring boundaries.
- [Evaluator provider construction](../src/graphrag_pipeline/evaluation/providers.py): RAGAS metric/client and local judge configuration.
- [Gold record and selection model](../src/graphrag_pipeline/evaluation/records.py): supported source and no-label-leak query construction.
- [Offline report validation and generation](../src/graphrag_pipeline/evaluation/report.py): artifact checks and summary recomputation.
- Latest run: [`manifest.json`](../runs/eval-prepared-4-to-6-part-3/manifest.json), [`records.jsonl`](../runs/eval-prepared-4-to-6-part-3/records.jsonl), [`summary.json`](../runs/eval-prepared-4-to-6-part-3/summary.json), [`report.html`](../runs/eval-prepared-4-to-6-part-3/report.html).

## External references checked

- RAGAS [available metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/) and its [v0.4.3 migration documentation](https://docs.ragas.io/en/v0.4.3/howtos/migrations/migrate_from_v03_to_v04/): confirms current metric families and that RAGAS includes both judge-based and reference-based options.
- RAGAS [Answer Relevancy description](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/): explains reverse-question generation plus embedding similarity, and explicitly distinguishes relevance from factual accuracy. This illustrates why the 0.982 score does not mean the GraphRAG answer is correct.
- Original [MuSiQue paper](https://arxiv.org/abs/2108.00573): describes MuSiQue-Ans and the added unanswerable contrast set MuSiQue-Full.
- Published [ARES paper](https://aclanthology.org/2024.naacl-long.20.pdf): combines specialized judges with human-annotated validation and prediction-powered inference, a relevant model for judge calibration and uncertainty.
- [RAGChecker paper](https://arxiv.org/abs/2408.08067) and [Amazon Science implementation](https://github.com/amazon-science/RAGChecker): a fine-grained diagnostic option for retrieval and generation, not a replacement for project-specific graph labels.
- Information retrieval evaluations commonly compare systems on per-query scores; the literature discusses paired significance analysis and query-level resampling. See this [IR significance-testing study](https://arxiv.org/abs/1905.11096). The key point for this review is not which test to choose at n=3; it is that the current sample is far too small for a persuasive estimate.

## Final judgement

**I agree with the architecture of the evaluator more than I agree with any conclusion someone might draw from this run.** The harness has strong professional instincts: controls, traces, explicit N/A values, durable evidence, and honest documentation of limitations. The scores do not support “GraphRAG is better,” “VectorRAG is better,” or “RAGAS validated the system.” They support a much narrower conclusion: on these three answerable questions, under this local Qwen configuration and this currently disconnected-per-paragraph graph, the vector-only answer path got one answer exactly right, the graph-context answer path got none exactly right, both retrieved the same five seeds, graph expansion added no gold support paragraphs, and two questions were missed by both.

That is an uncomfortable result, but it is useful. The next work should make the experiment representative and make GraphRAG's intended capability observable before optimizing the chart or selecting a winner.
