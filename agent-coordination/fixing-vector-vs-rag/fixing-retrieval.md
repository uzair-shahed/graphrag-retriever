# Retrieval recovery, deterministic verdicts, and answerability reporting

Saved: October 20, 2026.

**Status: planning only. Implementation has not started for this plan. The user explicitly requested saving it for later and instructed: “Do NOT implement this right now.”**

This document preserves the proposed plan and the architecture choices discussed
with the user. Saving it does not authorize implementation.

Related documents:

- [RAGAS implementation critique](../ragas-implementation-critique-2026-10-20.md)
- [Evaluation implementation and CLI reference](../../vectorrag_vs_graphrag_evaluation_ragas.md)
- [VectorRAG/GraphRAG explanation](../vector_vs_graph_explanation.md)

## Summary and agreed boundaries

Improve evidence retrieval before changing the graph architecture. Keep entity resolution, entity-link overlays, lexical search, and extraction migrations deferred.

The saved traces establish:

- Both abstained questions lacked required supporting paragraphs. Context truncation did not cause these misses.
- The Learjet graph contains the correct division relation, but the generated answer reverses the conclusion.
- A separate extracted ownership edge reverses the Aerojet/GenCorp relationship. Its exact effect on answer generation remains unproven.

The primary comparison will use **source-text answers for both backends**. Graph relationships remain retrieval inputs and diagnostic artifacts; source-plus-graph answering becomes an explicit ablation.

Without cross-paragraph links, GraphRAG may retrieve no additional source paragraphs. Reports must expose this limitation rather than imply a graph advantage.

## 1. Explicit deterministic verdicts and answerability statistics

Reuse the existing normalization, answer aliases, paragraph identity/text checks, and hop-support checks. Add clearly named verdicts per result:

| Verdict | Pass condition |
|---|---|
| **Gold answer achieved** | A successful answer exactly matches the normalized gold answer or an accepted alias. |
| **Gold context retrieved** | Every labeled supporting paragraph and known hop support is present, with matching paragraph IDs and unchanged source text. |
| **Gold context shown** | The same support requirements hold in the bounded context actually supplied to the answer model. |
| **Answer and context achieved** | Both gold answer matching and model-visible gold-context coverage pass. |

Keep token F1 as a separate partial-overlap metric. Do not introduce substring matching, semantic judging, answer rewriting, or gold-aware retries into these verdicts. Context coverage establishes labeled evidence availability, not proof that the model reasoned correctly.

For future unanswerable records, report correct abstention separately; answer/context achievement is N/A where no corresponding gold target exists.

Add a confusion matrix **per backend and retrieval condition**:

| Gold class | System answers | System abstains |
|---|---|---|
| Answerable | Answerable answered | **False abstention** |
| Unanswerable | **False answer** | Correct abstention |

An incorrect substantive answer still counts as “system answers” in this matrix; gold answer correctness remains separate.

Report counts, false-abstention/false-answer rates, denominators, and classification coverage. Infrastructure failures and missing results remain separate from either predicted class. Absent classes receive N/A rates.

Add these verdicts, matrices, and failure categories to JSON artifacts, HTML tables/charts, and presentation exports. Derive them for existing reports from saved labels and evidence without rerunning models.

## 2. Source-grounded answering and relation-error diagnosis

- Add an explicit answer-context policy: `source-only` for the primary comparison and `source-plus-graph` for diagnostic ablations.
- Apply the same answer prompt, source formatting, context budgets, citation rules, and abstention behavior to both backends.
- Preserve complete graph evidence separately even when graph facts are excluded from the answer prompt.
- Record the presentation policy and prompt/context hashes. Teach report validation to interpret older artifacts using their original rendering policy.
- Add an audit view pairing suspect relationships with their source paragraphs. Record the reversed Aerojet ownership edge as an observed extraction defect; leave other unaudited edges explicitly unreviewed.
- Compare source-only and source-plus-graph answers under matched retrieval settings. Treat changes as evidence of presentation sensitivity, not proof of a model’s internal cause.

This stage will not modify existing edges, silently repair direction, or rebuild the graph. Extraction correction and provenance versioning require a later approved plan.

## 3. Bounded dense follow-up retrieval

Add a separately labeled retrieval profile while retaining the current single-pass baseline.

### Public controls

- `--retrieval-strategy single-pass|dense-followup`, default `single-pass`.
- `--max-followup-searches 0|1|2`, default `2` for the enhanced profile.
- `--answer-context source-only|source-plus-graph`, default `source-only`.

Use shared retrieval/answer components for evaluation and the normal query path, with equivalent controls.

### Exact enhanced procedure

1. Run the original question through the existing dense search with the configured top-k and record/corpus restriction.
2. Request one source-anchored follow-up query from the configured answer-provider/model.
3. Execute it, then allow a second follow-up to use the newly retrieved source text.
4. Stop after two additional searches, or earlier when no valid new query is available.

At top-k 5, this permits at most **three searches and fifteen unique candidate paragraphs** per question/condition. There is no automatic top-k increase or exhaustive-record fallback.

The planner receives only the original question and bounded retrieved source passages/titles. It selects literal anchor text from those inputs and optional relation wording from the original question. Application code validates those spans and constructs the search query. It cannot supply an unseen entity alias or arbitrary free-form query.

Both backends share the same planner executions, query embeddings, search results, candidate order, and budgets. GraphRAG then applies its graph retrieval treatment to those shared candidates.

Deduplicate candidates by source chunk ID. Use deterministic reciprocal-rank fusion with constant 60 and chunk-ID tie-breaking; select context with query coverage first, reserving eligible evidence from each search before filling remaining slots by fused rank. Keep the existing five-chunk/4096-token defaults and whole-paragraph boundaries. Record skipped oversized paragraphs.

Persist every query, its source anchors, retrieved ranks/scores, selection decisions, actual search count, and stop reason. Planner/provider failures remain visible execution errors.

### Integrity boundary

Gold answers, aliases, answerability, supporting flags, gold decomposition, and unretrieved paragraph lists never enter planning, retrieval, selection, or answering. Enforce this through narrow input types and leakage tests. No question-specific fixes, manual aliases, gold-triggered retries, or best-trial selection.

## 4. Validation and acceptance

Deliver and review the changes in this order: deterministic reporting; source-only answering/audit; dense follow-ups; comparative validation and documentation.

For each stage, add focused unit tests and appropriate live Neo4j/Ollama checks before claiming completion:

- **Deterministic tests:** exact aliases, extra incorrect words, missing/altered support, support lost during context selection, both gold classes, failures, empty denominators, and old-artifact compatibility.
- **Retrieval tests:** sequential discovery, invalid/invented anchors, duplicate queries, strict search limits, scope filtering, deterministic selection, and identical paired candidates. Poison or remove evaluation labels and verify retrieval decisions remain unchanged.
- **Browser tests:** verdicts, confusion-matrix counts, N/A handling, search traces, suspect-edge display, and exported charts/PDF layouts.
- **Live runs:** first verify service readiness and current configuration. Compare single-pass/source-plus-graph, single-pass/source-only, and dense-followup/source-only on records 4–6 in new output directories. Repeat trials and retain every result.
- Before drawing broader conclusions, freeze the policy and validate on additional dev records not used to develop it. Record the loaded corpus inventory and before/after graph fingerprints.

Measure whether the formerly missed supporting paragraphs are retrieved and shown, whether false abstentions decrease, and whether answers match gold. If an outcome remains wrong, report the unresolved stage; do not force an answer or mark it fixed.

The current dataset cannot provide live evidence for unanswerable-class performance. Test that branch with synthetic fixtures and display N/A in actual reports until a real negative cohort is separately authorized.

Update the evaluation progress document, README, and operational reference with policies, commands, test evidence, and remaining limitations. No live OpenRouter tests are included.

## Handoff notes

- Continue using only `dataset/musique/data/data/musique_ans_v1.0_dev.jsonl` for real evaluation. No additional dataset was authorized.
- The user selected dense follow-ups only, a shared query plan, source-text answers for the primary comparison, and relation auditing without changing stored extraction results.
- The selected budget is one follow-up phase containing **at most two sequential searches**. The second may use source text returned by the first; the original search makes three total at most. Do not reinterpret this as multiple rounds of two searches each.
- The user will implement an entity-resolution subsystem separately. A “versioned entity-link layer” discussed as an alternative would store separately versioned links between mentions believed to denote the same entity while preserving original paragraph graphs. It is deferred here.
- Application-level isolation can prevent access to gold labels. It cannot prove a pretrained model has never encountered MuSiQue; do not claim that stronger guarantee.
- No implementation or live validation of this proposed architecture was performed while saving this document. Existing report artifacts are diagnostic evidence, not acceptance evidence for the proposed changes.
- Resume only after the user requests implementation; consult before significant architecture changes and validate claimed results with live Neo4j/Ollama runs.
