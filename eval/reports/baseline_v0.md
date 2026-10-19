# Baseline Report v0 · Free Keyword & Random Baselines

## Performance Summary

| System | Setting | Split | R@2 (%) | R@5 (%) | Later-hop R@5 (%) | Support F1 (%) | Answer F1 (%) | Answer EM (%) | Cost/q ($) | Latency p50 (ms) | Latency p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | per_question | tune | 33.50 | 49.33 | 37.69 | 32.99 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |
| BM25 | pooled | tune | 34.75 | 46.04 | 26.75 | 30.30 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |
| Random-5 Floor | per_question | tune | 0.00 | 24.96 | 27.36 | 0.00 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |
| BM25 | per_question | test | 30.75 | 47.44 | 37.70 | 31.81 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |
| BM25 | pooled | test | 34.64 | 45.22 | 22.98 | 29.80 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |
| Random-5 Floor | per_question | test | 0.00 | 25.36 | 24.80 | 0.00 | 0.00 | 0.00 | $0.0000 | 0.0 | 0.0 |

## Paired Comparisons (95% Bootstrap Confidence Intervals)

| System A | System B | Metric | Difference (A - B) | 95% Confidence Interval |
| --- | --- | --- | --- | --- |
| BM25 (per-question, test) | Random-5 (test) | recall_at_5 | +22.08 | [+17.89, +26.53] |
| BM25 (per-question, tune) | Random-5 (tune) | recall_at_5 | +24.38 | [+18.67, +29.96] |

## Evaluation Notes

- Evaluation Slice: ans_dev_mvp_v1 (500 questions: 200 tune, 300 test).
- Setting 1 (per_question): Question searches strictly within its 20 candidate paragraphs (official MuSiQue RC setting).
- Setting 2 (pooled): Question searches across all unique passages in the slice corpus (HippoRAG open-retrieval protocol).
- Cost: $0.00 spend across all baselines.
- Decision rule: Any GraphRAG pipeline or retrieval modification must demonstrate statistically significant improvement over these baselines to justify extraction/embedding cost.
