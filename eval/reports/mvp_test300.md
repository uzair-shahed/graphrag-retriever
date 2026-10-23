# MuSiQue MVP Comparative Evaluation Report

- **Slice:** `ans_dev_mvp_v1`
- **Split:** `test` (Evaluated Questions: 300)
- **Evidence Budget:** Top 5 paragraphs (Seeds: 2, Fan-out: 2)
- **Source summary:** `eval/reports/mvp_test300.json`, copied from the local runner output `runs/mvp_results/mvp_report.json`. Per-question predictions stay in gitignored `runs/mvp_results/predictions_*.jsonl`.

| Condition | Answer F1 (95% CI) | Answer EM (95% CI) | Containment (95% CI) | Support F1 (95% CI) | R@2 (95% CI) | R@5 (95% CI) | Later-hop R@5 (95% CI) | Latency (s) | Cost ($/q) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 28.0 [23.4, 32.3] | 18.7 [14.3, 23.0] | 22.0 [17.3, 26.7] | 31.2 [27.6, 34.7] | 30.7 [27.8, 33.9] | 47.4 [44.2, 50.9] | 37.1 [32.4, 41.8] | 1.05 | $0.0002 |
| `vector` | 43.2 [38.0, 47.8] | 33.3 [28.3, 38.3] | 38.0 [32.7, 43.0] | 43.1 [39.5, 46.5] | 46.6 [43.8, 49.8] | 67.4 [64.2, 70.7] | 57.4 [52.7, 62.3] | 1.43 | $0.0002 |
| `walk_off` | 43.2 [38.0, 47.8] | 33.3 [28.3, 38.3] | 38.0 [32.7, 43.0] | 43.1 [39.5, 46.5] | 46.6 [43.8, 49.8] | 67.4 [64.2, 70.7] | 57.4 [52.7, 62.3] | 0.03 | $0.0000 |
| `walk_exact` | 46.4 [41.6, 50.9] | 37.3 [32.3, 42.7] | 41.7 [36.7, 46.7] | 45.4 [42.0, 48.7] | 46.6 [43.8, 49.8] | 72.5 [69.1, 76.0] | 68.6 [64.1, 73.1] | 0.86 | $0.0001 |

### Methodology & Verification

- **BM25:** Candidate paragraph keyword matching per question.
- **Vector:** Cosine similarity over candidate document embeddings scoped to question context.
- **Walk Off:** Graph walk executed with entity linking disabled (ablation baseline).
- **Walk Exact:** Graph walk traversing exact-name `SAME_AS` entity clusters.
- **Support Grounding:** Citations are checked against retrieved paragraphs. A citation outside that set fails the answer, so the saved prediction has an empty answer and empty support indexes.
- **Scope:** Each question is scored inside its own candidate context.

`walk_resolved` and hybrid retrieval were not part of this four-condition table.

### How to read this file

- This is the held-out test split. It is one piece of the MVP sign-off. It is not the sign-off itself: there is no `mvp-v1` tag, no paired test, and no classification of failures in this file.
- `predictions_walk_off.jsonl` is byte-identical to `predictions_vector.jsonl`. The runner removes entity clusters before `walk_off`, so the walk has no exact-name bridge to follow. The $0.0000 cost and 0.03s latency are cache replay of the vector answers.
- Later-hop Recall@5 is the separated comparison. `vector` is 57.4 [52.7, 62.3] and `walk_exact` is 68.6 [64.1, 73.1]. Those two intervals do not overlap.
- Answer F1 intervals overlap: `vector` [38.0, 47.8] and `walk_exact` [41.6, 50.9]. Recall@5 intervals also overlap. Recall@2 is 46.6 for `vector`, `walk_off`, and `walk_exact`. This file does not contain a paired question-by-question test.
- The later-hop gap is a result for the exact-name walk. It is not a result for autonomous entity resolution.
