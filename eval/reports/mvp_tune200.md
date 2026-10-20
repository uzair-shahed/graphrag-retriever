# MuSiQue MVP Comparative Evaluation Report

- **Slice:** `ans_dev_mvp_v1`
- **Split:** `tune` (Evaluated Questions: 200)
- **Evidence Budget:** Top 5 paragraphs (Seeds: 2, Fan-out: 2)

| Condition | Answer F1 (95% CI) | Answer EM (95% CI) | Support F1 (95% CI) | R@2 (95% CI) | R@5 (95% CI) | Later-hop R@5 (95% CI) | Latency (s) | Cost ($/q) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 11.6 [11.6, 11.6] | 0.0 [0.0, 0.0] | 37.9 [37.9, 37.9] | 33.5 [29.3, 37.5] | 49.3 [45.5, 53.4] | 37.5 [32.1, 43.0] | 1.38 | $0.0010 |
| `vector` | 15.1 [15.1, 15.1] | 0.0 [0.0, 0.0] | 53.4 [53.4, 53.4] | 45.7 [41.7, 49.7] | 69.8 [66.2, 73.5] | 65.2 [60.3, 71.1] | 2.20 | $0.0010 |
| `walk_off` | 15.1 [15.1, 15.1] | 0.0 [0.0, 0.0] | 53.4 [53.4, 53.4] | 45.7 [41.7, 49.7] | 69.8 [66.2, 73.5] | 65.2 [60.3, 71.1] | 0.00 | $0.0010 |
| `walk_exact` | 16.4 [16.4, 16.4] | 0.5 [0.5, 0.5] | 52.8 [52.8, 52.8] | 45.7 [41.7, 49.7] | 72.6 [68.4, 76.7] | 69.4 [63.7, 74.9] | 1.37 | $0.0010 |

### Methodology & Verification
- **BM25:** Candidate paragraph keyword matching per question.
- **Vector:** Cosine similarity over candidate document embeddings scoped to question context.
- **Walk Off:** Graph walk executed with entity linking disabled (ablation baseline).
- **Walk Exact:** Graph walk traversing exact-name `SAME_AS` entity clusters.
- **Support Grounding:** Citations strictly validated against retrieved paragraphs.
- **Zero Cross-Question Leakage:** Scoped to each question's candidate context by construction.