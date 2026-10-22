# MuSiQue MVP Comparative Evaluation Report

- **Slice:** `ans_dev_mvp_v1`
- **Split:** `tune` (Evaluated Questions: 200)
- **Evidence Budget:** Top 5 paragraphs (Seeds: 2, Fan-out: 2)

| Condition | Answer F1 (95% CI) | Answer EM (95% CI) | Containment (95% CI) | Support F1 (95% CI) | R@2 (95% CI) | R@5 (95% CI) | Later-hop R@5 (95% CI) | Latency (s) | Cost ($/q) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 32.9 [27.5, 38.7] | 24.5 [19.0, 30.5] | 25.0 [19.5, 31.0] | 32.9 [28.5, 37.3] | 33.5 [29.3, 37.5] | 49.3 [45.5, 53.4] | 37.5 [32.1, 43.0] | 1.88 | $0.0002 |
| `vector` | 3.7 [1.7, 6.2] | 3.0 [1.0, 5.5] | 3.0 [1.0, 5.5] | 4.0 [2.0, 6.5] | 4.0 [2.0, 6.5] | 5.5 [2.8, 8.7] | 6.0 [3.0, 9.5] | 0.32 | $0.0000 |
| `walk_off` | 3.7 [1.7, 6.2] | 3.0 [1.0, 5.5] | 3.0 [1.0, 5.5] | 4.0 [2.0, 6.5] | 4.0 [2.0, 6.5] | 5.5 [2.8, 8.7] | 6.0 [3.0, 9.5] | 0.01 | $0.0000 |
| `walk_exact` | 4.1 [1.8, 7.1] | 3.5 [1.0, 6.5] | 3.5 [1.0, 6.5] | 4.0 [1.9, 6.6] | 4.0 [2.0, 6.5] | 5.8 [3.0, 9.2] | 6.0 [3.0, 9.5] | 0.04 | $0.0000 |

### Methodology & Verification
- **BM25:** Candidate paragraph keyword matching per question.
- **Vector:** Cosine similarity over candidate document embeddings scoped to question context.
- **Walk Off:** Graph walk executed with entity linking disabled (ablation baseline).
- **Walk Exact:** Graph walk traversing exact-name `SAME_AS` entity clusters.
- **Walk Resolved:** Graph walk traversing autonomous LLM-resolved entity clusters.
- **Hybrid:** Reciprocal Rank Fusion (RRF) combining BM25 keyword matching, vector similarity, and bounded graph walk.
- **Support Grounding:** Citations strictly validated against retrieved paragraphs.
- **Zero Cross-Question Leakage:** Scoped to each question's candidate context by construction.