### Paired differences

Each number is the mean of 300 question-level differences, left minus right. The interval is a 95% bootstrap of that mean, using the same seed as the MVP runner. An interval that stays above 0 separates the systems in favor of the left condition. An interval that contains 0 does not.

Recall@2, Recall@5, and later-hop Recall@5 are absent because the saved prediction files do not store the retrieved paragraph sets.

#### `walk_exact` minus `vector`

| Metric | Mean difference | 95% CI | Left higher | Tie | Right higher |
| :--- | ---: | :--- | ---: | ---: | ---: |
| answer_f1 | +0.032 | [+0.002, +0.064] | 23 | 262 | 15 |
| answer_em | +0.040 | [+0.007, +0.073] | 19 | 274 | 7 |
| answer_containment | +0.037 | [+0.003, +0.070] | 20 | 271 | 9 |
| support_f1 | +0.022 | [-0.001, +0.047] | 37 | 239 | 24 |

#### `walk_exact` minus `bm25`

| Metric | Mean difference | 95% CI | Left higher | Tie | Right higher |
| :--- | ---: | :--- | ---: | ---: | ---: |
| answer_f1 | +0.184 | [+0.132, +0.239] | 93 | 176 | 31 |
| answer_em | +0.187 | [+0.137, +0.240] | 71 | 214 | 15 |
| answer_containment | +0.197 | [+0.147, +0.253] | 73 | 213 | 14 |
| support_f1 | +0.142 | [+0.104, +0.186] | 107 | 159 | 34 |

#### `vector` minus `bm25`

| Metric | Mean difference | 95% CI | Left higher | Tie | Right higher |
| :--- | ---: | :--- | ---: | ---: | ---: |
| answer_f1 | +0.152 | [+0.104, +0.203] | 81 | 193 | 26 |
| answer_em | +0.147 | [+0.100, +0.200] | 59 | 226 | 15 |
| answer_containment | +0.160 | [+0.110, +0.213] | 63 | 222 | 15 |
| support_f1 | +0.120 | [+0.082, +0.158] | 100 | 166 | 34 |

#### `walk_off` minus `vector`

| Metric | Mean difference | 95% CI | Left higher | Tie | Right higher |
| :--- | ---: | :--- | ---: | ---: | ---: |
| answer_f1 | +0.000 | [+0.000, +0.000] | 0 | 300 | 0 |
| answer_em | +0.000 | [+0.000, +0.000] | 0 | 300 | 0 |
| answer_containment | +0.000 | [+0.000, +0.000] | 0 | 300 | 0 |
| support_f1 | +0.000 | [+0.000, +0.000] | 0 | 300 | 0 |
