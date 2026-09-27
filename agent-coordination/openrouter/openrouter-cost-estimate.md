# OpenRouter Cost Estimate for MuSiQue

**Date:** 25 September 2026

## Scope

This estimate covers only these two datasets:

- `musique_full_v1.0_dev.jsonl`: 4,834 records
- `musique_full_v1.0_train.jsonl`: 39,876 records

**Maximum scope:** 44,710 records

The other MuSiQue files are excluded from this budget.

## Observed Usage Baseline

One run of the project fixture through the Neo4j GenAI Knowledge Graph
Pipeline produced the following dashboard usage:

| Service | Observed usage per pipeline run |
|---|---:|
| GPT-4o-mini | Approximately 2,400 tokens: 2,000 input and 400 output |
| Text Embedding 3 Small | 17 tokens |

The estimate treats one dataset record as one pipeline run. This is the
planning unit requested for the four scenarios below.

The updated dashboard-based usage estimate is approximately 2,400 GPT-4o-mini
tokens per pipeline call, split as follows:

- 2,000 input tokens
- 400 output tokens

Actual costs may differ because the observed values are approximate.

## Pricing

| Service | Input price | Output price |
|---|---:|---:|
| GPT-4o-mini | $0.15 per 1 million tokens | $0.60 per 1 million tokens |
| Text Embedding 3 Small | $0.02 per 1 million tokens | Not applicable |

## Cost Per Record

GPT-4o-mini:

- Input: `2,000 / 1,000,000 x $0.15 = $0.0003000`
- Output: `400 / 1,000,000 x $0.60 = $0.0002400`
- GPT-4o-mini total: **$0.0005400**

Text Embedding 3 Small:

- `17 / 1,000,000 x $0.02 = $0.00000034`

**Estimated total per record: $0.00054034**

## Requested Scenarios

| Scenario | Share of approved data | Records used | GPT-4o-mini cost | Embedding cost | Estimated total |
|---|---:|---:|---:|---:|---:|
| Low | 25% | approximately 11,178 | $6.04 | less than $0.01 | **$6.04** |
| Medium | 50% | 22,355 | $12.07 | $0.01 | **$12.08** |
| High | 75% | approximately 33,533 | $18.11 | $0.01 | **$18.12** |
| Full | 100% | 44,710 | $24.14 | $0.02 | **$24.16** |

The quarter and three-quarter record counts are rounded to the nearest whole
record for operational planning. The cost calculations use the corresponding
percentage of the full 44,710-record total.

## Budget Summary

| Budget level | Recommended amount |
|---|---:|
| Low | $6.04 |
| Medium | $12.08 |
| High | $18.12 |
| Full | $24.16 |

These amounts cover the observed extraction and embedding usage only. They do
not include additional runs caused by retries, failed requests, database resets,
experiments, retrieval-time question processing, answer generation, or
reflection runs.

For a practical spending limit, add a reserve after the first larger sample is
measured. The current full-data estimate should therefore be treated as the
baseline estimate, not a guaranteed maximum charge.
