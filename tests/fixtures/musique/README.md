# MuSiQue minimal fixture

This directory contains a small, project-specific MuSiQue fixture designed to exercise the core ingestion and validation path without requiring the full dataset or paid model calls.

## Purpose

The fixture is intentionally small but covers the edge cases the project needs in the earliest implementation:

- valid multi-hop records
- distracting paragraphs
- alternate entity mentions and aliases
- malformed JSON input
- missing or partially missing fields
- insufficient / unanswerable cases

The fixture is meant to validate the adapter, chunking, extraction, and validation contracts before the full MuSiQue dev/train files are used.

## Location

- `valid_records.jsonl` contains valid MuSiQue-shaped records.
- `invalid_records.jsonl` contains malformed or schema-invalid records.
- `fixture_manifest.json` describes each case and the expected result.

## Schema shape

The fixture follows the official MuSiQue record structure closely:

```json
{
  "id": "fixture_2hop_001",
  "question": "Which city is the headquarters of the company that ... ?",
  "question_decomposition": [
    {
      "id": 101,
      "question": "Which company was founded in ... ?",
      "answer": "Acme Systems",
      "paragraph_support_idx": 1
    }
  ],
  "answer": "Boston",
  "answer_aliases": ["Boston, Massachusetts", "Beantown"],
  "answerable": true,
  "paragraphs": [
    {
      "idx": 0,
      "title": "Acme Systems",
      "paragraph_text": "Acme Systems was founded in 2002.",
      "is_supporting": false
    },
    {
      "idx": 1,
      "title": "Acme Systems Headquarters",
      "paragraph_text": "Acme Systems is headquartered in Boston.",
      "is_supporting": true
    }
  ]
}
```

## Required fixture cases

1. `fixture_valid_multi_hop_001`
   - valid 2-hop question
   - two supporting paragraphs
   - one distractor paragraph

2. `fixture_valid_alias_002`
   - valid question with alternate mention or alias
   - demonstrates entity variation

3. `fixture_valid_unanswerable_003`
   - valid MuSiQue-Full style insufficient-evidence case
   - `answerable: false`

4. `fixture_invalid_malformed_001`
   - not valid JSON
   - should fail during JSON parsing

5. `fixture_invalid_missing_field_002`
   - valid JSON but missing required fields or schema information such as `answerable` or `is_supporting`
   - should fail validation

6. `fixture_invalid_duplicate_para_idx_003`
   - valid JSON but duplicate paragraph indexes
   - should be rejected or dead-lettered by the ingestion layer

## Intended usage

Use the fixture during the earliest local implementation stages to prove that:

- valid records are accepted;
- invalid records are rejected or dead-lettered;
- paragraph-level provenance and chunk IDs remain stable;
- distractors and multi-hop context do not bypass validation;
- insufficient evidence is handled explicitly.

The fixture is not a replacement for the real MuSiQue dev/test splits. It is a small validation harness for the project-specific pipeline.
