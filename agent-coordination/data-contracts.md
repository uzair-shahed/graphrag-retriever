# Data Contracts

## Purpose

This document defines the boundaries between the components in the current GraphRAG system. A data contract should answer:

1. Which component owns this boundary?
2. What input does it accept?
3. What output does it produce?
4. How is the output validated?
5. What provenance must be preserved?
6. What happens when processing fails?

The current implementation is local, linear, and based on MuSiQue paragraphs as the initial chunk unit. LangGraph orchestrates agent state transitions; Neo4j stores graph data and evidence.

## Architecture Scope

```text
MuSiQue JSONL
  -> Normalizer
  -> Chunker
  -> Extraction Agent
  -> Entity Resolution Agent
  -> Neo4j Knowledge Graph

User Question
  -> Answering Agent
  -> Retriever Agent
  -> Neo4j Evidence
  -> Answering Agent
  -> Reflection Agent
  -> Grounded Response
```

## How To Complete This File

For each unfinished contract:

- replace every `<placeholder>` with a project decision;
- include one small JSON example for the input and output;
- state which fields are required and which are optional;
- describe validation and failure behavior;
- link each output back to its source record or chunk;
- keep the contract narrower than the implementation details.

Do not begin by designing every possible future feature. Define the smallest contract that allows the current local pipeline to run and be tested.

## Contract 1: MuSiQue Record To Application Chunk

This is the completed example. It defines the boundary between the dataset adapter and the rest of the application.

### Owner

- **Producer:** MuSiQue adapter
- **Consumer:** extraction pipeline, fixture tests, and later graph ingestion

### Purpose

Convert one official MuSiQue JSONL record into validated application chunks without changing the original question or paragraph text. Initially, one MuSiQue paragraph becomes one application chunk.

### Input

The adapter accepts one official MuSiQue record. Labeled train/dev records may contain answer and supervision fields; test records may omit them.

```json
{
  "id": "2hop__460946_294723",
  "question": "<question text>",
  "question_decomposition": [
    {
      "id": 259228,
      "question": "<constituent question>",
      "answer": "<intermediate answer>",
      "paragraph_support_idx": 14
    },
    {
      "id": 793698,
      "question": "<question using the previous answer>",
      "answer": "<intermediate answer>",
      "paragraph_support_idx": 11
    }
  ],
  "answer": "<gold answer>",
  "answer_aliases": ["<accepted alias>"],
  "answerable": true,
  "paragraphs": [
    {
      "idx": 0,
      "title": "<Wikipedia title>",
      "paragraph_text": "<paragraph text>",
      "is_supporting": false
    },
    {
      "idx": 3,
      "title": "<supporting Wikipedia title>",
      "paragraph_text": "<supporting paragraph text>",
      "is_supporting": true
    }
  ]
}
```

This is the complete labeled-record shape used by train/dev examples. Test records may contain only `id`, `question`, and `paragraphs`; the adapter must accept that reduced test shape without inventing missing labels.

The adapter emits one application chunk for each paragraph. It preserves the other record-level fields for later evaluation and provenance, but does not copy all of them into every chunk.

### Output

The adapter emits one application chunk for each paragraph.

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "document_id": "musique:document:<normalized-title>",
  "paragraph_idx": 3,
  "title": "<Wikipedia title>",
  "text": "<unchanged paragraph text>",
  "is_supporting": true,
  "source_dataset": "musique",
  "source_split": "dev"
}
```

### Required fields

- `record_id`
- `chunk_id`
- `document_id`
- `paragraph_idx`
- `title`
- `text`
- `source_dataset`
- `source_split`

`is_supporting` is required for labeled records and should be `null` or omitted when the source split does not provide that label.

`document_id` identifies the source Wikipedia document derived from the paragraph title. It must not contain the paragraph index. `chunk_id` identifies the specific paragraph evidence unit within the MuSiQue record and may therefore contain the record ID and paragraph index.

### Validation rules

- The input must be valid JSON.
- `id` must be present and non-empty.
- `question` must be present for a usable record.
- `paragraphs` must be a list.
- Each paragraph must have an integer `idx`, a title, and paragraph text.
- When present, `question_decomposition` must be a list of structured decomposition objects.
- When present, `answer_aliases` must be a list of strings.
- When present, `answerable` must be a boolean.
- The original paragraph text must not be silently rewritten.
- `chunk_id` must be deterministic for the same dataset, record, and paragraph index.
- A test record without answer labels must not be treated as a labeled evaluation record.

### Failure behavior

Malformed records should be reported with the source file, line number, record ID if available, and error category. The adapter should skip or dead-letter the malformed record according to the ingestion command's configured policy; it should not invent missing text or identifiers.

### Provenance

The emitted chunk must preserve:

- dataset name;
- source split;
- source record ID;
- official paragraph index;
- paragraph title;
- extraction or ingestion run ID when generated by a run.

### Questions for you

When you revise this example, decide:

- Should `is_supporting` be `null` or omitted for test records?
- Should duplicate paragraph indices reject the entire record or only the duplicate paragraph?
- What exact error categories will your implementation use?
- Where will the source file checksum and line number be stored?

## Contract 2: Chunk To Extraction Agent

**Status:** TODO

### Owner

- **Producer:** MuSiQue Adapter
- **Consumer:** Extractor System

### Purpose

Describe how one validated chunk is submitted to the extraction agent and how validated extraction results are returned.

### Input shape

The extractor's **semantic input** is the paragraph text and its source context. The pipeline may carry additional metadata alongside that input for routing and provenance, but gold labels such as `is_supporting` must not be included in the extractor prompt or model-visible payload. Otherwise, the extractor could use evaluation labels that would not be available in a real deployment.

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "document_id": "musique:document:<normalized-title>",
  "paragraph_idx": 3,
  "text": "<chunk text>",
  "title": "<paragraph title>",
  "provenance": {
    "source_dataset": "musique",
    "source_split": "dev"
  }
}
```

The minimum model-visible input is therefore `text` plus enough stable identity and context to trace the result: `chunk_id`, `record_id`, `document_id`, `paragraph_idx`, and optionally `title`. `is_supporting` is retained in the dataset/chunk store for later evaluation, but is not sent to the extraction agent. `extraction_run_id` belongs in the surrounding execution envelope and logs rather than in the extraction prompt.

Complete this section by deciding:

- Which fields are required?
- What entity fields must the agent return?
- What relationship fields must the agent return?
- How are evidence spans represented?
- What confidence range is allowed?
- How are malformed model responses handled?
- How many corrective retries are allowed?
- What is written to a dead-letter result after retries fail?

### Output shape

```json
{
  "chunk_id": "<stable chunk ID>",
  "entities": [
    {
      "mention": "<entity mention>",
      "entity_type": "<type>",
      "surface_span": "<source span>"
    }
  ],
  "relations": [
    {
      "source_mention": "<source>",
      "relationship_type": "<relationship>",
      "target_mention": "<target>",
      "evidence_text": "<supporting text>",
      "confidence": 0.0
    }
  ]
}
```

## Contract 3: Extraction Output To Entity Resolution

**Status:** TODO

### Owner

- **Producer:** `<extraction agent or validation layer>`
- **Consumer:** `<entity-resolution agent>`

### Purpose

Define how extracted mentions and relationships are passed to entity resolution before graph writes.

### Questions for you

- Does resolution receive individual mentions, complete triples, or both?
- What candidate entities are supplied to the agent?
- Which component performs candidate retrieval?
- What information explains a match or non-match decision?
- What are the allowed decisions: `match`, `non_match`, `uncertain`, `abstain`, or another set?
- How are conflicting properties represented?

### Input shape

```json
{
  "chunk_id": "<chunk ID>",
  "mentions": [
    {
      "mention": "<text>",
      "entity_type": "<type>",
      "context": "<optional context>"
    }
  ],
  "candidate_entities": [
    {
      "canonical_entity_id": "<candidate ID>",
      "name": "<candidate name>",
      "entity_type": "<type>",
      "similarity": 0.0
    }
  ]
}
```

### Output shape

```json
{
  "chunk_id": "<chunk ID>",
  "decisions": [
    {
      "mention": "<mention>",
      "canonical_entity_id": "<ID or null>",
      "decision": "<match|non_match|uncertain|abstain>",
      "confidence": 0.0,
      "rationale": "<short rationale>",
      "provenance": {
        "source_chunk_id": "<chunk ID>"
      }
    }
  ]
}
```

## Contract 4: Entity Resolution To Neo4j Storage

**Status:** TODO

### Owner

- **Producer:** `<validated graph-write layer>`
- **Consumer:** `<Neo4j writer/database>`

### Purpose

Define the validated graph operations that may reach Neo4j. The LLM should not write raw, unvalidated output directly to the database.

### Questions for you

- What node labels will exist initially?
- What relationship types will exist initially?
- What stable property uniquely identifies an entity?
- Which constraints and indexes are required?
- Which properties belong on nodes, relationships, or both?
- How will `MERGE` operations remain idempotent?
- What should happen when a relationship conflicts with an existing property?
- Which transaction failures are retried?

### Input shape

```json
{
  "run_id": "<run ID>",
  "entities": [
    {
      "canonical_entity_id": "<entity ID>",
      "name": "<name>",
      "entity_type": "<type>",
      "provenance": ["<chunk ID>"]
    }
  ],
  "relationships": [
    {
      "source_entity_id": "<entity ID>",
      "relationship_type": "<type>",
      "target_entity_id": "<entity ID>",
      "source_chunk_id": "<chunk ID>",
      "confidence": 0.0
    }
  ]
}
```

### Output shape

```json
{
  "run_id": "<run ID>",
  "status": "<committed|rolled_back|failed>",
  "created_entity_ids": ["<ID>"],
  "created_relationship_ids": ["<ID>"],
  "errors": []
}
```

## Contract 5: User Question To Retriever Agent

**Status:** TODO

### Owner

- **Producer:** `<Answering Agent or query entry point>`
- **Consumer:** `<Retriever Agent>`

### Purpose

Define how the answering workflow requests graph evidence for a user question.

### Questions for you

- What is the exact query input?
- Is retrieval mode fixed to GraphRAG for the current implementation?
- What hop limit and result limit are allowed?
- Does the retriever return nodes, relationships, paths, chunks, or all of them?
- How are empty results represented?
- Can retrieval modify Neo4j, or is it read-only?

### Input shape

```json
{
  "query": "<natural-language question>",
  "max_hops": 0,
  "top_k": 0,
  "trace_id": "<trace ID>"
}
```

### Output shape

```json
{
  "trace_id": "<trace ID>",
  "evidence": [
    {
      "path": ["<entity>", "<relationship>", "<entity>"],
      "source_chunk_ids": ["<chunk ID>"],
      "properties": {}
    }
  ],
  "status": "<evidence_found|no_evidence|error>"
}
```

## Contract 6: Retriever Evidence To Answering Agent

**Status:** TODO

### Purpose

Define how retrieved graph evidence becomes the context for answer generation.

### Questions for you

- What does the Answering Agent receive besides evidence?
- Must every draft claim reference one or more evidence IDs?
- How are contradictory paths represented?
- What should the Answering Agent return when evidence is insufficient?
- Which model and prompt version are recorded?

### Input shape

```json
{
  "query": "<original question>",
  "evidence": [],
  "trace_id": "<trace ID>"
}
```

### Output shape

```json
{
  "trace_id": "<trace ID>",
  "draft_answer": "<draft answer>",
  "claims": [
    {
      "claim_id": "<claim ID>",
      "text": "<claim text>",
      "evidence_ids": ["<evidence ID>"]
    }
  ],
  "status": "<drafted|insufficient_evidence|error>"
}
```

## Contract 7: Answering Agent To Reflection Agent

**Status:** TODO

### Purpose

Define how the draft answer and its claims are checked before delivery.

### Questions for you

- Which claims must be verified against Neo4j?
- Does reflection query Neo4j directly or request verification through another tool?
- What statuses are possible?
- What causes a retry to retrieval or answering?
- What is the maximum retry count?
- What happens when the retry budget is exhausted?

### Input shape

```json
{
  "trace_id": "<trace ID>",
  "query": "<original question>",
  "draft_answer": "<draft answer>",
  "claims": [],
  "retry_count": 0
}
```

### Output shape

```json
{
  "trace_id": "<trace ID>",
  "status": "<grounded|retry_required|insufficient_evidence|contradictory_evidence>",
  "final_answer": "<answer or null>",
  "verified_claims": [],
  "citations": ["<chunk ID>"]
}
```

## Contract 8: LangGraph Shared State And Transitions

**Status:** TODO

### Purpose

Define the state object and transition rules that LangGraph uses to coordinate the agents.

### Questions for you

- Which fields are present from the start of a run?
- Which fields are added by retrieval, answering, and reflection?
- Which fields may be overwritten, and which are append-only?
- Which node handles each transition?
- What condition terminates the graph?
- How are retry counts incremented and bounded?

### State shape

```python
class GraphRAGState(TypedDict):
    query: str
    trace_id: str
    evidence: list[dict]
    draft_answer: str | None
    claims: list[dict]
    citations: list[str]
    reflection_status: str | None
    retry_count: int
    final_answer: str | None
```

### Transition table

| Current state | Event/condition | Next node | Data changed |
|---|---|---|---|
| `<state>` | `<condition>` | `<node>` | `<fields>` |
| `<state>` | `<condition>` | `<node>` | `<fields>` |

## Contract 9: Future Evaluation Inputs And Outputs

**Status:** TODO - defer detailed completion until the implementation is more mature.

### Purpose

Define the artifacts that the future evaluation phase will consume and produce without making evaluation part of the current interim architecture diagram.

### Questions for you

- Which frozen MuSiQue records form the evaluation query suite?
- What gold facts and expected paths are stored?
- What per-query fields are required for reproducibility?
- Which metrics are calculated from saved predictions?
- How are model, prompt, threshold, seed, latency, token usage, and errors recorded?
- Which data must never come from the unlabeled test split?

### Input shape

```json
{
  "query_id": "<query ID>",
  "question": "<question>",
  "gold_answer": "<answer when labels are available>",
  "gold_chunk_ids": ["<chunk ID>"],
  "gold_paths": [],
  "system_output": {},
  "run_metadata": {}
}
```

### Output shape

```json
{
  "experiment_id": "<experiment ID>",
  "metrics": {
    "retrieval_recall_at_k": 0.0,
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "latency_p50_ms": 0.0,
    "latency_p95_ms": 0.0
  },
  "artifacts": ["<raw output path>", "<report path>"]
}
```

## Contract Completion Checklist

- [ ] Contract 1 reviewed against an actual MuSiQue record.
- [ ] Contract 2 completed and tested with a deterministic extraction response.
- [ ] Contract 3 completed with match, non-match, uncertain, and abstain examples.
- [ ] Contract 4 completed with initial labels, relationships, constraints, indexes, and idempotent write rules.
- [ ] Contract 5 completed with bounded retrieval behavior and empty-result handling.
- [ ] Contract 6 completed with claim-to-evidence requirements.
- [ ] Contract 7 completed with verification, retry, and termination rules.
- [ ] Contract 8 completed with LangGraph nodes and transition conditions.
- [ ] Contract 9 completed after the evaluation dataset and protocol are frozen.
- [ ] Every contract names an owner, consumer, validation rule, provenance requirement, and failure behavior.
- [ ] Every contract has at least one passing test or fixture case.
