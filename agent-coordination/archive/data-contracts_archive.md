# Archived Data Contracts

This file preserves the earlier, non-active design drafts for the hand-built extraction, retrieval, answer-generation, reflection, and future-evaluation contracts. They are kept for historical reference only and are not the live implementation contract for the current project.

The active project boundary is the Neo4j GenAI Plugin `SimpleKGPipeline` path documented in the main [agent-coordination/data-contracts.md](../data-contracts.md) file.

---

## Contract 3: Extraction Output To Entity Resolution

**Status:** Specified; implementation and tests remain TODO.

### Owner

- **Producer:** extraction validation layer
- **Consumer:** entity-resolution agent and resolution audit store

### Purpose

Define how extracted mentions and complete triples are passed to entity resolution before graph writes. Resolution maps mentions to existing canonical entities conservatively. The Entity Resolution Agent may invoke the constrained `write_graph` tool, but the tool accepts only validated structured resolution decisions and delegates to deterministic graph-write logic; the agent cannot submit raw Cypher or bypass validation.

The current decisions are:

- Resolution receives both individual mentions and complete extracted triples.
- Candidate retrieval is deterministic application logic, not an unconstrained LLM action.
- Candidates are filtered by compatible entity type and limited to a configured top-K.
- Candidate generation uses exact normalized-name matches first, then lexical similarity and optional embedding similarity.
- Context and graph-neighborhood compatibility are supplied to the resolution agent when available.
- The resolution agent returns `match`, `non_match`, `uncertain`, or `abstain`.
- Only `match` may produce a canonical entity link. `uncertain` and `abstain` preserve an unresolved mention and do not merge it.
- Conflicting properties are retained as separate values with their source chunks and confidence; the resolver must not silently overwrite prior evidence.
- The Entity Resolution Agent may call `write_graph` after producing structured resolution decisions.
- `write_graph` validates those decisions, produces the Contract 4 `graph_write` projection, and invokes the Neo4j writer transactionally.
- Raw Cypher, arbitrary tool arguments, and direct Neo4j access are rejected.

### Input shape

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "chunk_text": "The Identity API validates user credentials.",
  "provenance": {
    "source_dataset": "musique",
    "source_split": "dev",
    "source_file": "musique_ans_v1.0_dev.jsonl",
    "source_sha256": "<file checksum>",
    "source_line": 1842
  },
  "mentions": [
    {
      "mention_id": "mention-1",
      "mention": "Identity API",
      "entity_type": "SoftwareService",
      "context": "The Identity API validates user credentials.",
      "evidence_span": {
        "start": 4,
        "end": 16,
        "text": "Identity API"
      }
    },
    {
      "mention_id": "mention-2",
      "mention": "user credentials",
      "entity_type": "Credential",
      "context": "The Identity API validates user credentials.",
      "evidence_span": {
        "start": 27,
        "end": 43,
        "text": "user credentials"
      }
    }
  ],
  "triples": [
    {
      "source_mention_id": "mention-1",
      "relationship_type": "VALIDATES",
      "target_mention_id": "mention-2",
      "evidence_span": {
        "start": 0,
        "end": 44,
        "text": "The Identity API validates user credentials."
      },
      "confidence": 0.94
    }
  ],
  "candidate_entities": [
    {
      "mention_id": "mention-1",
      "canonical_entity_id": "entity:identity-api",
      "name": "Identity API",
      "entity_type": "SoftwareService",
      "similarity": 0.98,
      "match_sources": ["exact_name", "entity_type"],
      "neighborhood_summary": "VALIDATES Credential entities"
    },
    {
      "mention_id": "mention-1",
      "canonical_entity_id": "entity:authentication-service",
      "name": "Authentication Service",
      "entity_type": "SoftwareService",
      "similarity": 0.78,
      "match_sources": ["lexical", "embedding"],
      "neighborhood_summary": "AUTHENTICATES User entities"
    },
    {
      "mention_id": "mention-2",
      "canonical_entity_id": "entity:user-credentials",
      "name": "User Credentials",
      "entity_type": "Credential",
      "similarity": 0.96,
      "match_sources": ["exact_name", "entity_type"],
      "neighborhood_summary": "VALIDATED_BY SoftwareService entities"
    },
    {
      "mention_id": "mention-2",
      "canonical_entity_id": "entity:access-token",
      "name": "Access Token",
      "entity_type": "Credential",
      "similarity": 0.61,
      "match_sources": ["embedding"],
      "neighborhood_summary": "ISSUED_BY SoftwareService entities"
    }
  ]
}
```

`mention` is the exact surface form found in the source chunk, such as `Identity API`. It is not yet a canonical entity name or Neo4j ID. The resolver uses it, together with its context and entity type, to search for possible canonical entities.

`candidate_entities` stores existing canonical-entity candidates retrieved from the graph or candidate index. It is not a list of entities extracted from the current chunk, and it is not a list of entities that the resolver has already accepted. Each candidate is associated with a `mention_id` so the resolver knows which extracted mention it may match.

Each candidate contains:

- `canonical_entity_id`: the stable ID of an existing graph entity;
- `name`: the entity's current canonical name;
- `entity_type`: the ontology type used for compatibility filtering;
- `similarity`: a normalized candidate score in the range `[0.0, 1.0]`;
- `match_sources`: the retrieval signals that produced the candidate;
- `neighborhood_summary`: optional graph context used by the resolver.

Candidate retrieval may return several candidates for one mention. The resolver chooses a decision from those candidates, or returns `uncertain`/`abstain`; candidate retrieval itself does not create or merge graph entities.

`entity_type` is an ontology category inferred by the extraction system; MuSiQue does not provide it. In this example, `Identity API` is classified as `SoftwareService` because it denotes a technical system, while `user credentials` is classified as `Credential` because it denotes authentication information. Your final ontology may choose different controlled labels, but each label must be defined consistently and validated.

Evidence offsets are zero-based character positions into `chunk_text`, using an exclusive `end` index. For the example:

```text
The Identity API validates user credentials.
  ^-----------^          Identity API: start 4, end 16
               user credentials: start 27, end 43
```

Therefore, `chunk_text[4:16]` is `Identity API` and `chunk_text[27:43]` is `user credentials`. The resolver should obtain the original chunk text from the chunk store or receive it explicitly so it can validate these spans.

The candidate list contains at most the configured top-K candidates, initially recommended as `5`. Candidates below the configured minimum similarity threshold are excluded. If no candidate passes the threshold, the resolver receives an empty candidate list and may return `abstain`.

### Output shape

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "provenance": {
    "source_dataset": "musique",
    "source_split": "dev",
    "source_file": "musique_ans_v1.0_dev.jsonl",
    "source_sha256": "<file checksum>",
    "source_line": 1842
  },
  "decisions": [
    {
      "mention_id": "mention-1",
      "mention": "Identity API",
      "canonical_entity_id": "entity:identity-api",
      "decision": "match",
      "confidence": 0.93,
      "candidate_ids": ["entity:identity-api", "entity:auth-service"],
      "rationale": "Exact normalized name and compatible entity type.",
      "resolution_basis": ["exact_name", "entity_type"],
      "provenance": {
        "source_chunk_id": "musique:2hop__460946_294723:paragraph:3"
      }
    }
  ],
  "graph_write": {
    "run_id": "run-2026-09-15-001",
    "chunk": {
      "chunk_id": "musique:2hop__460946_294723:paragraph:3",
      "document_id": "musique:document:<normalized-title>",
      "title": "<Wikipedia title>",
      "source_split": "dev"
    },
    "entities": [
      {
        "canonical_entity_id": "entity:identity-api",
        "name": "Identity API",
        "entity_type": "SoftwareService",
        "provenance": ["musique:2hop__460946_294723:paragraph:3"]
      },
      {
        "canonical_entity_id": "entity:user-credentials",
        "name": "User Credentials",
        "entity_type": "Credential",
        "provenance": ["musique:2hop__460946_294723:paragraph:3"]
      }
    ],
    "relationships": [
      {
        "source_entity_id": "entity:identity-api",
        "relationship_type": "VALIDATES",
        "target_entity_id": "entity:user-credentials",
        "source_chunk_id": "musique:2hop__460946_294723:paragraph:3",
        "confidence": 0.94
      }
    ],
    "property_observations": [],
    "provenance": {
      "source_dataset": "musique",
      "source_split": "dev",
      "source_file": "musique_ans_v1.0_dev.jsonl",
      "source_sha256": "<file checksum>",
      "source_line": 1842
    }
  }
}
```

The `decisions` list is the auditable resolver result. The `graph_write` projection is produced by deterministic application code from `match` decisions and is the payload passed to Contract 4. `non_match`, `uncertain`, and `abstain` decisions are retained for audit but do not contribute canonical entities or relationships to `graph_write`.

When the Entity Resolution Agent invokes `write_graph`, the tool performs this projection and handoff internally. The agent receives the structured write result, including committed IDs, conflicts, and errors; it never receives permission to submit Cypher directly.

### Candidate decision semantics

- `match`: one candidate is sufficiently supported; link the mention to that canonical ID.
- `non_match`: a supplied candidate was considered and rejected; do not link the mention to it.
- `uncertain`: plausible candidates exist but the evidence does not safely distinguish them; do not merge.
- `abstain`: no safe resolution is available, including when no candidate passes the threshold; do not merge.

`uncertain` and `abstain` decisions are stored for audit and may create a provisional unresolved mention record, but they must not create a canonical entity link. A later run may resolve them with additional context or revised configuration.

### Decision evidence and conflicts

Every decision must record:

- the mention and candidate IDs considered;
- the candidate-generation methods and similarity scores;
- the selected decision and confidence;
- a short rationale;
- the source chunk and dataset provenance.

Conflicting properties are represented as competing observations rather than overwritten values:

```json
{
  "property": "headquarters",
  "observations": [
    {
      "value": "Paris",
      "source_chunk_id": "<chunk A>",
      "confidence": 0.88
    },
    {
      "value": "Lyon",
      "source_chunk_id": "<chunk B>",
      "confidence": 0.81
    }
  ],
  "status": "conflict"
}
```

Conflict resolution is deferred to the graph-write policy in Contract 4. The entity resolver should report the conflict, not silently choose a value.

### Validation and failure behavior

- Input `chunk_id`, `record_id`, and provenance must be present.
- Every mention ID must be unique within the request.
- Every triple reference must point to a supplied mention ID.
- Candidate entity IDs must be unique within a candidate list.
- Candidate entity types must be compatible with the mention type unless the candidate is explicitly marked as type-uncertain.
- Similarity and decision confidence must be numbers in `[0.0, 1.0]`.
- `match` requires a non-null `canonical_entity_id` and at least one candidate ID.
- `non_match`, `uncertain`, and `abstain` must not assign a canonical entity ID.
- Invalid resolver output is rejected and sent to a resolution dead-letter queue after the configured retry policy.

## Contract 4: Entity Resolution To Neo4j Storage

**Status:** The active path is plugin-managed KG creation in Neo4j, and graph inspection is read-only. The local Python app validates the source provenance and then inspects the persisted graph using the actual labels and relationships created by `SimpleKGPipeline`.

The verified runtime contract for the live plugin is: `Document` and `Chunk` nodes are created for source-backed evidence; extracted entities are persisted with their own labels; and the observed relationship types include `FROM_CHUNK`, `FROM_DOCUMENT`, and `ACQUIRED` in the live fixture run. The archived manual `write_graph` contract remains available only as a fallback design reference and is not the active graph-write path.

### Owner

- **Producer:** validated graph-write layer
- **Consumer:** Neo4j writer/database

### Purpose

Define the validated graph operations that may reach Neo4j. The Entity Resolution Agent may invoke `write_graph`, but the tool accepts only validated structured resolution decisions. `write_graph` produces the Contract 4 `graph_write` projection, delegates to the deterministic writer, constructs parameterized Cypher, and executes one transaction. Neither the agent nor the tool accepts raw Cypher from the model.

### Questions for you

The verified graph vocabulary from the active plugin run is:

- `Document`: a source-backed document node carrying dataset and paragraph provenance.
- `Chunk`: a MuSiQue paragraph evidence unit identified by `chunk_id` and storing the original text and embedding.
- `Entity`: plugin-created extracted entity nodes, such as `Company` in the live fixture graph.
- plugin-generated lexical and semantic relationships, including `FROM_CHUNK`, `FROM_DOCUMENT`, and domain relations such as `ACQUIRED`.

Verified relationship types from the live fixture run are:

- `FROM_CHUNK`: source evidence link from the document/chunk to the extracted fact graph;
- `FROM_DOCUMENT`: identity link from a chunk back to the source document;
- `ACQUIRED`: the actual extraction relation created by the plugin for the fixture.

The project must continue to treat the plugin-created graph as the authority for graph shape until a later resolver or schema layer intentionally changes it.

Stable identifiers are:
- `document_id` for source documents;
- `chunk_id` for paragraph evidence units;
- `canonical_entity_id` for canonical entities;
- `mention_id` for unresolved or audit-only mentions.

Each property observation must identify exactly one owner using either `owner_entity_id` or a stable `owner_relationship_id`. The writer must not discard a lower-confidence observation merely because a higher-confidence value already exists.

Required constraints and indexes are:

- uniqueness constraint on `Document.document_id`;
- uniqueness constraint on `Chunk.chunk_id`;
- uniqueness constraint on `Entity.canonical_entity_id`;
- uniqueness constraint on `Mention.mention_id` when provisional mentions are stored;
- uniqueness constraint on `PropertyObservation.observation_id`;
- indexes on `Entity.name`, `Entity.entity_type`, and provenance IDs used for retrieval;
- relationship provenance properties indexed only if later query measurements justify them.

Node properties include identifiers, names/types, source metadata, and provenance. Relationship properties include source chunk, confidence, extraction run, and conflict/observation metadata.

Writes use `MERGE` on stable identifiers and do not use display names as identity. Reprocessing the same validated payload must not create duplicate nodes or relationships.

For conflicting properties, retain competing observations with source chunks, runs, confidence, and optional validity intervals instead of silently overwriting the existing value. The writer may update an aggregate status such as `conflict`, but it must preserve the underlying observations.

Retry only transient connection, transaction, or Neo4j availability failures. Do not retry schema-validation failures without changing the invalid input.

### Input shape

The `write_graph` tool accepts structured resolution decisions, not the final graph projection directly:

```json
{
  "run_id": "run-2026-09-15-001",
  "resolution_decisions": [
    {
      "mention_id": "mention-1",
      "mention": "Identity API",
      "canonical_entity_id": "entity:identity-api",
      "decision": "match",
      "confidence": 0.93,
      "candidate_ids": ["entity:identity-api", "entity:authentication-service"],
      "rationale": "Exact normalized name and compatible entity type.",
      "provenance": {
        "source_chunk_id": "musique:2hop__460946_294723:paragraph:3"
      }
    }
  ],
  "chunk": {
    "chunk_id": "musique:2hop__460946_294723:paragraph:3",
    "document_id": "musique:document:<normalized-title>",
    "title": "<Wikipedia title>",
    "source_split": "dev"
  },
  "provenance": {
    "source_dataset": "musique",
    "source_split": "dev",
    "source_file": "musique_ans_v1.0_dev.jsonl",
    "source_sha256": "<file checksum>",
    "source_line": 1842
  }
}
```

The tool validates `resolution_decisions`, deterministically constructs the `graph_write` projection shown below, and passes only that validated projection to the Neo4j writer. `uncertain`, `abstain`, and `non_match` decisions cannot create canonical entity links. Raw Cypher is not a valid field in this input contract.

The `write_graph` tool returns the transactional result described in the Contract 4 output shape, including committed or updated IDs, conflict details, and errors. The Entity Resolution Agent receives this structured result rather than a database session or Cypher response.

### Write rules

- Every entity must have a non-empty canonical ID, name, entity type, and provenance.
- Every relationship endpoint must refer to an entity in the same validated write set or an existing canonical entity.
- Relationship types must come from the controlled ontology vocabulary; arbitrary model-generated Cypher is rejected.
- `confidence` must be a number in `[0.0, 1.0]`.
- Unresolved `uncertain` and `abstain` decisions are not included in canonical entity relationships. They may be stored as `Mention` records for audit.
- The writer uses parameterized Cypher only.
- The write is atomic: all operations commit together or roll back together.
- The writer records the run ID and source chunk on every created or updated graph fact.
- Property observations are append-only by `observation_id`; reprocessing the same observation must not duplicate it.
- A canonical summary property may be populated only when the write policy has no active conflict; otherwise it remains unset or is marked `conflict`.

### Idempotency and conflict behavior

The writer should use the following identity model:

```cypher
MERGE (entity:Entity {canonical_entity_id: $canonical_entity_id})
ON CREATE SET entity.name = $name,
              entity.entity_type = $entity_type
ON MATCH SET entity.last_seen_run_id = $run_id

MERGE (source:Entity {canonical_entity_id: $source_entity_id})
MERGE (target:Entity {canonical_entity_id: $target_entity_id})
MERGE (source)-[relationship:VALIDATES {source_chunk_id: $source_chunk_id}]->(target)
SET relationship.confidence = $confidence,
    relationship.run_id = $run_id
```

The actual implementation must use a safe relationship-type mapping rather than interpolating untrusted relationship types directly into Cypher. Re-running the same write with the same stable IDs and source chunk must not duplicate graph facts.

Property conflicts should be stored as `PropertyObservation` records with value, source chunk, run ID, confidence, and optional `valid_from`/`valid_to` fields. The writer must not discard the lower-confidence observation merely because a higher-confidence value already exists.

### Conflict policy

A conflict exists when two observations for the same owner and property are incompatible and no validity interval explains the difference. The initial conflict categories are:

- `property_value_conflict`: incompatible values for one entity property;
- `relationship_type_conflict`: incompatible relationship types for the same endpoints and evidence scope;
- `entity_type_conflict`: incompatible ontology types for one canonical entity;
- `temporal_overlap_conflict`: incompatible values whose validity intervals overlap;
- `provenance_conflict`: observations claim incompatible source identity or chunk provenance.

The writer applies this policy:

1. Store every validated observation with its provenance and confidence.
2. If an identical observation already exists, merge only its run/provenance references without duplicating the observation.
3. If different values have non-overlapping validity intervals, store both as historical observations without marking an active conflict.
4. If different values overlap in validity or have no time information, store both and mark the property or relationship `conflict`.
5. Do not select a winner based only on confidence.
6. Do not silently overwrite the canonical summary value. Leave it unset or mark it `conflict` until a separate, explicit resolution policy is applied.
7. Return conflict details in the write result and make them available to retrieval/reflection for later handling.

Conflicts do not automatically roll back an otherwise valid transaction. Malformed observations, invalid IDs, invalid relationship types, and invalid provenance do roll back or dead-letter according to the failure policy.

Example observation storage:

```cypher
MERGE (observation:PropertyObservation {
  observation_id: $observation_id
})
SET observation.property_name = $property_name,
    observation.value = $value,
    observation.source_chunk_id = $source_chunk_id,
    observation.run_id = $run_id,
    observation.confidence = $confidence,
    observation.valid_from = $valid_from,
    observation.valid_to = $valid_to,
    observation.status = $status
MERGE (owner:Entity {canonical_entity_id: $owner_entity_id})
MERGE (owner)-[:HAS_OBSERVATION]->(observation)
```

For a relationship-owned observation, use the same `PropertyObservation` fields but attach it through the implementation's stable relationship key or a relationship observation node. The writer must not use a display string as the relationship identity.

### Validation and failure behavior

- Reject the complete write before opening a transaction if required fields, IDs, relationship types, provenance, or confidence values are invalid.
- Reject writes containing canonical IDs for `uncertain` or `abstain` decisions.
- Reject observations with incompatible values only when they are malformed; valid conflicting observations are stored and marked `conflict`.
- Retry transient Neo4j connectivity or transaction failures with bounded backoff.
- Roll back the transaction on any non-retryable write failure.
- Record rejected writes and failed transactions in a graph-write dead-letter log with run ID, error category, and provenance.

### Output shape

```json
{
  "run_id": "run-2026-09-15-001",
  "status": "committed",
  "created_entity_ids": ["entity:identity-api", "entity:user-credentials"],
  "created_relationship_ids": ["relationship:validates:identity-api:user-credentials:chunk-3"],
  "updated_entity_ids": [],
  "updated_relationship_ids": [],
  "conflicts": [
    {
      "owner_id": "entity:identity-api",
      "property_name": "headquarters",
      "category": "property_value_conflict",
      "observation_ids": [
        "observation:identity-api:headquarters:chunk-3",
        "observation:identity-api:headquarters:chunk-8"
      ],
      "status": "conflict"
    }
  ],
  "errors": []
}
```

## Contract 5: User Question To Retriever Agent

**Status:** Specified; implementation and tests remain TODO.

### Owner

- **Producer:** Answering Agent or query entry point
- **Consumer:** Retriever Agent

### Purpose

Define how the answering workflow requests bounded, read-only evidence retrieval from Neo4j. The Retriever Agent returns inspectable graph paths and source chunks; it does not generate the final answer or modify the graph during normal retrieval.

### Questions for you

The current retrieval mode is GraphRAG over Neo4j. Vector RAG and hybrid routing remain future extensions of this contract.

- `query` is the user's natural-language question.
- `max_hops` bounds graph traversal; the initial default is `2` and the maximum allowed value is `3`.
- `top_k` bounds the number of evidence results; the initial default is `5` and the maximum allowed value is `20`.
- Retrieval returns evidence records containing paths, entity IDs, relationship types, source chunk IDs, and provenance.
- Empty results return `status: no_evidence` with an empty `evidence` list, not an invented answer.
- Neo4j access is read-only for this contract.
- `trace_id` correlates retrieval with the LangGraph run and later answer/reflection steps.

### Input shape

```json
{
  "query": "Which service validates user credentials?",
  "max_hops": 2,
  "top_k": 5,
  "trace_id": "trace-2026-09-15-001",
  "filters": {
    "entity_types": ["SoftwareService", "Credential"],
    "relationship_types": ["VALIDATES"]
  }
}
```

`filters` is optional. When present, relationship and entity types must come from the controlled ontology vocabulary. The retriever must reject negative, zero, non-integer, or over-limit `max_hops` and `top_k` values rather than silently widening or ignoring them.

### Output shape

```json
{
  "trace_id": "trace-2026-09-15-001",
  "evidence": [
    {
      "evidence_id": "evidence-1",
      "path": [
        {
          "entity_id": "entity:identity-api",
          "name": "Identity API",
          "entity_type": "SoftwareService"
        },
        {
          "relationship_type": "VALIDATES",
          "confidence": 0.94
        },
        {
          "entity_id": "entity:user-credentials",
          "name": "User Credentials",
          "entity_type": "Credential"
        }
      ],
      "source_chunk_ids": ["musique:2hop__460946_294723:paragraph:3"],
      "citations": [
        {
          "chunk_id": "musique:2hop__460946_294723:paragraph:3",
          "document_id": "musique:document:<normalized-title>",
          "title": "<Wikipedia title>",
          "source_split": "dev"
        }
      ],
      "properties": {}
    }
  ],
  "status": "evidence_found",
  "retrieval_metadata": {
    "max_hops": 2,
    "top_k": 5,
    "returned_count": 1,
    "read_only": true
  }
}
```

### Retrieval behavior

- Resolve query terms to candidate graph entities using deterministic name/type matching and the configured graph search strategy.
- Traverse only relationships allowed by `filters` and no farther than `max_hops`.
- Rank evidence deterministically under fixed configuration and return no more than `top_k` results.
- Preserve source chunk, document, entity, relationship, and trace provenance.
- Return graph evidence only; answer generation belongs to Contract 6.
- Do not use MuSiQue `is_supporting` labels during ordinary retrieval. Those labels are reserved for later evaluation.

### Status and failure behavior

Allowed statuses are:

- `evidence_found`: one or more validated evidence records were returned;
- `no_evidence`: the query was valid but no qualifying path or source chunk was found;
- `invalid_request`: query or retrieval controls failed validation;
- `query_timeout`: Neo4j did not respond within the configured timeout;
- `storage_unavailable`: Neo4j was unavailable after bounded retries;
- `error`: an unexpected retrieval failure occurred.

For `no_evidence`, return an empty evidence list and preserve the trace ID. For storage timeouts or availability failures, retry only transient failures with bounded backoff, then return the error status. Retrieval must not create, update, or delete Neo4j data.

### Validation rules

- `query` must be non-empty.
- `trace_id` must be non-empty.
- `max_hops` must be an integer from `1` through `3`.
- `top_k` must be an integer from `1` through `20`.
- Every returned `evidence_id` must be unique within the response.
- Every path must contain valid entity IDs, relationship types, and bounded hop count.
- Every citation must reference an existing chunk and preserve its source provenance.
- Returned evidence must be read from Neo4j without mutating it.

## Contract 6: Retriever Evidence To Answering Agent

**Status:** Specified; implementation and tests remain TODO.

### Purpose

Define how validated evidence from Contract 5 becomes the bounded context for the Answering Agent. The Answering Agent generates a draft response and claim-level evidence references; it does not query or modify Neo4j directly under this contract.

### Questions for you

The Answering Agent receives the original query, retrieved evidence, trace metadata, and generation configuration.

- Every factual claim must reference one or more returned `evidence_id` values.
- Evidence records with `status: conflict` or contradictory paths are passed through explicitly; the agent must not silently choose between them.
- Empty evidence produces `status: insufficient_evidence` rather than an unsupported factual answer.
- The model name, model version, prompt version, and generation settings are recorded in the output metadata.
- The draft is passed to Contract 7 for reflection and must not be treated as final until verified.

### Input shape

```json
{
  "query": "Which service validates user credentials?",
  "evidence": [
    {
      "evidence_id": "evidence-1",
      "path": [
        {
          "entity_id": "entity:identity-api",
          "name": "Identity API",
          "entity_type": "SoftwareService"
        },
        {
          "relationship_type": "VALIDATES",
          "confidence": 0.94
        },
        {
          "entity_id": "entity:user-credentials",
          "name": "User Credentials",
          "entity_type": "Credential"
        }
      ],
      "source_chunk_ids": ["musique:2hop__460946_294723:paragraph:3"],
      "citations": [
        {
          "chunk_id": "musique:2hop__460946_294723:paragraph:3",
          "document_id": "musique:document:<normalized-title>",
          "title": "<Wikipedia title>"
        }
      ],
      "status": "supported"
    }
  ],
  "trace_id": "trace-2026-09-15-001",
  "generation_config": {
    "model": "<answer model>",
    "model_version": "<model version>",
    "prompt_version": "answer-prompt-v1",
    "temperature": 0.0,
    "max_output_tokens": 512
  }
}
```

### Output shape

```json
{
  "trace_id": "trace-2026-09-15-001",
  "draft_answer": "The Identity API validates user credentials.",
  "claims": [
    {
      "claim_id": "claim-1",
      "text": "The Identity API validates user credentials.",
      "evidence_ids": ["evidence-1"],
      "support_status": "supported"
    }
  ],
  "status": "drafted",
  "metadata": {
    "model": "<answer model>",
    "model_version": "<model version>",
    "prompt_version": "answer-prompt-v1",
    "generation_config": {
      "temperature": 0.0,
      "max_output_tokens": 512
    }
  }
}
```

### Evidence and claim rules

- `evidence_id` values in `claims[].evidence_ids` must exist in the input evidence list.
- A factual claim must not have an empty `evidence_ids` list.
- The answer may include a clearly labeled uncertainty statement without evidence, but it must not present an unsupported proposition as fact.
- Claims supported by multiple paths may reference multiple evidence IDs.
- If evidence records disagree, claims depending on the disagreement receive `support_status: contradictory` and should not be presented as settled facts.
- The Answering Agent must not invent citations, chunk IDs, graph paths, or entity IDs.

### Status and failure behavior

Allowed output statuses are:

- `drafted`: a draft was generated with claim-level evidence references;
- `insufficient_evidence`: no adequate evidence was available for a factual answer;
- `contradictory_evidence`: relevant evidence conflicts and the draft explicitly reports the conflict;
- `provider_timeout`: the answer model did not respond within the configured timeout;
- `provider_unavailable`: the answer provider was unavailable after bounded retries;
- `invalid_output`: the answer failed schema or citation validation.

An empty evidence list must produce `insufficient_evidence`. Provider failures may be retried according to the run policy; invalid citations or unsupported claims must be rejected and sent to an answer-generation dead-letter log. The result is always a draft until Contract 7 reflection approves delivery.

### Validation rules

- `query` and `trace_id` must be non-empty.
- Every claim must have a unique `claim_id` and non-empty text.
- Every claim citation must reference an input `evidence_id`.
- `drafted` requires at least one claim or an explicitly non-factual response.
- Model and prompt metadata must be present for a generated draft.
- No answer output may mutate Neo4j.

## Contract 7: Answering Agent To Reflection Agent

**Status:** Specified; implementation and tests remain TODO.

### Purpose

Define the deterministic and model-assisted checks applied to a draft answer before delivery. The Reflection Agent verifies claim support, citations, graph paths, contradictions, and retry limits. It may request fresh retrieval or answer regeneration, but it must never approve an unsupported factual claim.

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
- [x] Contract 2 specified; implementation and tests remain TODO.
- [x] Contract 3 specified with candidate generation, match, non-match, uncertain, and abstain semantics; implementation and tests remain TODO.
- [x] Contract 4 specified with initial labels, relationships, constraints, indexes, and idempotent write rules; implementation and tests remain TODO.
- [x] Contract 5 specified with bounded retrieval behavior, provenance, read-only access, and empty-result handling; implementation and tests remain TODO.
- [x] Contract 6 specified with claim-to-evidence, contradiction, metadata, and failure rules; implementation and tests remain TODO.
- [x] Contract 7 specified with verification, retry, termination, contradiction, and failure rules; implementation and tests remain TODO.
- [ ] Contract 8 completed with LangGraph nodes and transition conditions.
- [ ] Contract 9 completed after the evaluation dataset and protocol are frozen.
- [ ] Every contract names an owner, consumer, validation rule, provenance requirement, and failure behavior.
- [ ] Every contract has at least one passing test or fixture case.
