# MuSiQue Data For This Project

## Purpose

This document explains how the local MuSiQue release should be used by the Self-Correcting GraphRAG Knowledge Pipeline with Autonomous Entity Resolution. It separates:

- what is needed now for the local core implementation;
- what will support later extraction, resolution, retrieval, reflection, and evaluation work;
- what belongs to the original MuSiQue research code but is not part of this capstone implementation.

The implementation roadmap in `agent-coordination/tasks.md` controls project scope. The core path is local Python execution, local MuSiQue storage, local Neo4j, LangGraph orchestration, linear processing, and Neo4j AI plugin-based retrieval. The project uses Neo4j's AI plugin to implement both GraphRAG and VectorRAG in the default runtime. Docker, Ray, Redis, PostgreSQL, Polars, DuckDB, custom hand-built retrieval layers, Hybrid Search, and automatic retrieval routing are deferred unless approved as stretch work.

## Core Technology Boundary

This section is the working boundary for the core implementation. It is intentionally narrower than the original proposal and must be reflected in setup instructions, dependency metadata, run scripts, and tests.

### Runtime

- The macOS setup uses `uv`, `.python-version`, and the committed `uv.lock`; invoke commands with `uv run --locked` rather than an absolute virtual-environment path. Python 3.14.3 is the current verified baseline, not a guarantee for every macOS release or architecture.
- Recheck the lock and tests when upgrading dependencies or changing the supported Python baseline. The package metadata in `pyproject.toml` remains the source of declared dependency compatibility; `uv.lock` records the concrete resolved environment.
- Core dependencies include the Neo4j driver, Neo4j GraphRAG, Pydantic, provider clients, and test tooling. Provider clients are installed through the declared GraphRAG extras.
- Configuration is supplied through environment variables or a local, ignored `.env` file. Secrets must never be embedded in source, fixtures, graph data, or benchmark artifacts.
- Fixture and mock workflows must run without OpenRouter, a paid embedding service, or any other external model API.

### Local storage and Neo4j

- Official MuSiQue JSONL files are read from local storage under `dataset/musique/data/data/`. Raw files are immutable inputs; normalization writes separate artifacts and never edits the official files in place.
- The core graph store is one locally running Neo4j database. It stores `Document`, `Chunk`, `Entity`, optional `Mention`, and provenance/observation data. Neo4j's AI plugin is the default implementation for both GraphRAG and VectorRAG retrieval, while the project continues to keep graph writes, chunk provenance, and verification logic in local Python and LangGraph.
- The local Neo4j instance must provide a documented connection URI, database name, credentials configuration, persistence location, transaction timeout, retry policy, initialization script, plugin configuration, and fixture reset procedure.
- Graph writes are performed by the local Python process through the Neo4j driver. Retrieval uses Neo4j AI plugin GraphRAG and VectorRAG capabilities, and reflection continues to use bounded, read-only verification queries.

### Linear processing contract

The supported ingestion path is sequential and deterministic:

```text
select one local JSONL split
  -> read one record
  -> validate the record
  -> emit its paragraphs in source order as chunks
  -> extract one chunk
  -> validate the extraction
  -> resolve its mentions
  -> write one idempotent Neo4j transaction
  -> record success or dead-letter failure
```

The flow has no worker pool, distributed scheduler, Ray task, or parallel graph writer. It preserves source order for reproducible logs, uses bounded retries only where the contracts permit them, and assigns one run ID to the complete invocation. A rerun of the same validated input must not create duplicate graph nodes or relationships.

### Deliberately outside the core boundary

The following are deferred and must not be required to install, test, or run the core fixture workflow:

- Docker and Docker Compose;
- Ray workers or Ray Data;
- Polars and DuckDB;
- Redis session state;
- PostgreSQL telemetry persistence;
- S3 or other cloud object storage;
- custom hand-built retrieval implementations that the Neo4j AI plugin replaces;
- automatic retrieval routing;
- Hybrid Search.

Neo4j AI plugin VectorRAG and GraphRAG are the required retrieval modes for the default runtime. Custom retrieval implementations, plugin comparison baselines, and custom vector/index wrappers are moved to the deferred stretch tasks section in `agent-coordination/tasks.md` rather than treated as part of the core project path. They may be introduced only through an approved stretch decision with updated dependency, resource, and reproducibility documentation.

### Boundary evidence required before marking the roadmap item complete

The project must provide:

1. a pinned Python/dependency setup and environment-variable template;
2. a local Neo4j setup, initialization, persistence, and reset procedure;
3. a fixture command that runs without paid APIs or deferred services;
4. a trace showing sequential record-to-chunk-to-Neo4j processing;
5. an idempotency test showing that the fixture can be processed twice without duplicate graph data; and
6. a decision record listing the deferred technologies and confirming that the core workflow does not import or provision them.

## What MuSiQue Is

MuSiQue means **Multi-hop Questions via Single-hop Question Composition**. It is a multi-hop reading-comprehension benchmark built from Wikipedia paragraphs. A question requires combining facts from multiple paragraphs, and the dataset identifies the paragraphs that support the answer for labeled records.

This makes MuSiQue useful for this project because it provides:

- naturally multi-hop questions;
- multiple candidate paragraphs, including distracting paragraphs;
- gold answers and answer aliases on labeled splits;
- supporting-paragraph labels;
- question decompositions that expose constituent reasoning steps;
- hop-shaped question IDs such as `2hop__460946_294723`;
- answerable and unanswerable examples through MuSiQue-Full.

The dataset is released under **CC BY 4.0**. The local license is in `dataset/musique/LICENSE`, and the upstream documentation is in `dataset/musique/README.md`. Any public release of derived artifacts must preserve the applicable attribution and licensing requirements.

## Local Directory And Available Data

The checked-out release is under `dataset/musique/`. The official files currently present are nested under `dataset/musique/data/data/`:

| File | Records | Project relevance |
|---|---:|---|
| `musique_ans_v1.0_train.jsonl` | 19,938 | Future development or larger ingestion runs; labels are available. |
| `musique_ans_v1.0_dev.jsonl` | 2,417 | Primary labeled validation source for early inspection and later evaluation. |
| `musique_ans_v1.0_test.jsonl` | 2,459 | Prediction-only testing; labels are not released. |
| `musique_full_v1.0_train.jsonl` | 39,876 | Future work involving answerable and unanswerable cases. |
| `musique_full_v1.0_dev.jsonl` | 4,834 | Future answerability, insufficient-evidence, and reflection evaluation. |
| `musique_full_v1.0_test.jsonl` | 4,918 | Prediction-only testing; labels are not released. |
| `dev_test_singlehop_questions_v1.0.json` | 6,371 entries | Leakage control if the project later uses source single-hop datasets. |

The current project should begin with a small local fixture rather than processing every record or making paid LLM calls. The existing MuSiQue repository describes fixture settings for its original AllenNLP experiments, but those fixture files are not currently present in this checkout. The capstone should create its own small fixture with the fields and edge cases required by the new pipeline.

## Official Record Schema

A labeled MuSiQue record contains these important top-level fields:

```json
{
  "id": "2hop__460946_294723",
  "question": "...",
  "question_decomposition": [
    {
      "id": 259228,
      "question": "Ulrich Walter >> employer",
      "answer": "German Aerospace Center",
      "paragraph_support_idx": 14
    },
    {
      "id": 793698,
      "question": "#1 >> headquarters location",
      "answer": "Cologne",
      "paragraph_support_idx": 11
    }
  ],
  "answer": "...",
  "answer_aliases": ["..."],
  "answerable": true,
  "paragraphs": [
    {
      "idx": 0,
      "title": "...",
      "paragraph_text": "...",
      "is_supporting": false
    }
  ]
}
```

The exact presence of labels depends on the split:

- train and dev records contain labels needed for supervised development and evaluation;
- test records contain the question and paragraphs but intentionally withhold answer and support labels;
- MuSiQue-Ans focuses on answerable questions;
- MuSiQue-Full includes answerable and unanswerable questions.

A real local dev record confirms that paragraph text is stored in `paragraph_text`, not a generic `text` field, and that each paragraph has an integer `idx`, a title, and an `is_supporting` flag. The adapter should preserve the official fields while exposing a stable internal model.

## What The Capstone Needs From Each Record

### Question and record identity

The record `id` is the primary dataset-level identifier. It must be preserved in every downstream artifact. The hop prefix is useful metadata but should not be treated as the only source of truth for the actual graph path.

Recommended internal names:

```text
record_id       <- id
question        <- question
answer          <- answer, when labels are available
answer_aliases  <- answer_aliases, when available
answerable      <- answerable, when available
```

### Paragraph context

Every paragraph is a candidate evidence unit. It has:

- a record-local index, `idx`;
- a Wikipedia title;
- paragraph text;
- a supporting label on labeled records.

The project should initially treat one MuSiQue paragraph as one application chunk. No additional sentence splitting is required unless context-window limits, extraction quality, or retrieval experiments show that it is necessary.

### Supporting facts

`is_supporting: true` identifies paragraphs selected by the dataset as supporting the answer. These labels are important for:

- checking whether extraction sees the relevant evidence;
- comparing graph paths with gold evidence;
- measuring supporting-paragraph retrieval;
- constructing reflection fixtures;
- later calculating multi-hop retrieval metrics.

A supporting paragraph is evidence metadata, not automatically a graph relationship. The extraction agent must still infer entities and relations from the text, and the system must retain the paragraph as provenance for those inferred facts.

### Question decomposition

`question_decomposition` contains the constituent single-hop questions and their intermediate answers or support references in labeled data. It is useful for understanding why a question is multi-hop and for building gold reasoning traces.

It is not required for the first ingestion and extraction slice. Later it can support:

- expected hop-count metadata;
- gold intermediate facts;
- path-construction checks;
- multi-hop query-suite annotation;
- comparison with decomposition-based baselines.

The project should not assume that the production GraphRAG pipeline must reproduce the dataset's decomposition format. The decomposition is evaluation and analysis evidence unless the advisor explicitly makes decomposition part of the system design.

### Answer aliases

`answer_aliases` lists accepted alternative answer strings. These are useful for later answer evaluation and for avoiding false negatives caused by harmless surface-form differences.

They should not be used to create graph entities or relationships. They belong to answer evaluation and possibly answer normalization only.

## Recommended Project Data Flow

The dataset already provides structured records and paragraph contexts. The capstone therefore needs a light adapter rather than a heavy cleaning pipeline:

```text
Official MuSiQue JSONL
  -> validate record shape and required fields
  -> preserve official values and labels
  -> assign stable application IDs
  -> expose one paragraph as one chunk
  -> extraction agent
  -> validated entities and relations
  -> entity resolution
  -> idempotent Neo4j writes
```

For querying:

```text
User question
  -> Answering Agent
  -> Retriever Agent
  -> Neo4j evidence and graph paths
  -> Answering Agent
  -> Reflection Agent
  -> Grounded response
```

The user question enters the answering workflow. The Answering Agent requests relevant context from the Retriever Agent, which queries the Neo4j knowledge graph and returns evidence and graph paths. The Answering Agent uses that evidence to produce a draft answer, and the Reflection Agent verifies the draft before the system returns a Grounded Response. The Retriever Agent retrieves evidence; it does not produce the final answer.

LangGraph owns the agent state and transitions. It does not replace the dataset adapter or Neo4j storage layer.

## Current Implementation Relevance

### 1. Start with a MuSiQue adapter

The first implementation should read official JSONL records and validate only what the pipeline needs. The adapter should:

- read JSONL one record at a time;
- reject or report malformed JSON and missing required fields;
- preserve the original record ID;
- preserve question text and paragraph text exactly;
- preserve paragraph indices and titles;
- preserve `is_supporting` when available;
- retain answer fields only when they exist;
- distinguish labeled records from test records;
- produce stable internal objects for ingestion and tests.

Do not rewrite the dataset in place. Store normalized output in a separate local artifact location and retain a reference to the source file and source record.

### 2. Use paragraph-as-chunk initially

The current chunk contract can be:

```json
{
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "record_id": "2hop__460946_294723",
  "paragraph_idx": 3,
  "document_id": "musique:document:<normalized-title>",
  "title": "...",
  "text": "...",
  "is_supporting": false,
  "source_split": "dev"
}
```

This gives every paragraph a stable evidence identity without introducing unnecessary sub-chunking. `document_id` identifies the source Wikipedia document using a deterministic normalized form of the paragraph title; `chunk_id` identifies this specific paragraph within the MuSiQue record. If a paragraph later exceeds the extraction model's context budget, derive child chunk IDs from the parent paragraph ID and a deterministic segment index while keeping the same `document_id`.

### 3. Build a fixture before using paid models

The project fixture should include at least:

- one answerable two-hop record;
- multiple candidate paragraphs and distractors;
- supporting and non-supporting paragraphs;
- an alternate entity mention or alias;
- a record with a missing optional field;
- a malformed record for validation tests;
- a record that exercises insufficient evidence or an unanswerable case if MuSiQue-Full behavior is being tested.

The fixture must run without downloading the full dataset or calling OpenRouter. Use deterministic mock extraction and embedding responses for unit and integration tests.

### 4. Preserve provenance from paragraph to graph

At minimum, graph evidence should retain:

```text
source_dataset       = musique
source_split         = train, dev, or test
record_id            = MuSiQue id
paragraph_idx        = official paragraph index
chunk_id             = stable application chunk ID
title                = paragraph title
extraction_run_id    = processing run identifier
```

For labeled records, also retain whether the paragraph was marked supporting. This lets the reflection and evaluation code compare retrieved graph evidence with the original MuSiQue evidence without treating labels as facts discovered by the LLM.

## Future Extraction And Graph Construction

The extraction agent will receive one validated chunk and return structured entities, relationships, properties, evidence, and confidence. The dataset itself does not provide the project's final Neo4j graph. It provides text and supervision; the agent must infer graph content.

A future extraction result should include:

```json
{
  "record_id": "2hop__460946_294723",
  "chunk_id": "musique:2hop__460946_294723:paragraph:3",
  "entities": [
    {
      "mention": "Example Entity",
      "entity_type": "Organization",
      "surface_span": "Example Entity"
    }
  ],
  "relations": [
    {
      "source_mention": "Example Entity",
      "relationship_type": "RELATED_TO",
      "target_mention": "Another Entity",
      "evidence_text": "...",
      "confidence": 0.91
    }
  ]
}
```

The output must be validated before graph writes. Entity resolution then maps mentions to canonical IDs, records match/non-match/uncertain/abstain decisions, and writes idempotently to Neo4j.

Relevant future Neo4j uses include:

- paragraph and chunk provenance;
- canonical entity nodes;
- extracted relationship edges;
- resolution decisions and confidence;
- source references for claims;
- bounded multi-hop traversal;
- deterministic reflection queries.

The original MuSiQue labels should remain metadata for evaluation and analysis. They should not be used to bypass extraction or silently construct a perfect graph.

## Future Retrieval And Reflection Work

MuSiQue is particularly relevant to the graph retriever because each question contains multiple candidate paragraphs and requires cross-paragraph reasoning.

The Retriever Agent will eventually:

- receive a natural-language question;
- retrieve relevant graph entities, relationships, chunks, and paths;
- enforce a bounded hop depth and result limit;
- return inspectable evidence with source chunk IDs;
- avoid modifying the graph during ordinary retrieval.

The Answering Agent will receive the retrieved evidence and produce a draft answer with claim-to-evidence references. It should be separate from Neo4j storage so the graph remains an evidence source rather than an uncontrolled output sink.

The Reflection Agent will later:

- inspect claims and citations;
- verify graph paths against Neo4j;
- compare evidence with MuSiQue supporting paragraphs when labels are available;
- detect insufficient or contradictory evidence;
- route failed drafts back through retrieval or answering within a retry budget;
- return a transparent insufficient-evidence result when verification fails.

The Full split is especially useful for testing the last behavior because it includes unanswerable records. The Ans split is simpler for the first working path because every record is intended to be answerable.

## Future Evaluation Relevance

Evaluation is intentionally not part of the initial architecture diagram, but the dataset already supplies artifacts needed for the roadmap's later benchmark phase.

### Entity-resolution evaluation

MuSiQue is not itself the complete 150-pair entity-resolution benchmark requested in `tasks.md`. Later, create a separate frozen labeled dataset of entity pairs drawn from representative MuSiQue contexts. Include:

- equivalent mentions;
- aliases and spelling variants;
- abbreviations;
- context-dependent matches;
- hard non-matches;
- rationale, provenance, and adjudication information.

Use MuSiQue paragraph and record IDs to trace each pair back to its source. Do not treat the existing `answer_aliases` field as an entity-resolution label.

### Multi-hop retrieval evaluation

The roadmap requires 30-50 cross-document multi-hop queries with gold facts and expected graph paths. MuSiQue provides a strong source for this suite:

- select records with clear multi-hop structure;
- use `question`, `question_decomposition`, and supporting paragraphs to annotate required facts;
- record expected paragraph evidence and eventual graph paths;
- verify that the answer cannot be recovered from one isolated paragraph;
- freeze the selected records and manifest them with checksums.

Use the labeled dev split for initial query-suite construction and reserve an untouched subset for final evaluation. Do not tune graph thresholds on the same records used for final reporting.

### Answer quality and support evaluation

The official evaluator computes answer and support metrics for labeled data. It can provide useful context, but it does not replace the project's RAGAS-based reporting requirements. The project should preserve:

- raw system answers;
- claim-level citations;
- retrieved chunks and graph paths;
- answer aliases;
- answerability status;
- configuration and model metadata.

For MuSiQue-Full, preserve unanswerable examples because they are useful for measuring whether reflection suppresses unsupported answers.

### Operational evaluation

MuSiQue records also support later measurement of:

- ingestion throughput in records or paragraphs per second;
- extraction failures and retry counts;
- graph write latency;
- retrieval latency and hop depth;
- answer-generation token usage;
- end-to-end latency and cost.

The roadmap's P50/P95, throughput, and cost targets require the project to add run metadata and benchmark logs; the dataset alone does not provide those measurements.

## Dataset Files And What To Use

### Use directly for the capstone

- `dataset/musique/data/data/musique_ans_v1.0_dev.jsonl`: convenient labeled development data for the first adapter and graph fixture sampling.
- `dataset/musique/data/data/musique_ans_v1.0_train.jsonl`: larger labeled source for later ingestion and extraction runs.
- `dataset/musique/data/data/musique_full_v1.0_dev.jsonl`: later tests for unanswerable questions and insufficient-evidence behavior.
- `dataset/musique/data/data/musique_full_v1.0_train.jsonl`: later larger-scale full-scope ingestion.
- `dataset/musique/data/data/dev_test_singlehop_questions_v1.0.json`: leakage control if source single-hop datasets are introduced.

### Use later for evaluation or prediction

- `musique_ans_v1.0_test.jsonl` and `musique_full_v1.0_test.jsonl`: prediction-only data because gold labels are withheld.
- `question_decomposition`: later gold reasoning and path annotation.
- `answer_aliases`: later answer scoring and acceptable-answer handling.
- official `evaluate_v1.0.py` and `metrics/`: reference metrics and compatibility checks, not the complete capstone evaluation harness.

### Do not make part of the core GraphRAG implementation

The following belong to the original MuSiQue research repository and should not be imported into the capstone pipeline unless there is a specific research reason:

- AllenNLP 2.1.0 and its patched checkout;
- the old Transformers and Longformer training stack;
- the original BART decomposition models;
- original End2End, Select+Answer, EX(EE), and EX(SA) model implementations;
- released checkpoints and raw prediction downloads;
- the dataset repository's evaluator Dockerfile;
- HotpotQA-20K and 2WikiMultihopQA-20K experiment configurations.

These materials are useful as references and possible baselines, but they would add substantial setup complexity and are not required for the local LangGraph/Neo4j implementation.

## Data Integrity And Leakage Rules

1. Do not train or evaluate against unreleased test labels. Test files are for prediction only.
2. Keep raw dataset files separate from normalized records, extracted triples, graph exports, and benchmark outputs.
3. Record file checksums and source split for reproducibility.
4. Preserve record IDs and paragraph indices exactly.
5. Do not silently change paragraph text during normalization.
6. Do not use `answer_aliases` as entity-equivalence labels.
7. If the project ever uses SQuAD, Natural Questions, T-REx, MLQA, or Zero Shot RE, use `dev_test_singlehop_questions_v1.0.json` to prevent leakage.
8. Keep API keys and private local data out of the repository.
9. Record model, prompt, threshold, retry, and embedding configuration with every generated artifact.
10. Preserve enough provenance to trace a final claim back to its MuSiQue record and paragraph.

## Recommended Implementation Sequence

### Current phase

1. Create a small project fixture based on the official paragraph schema.
2. Implement JSONL reading and required-field validation.
3. Implement a light adapter that maps official fields to internal record and chunk models.
4. Use one paragraph as one chunk and assign deterministic IDs.
5. Write tests for valid records, missing fields, malformed records, duplicate IDs, and test records without labels.
6. Load the fixture into Neo4j with document, chunk, and provenance information.
7. Add deterministic mock extraction before connecting to a paid LLM.

### Next implementation phase

1. Define Pydantic schemas for extraction output.
2. Implement the LangGraph extraction workflow with bounded retries.
3. Validate extraction output before graph writes.
4. Implement entity candidates, resolution decisions, canonical IDs, and idempotent `MERGE` operations.
5. Add resolution audit records and conflict handling.
6. Implement the Retriever Agent over Neo4j paths and source chunks.
7. Implement the Answering Agent and Reflection Agent with a shared LangGraph state.

### Later evaluation phase

1. Select and freeze labeled MuSiQue records for a 30-50 query multi-hop suite.
2. Build the separate 150-pair entity-resolution dataset.
3. Preserve raw predictions, claims, citations, paths, latency, token usage, and failures.
4. Compare retrieval and answer quality using the approved evaluation protocol.
5. Use the official MuSiQue evaluator as a reference where applicable, while reporting the capstone's required metrics separately.

## Relationship To The Architecture

The current architecture can use the following interpretation:

```text
MuSiQue Dataset
  -> Normalizer
  -> Chunker
  -> Extractor System
  -> Entity Resolution System
  -> Neo4j Knowledge Graph

User Query
  -> Retriever Agent
  -> Answering Agent
  -> Reflection Agent
  -> Grounded Response
```

This is compatible with the dataset because:

- Normalizer validates and adapts the official record schema.
- Chunker initially preserves each provided paragraph as one chunk.
- Extractor infers graph facts from paragraph text.
- Entity resolution maps mentions to canonical graph entities.
- Neo4j stores graph facts and provenance.
- Retriever and Answering agents consume graph evidence rather than writing answers into the graph.
- Reflection verifies the answer and can query the graph again.
- LangGraph orchestrates the agent transitions.

Evaluation, Vector RAG, Docker, Ray, and other deferred features can be added later without changing the basic MuSiQue adapter contract.

## Practical Definition Of Done For This Dataset Work

The MuSiQue data setup is ready for implementation when:

- a clean local command can read a selected official JSONL split;
- the adapter validates and preserves the official fields;
- every paragraph has a deterministic chunk ID;
- supporting labels and answer metadata remain available when present;
- test records without labels are not accidentally evaluated as labeled data;
- a fixture can flow through normalization, chunking, mock extraction, and Neo4j storage;
- every graph record retains dataset, split, record, paragraph, and chunk provenance;
- the original research-code environment is not required to run the capstone core path.
