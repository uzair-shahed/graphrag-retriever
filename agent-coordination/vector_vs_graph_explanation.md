# VectorRAG vs. GraphRAG: what the latest report says

## What this report covers

This explanation is based on the completed run in `runs/eval-prepared-4-to-6-part-3/report.html` and its JSON artifacts. It used three MuSiQue development questions (records 4–6 in the dataset), each evaluated in two search conditions (`record` and `corpus`) and with both VectorRAG and GraphRAG: 12 rows total.

This is a useful debugging sample, but it is far too small to establish which system is generally better. Only one of the three questions received an answer; the other two were abstentions. So answer-quality averages have a denominator of one per backend and condition. The result applies to this run and configuration, not to all 2,417 dataset records.

## RAG and the RAG Triad, in plain language

RAG means **retrieval-augmented generation**. The system first searches its stored text for evidence, then gives some of that evidence to a language model to produce an answer.

The **RAG Triad** checks three links in that process:

1. **Context relevance:** Did the search return text that relates to the question?
2. **Faithfulness:** Is the answer supported by the text the model was shown?
3. **Answer relevance** (called `answer_relevancy` in the files): Does the answer address the question?

RAGAS is the evaluation library used here. These three scores are judged by a model rather than checked against MuSiQue's expected answer. In this project the judge is also `qwen3:4b-instruct`, through Ollama. The scores range from 0 to 1; closer to 1 is better. A score is a judge result, not a proof of truth. Gold-answer and gold-support checks in this report provide a separate, dataset-based view.

## What outcome was the project aiming for?

The project is comparing ordinary vector search with a graph-expanded version using the same question, vector seeds, answer model, and answer rules. The hoped-for result is that graph connections help retrieve missing evidence for multi-step questions, so GraphRAG covers more gold supporting paragraphs and answers more questions correctly than VectorRAG, without adding too much irrelevant text or latency.

The intended comparison is not “graph always wins.” A good result would show a measurable gain on questions where graph expansion adds useful evidence. The project notes also explain an important current limitation: extracted graph components are scoped to individual source paragraphs and do not connect separate paragraphs. Therefore graph expansion currently adds entity and relationship descriptions, but does not generally bring in additional gold source paragraphs. This run cannot demonstrate the planned cross-paragraph multi-hop benefit.

## Results at a glance

Both search conditions produced the same quality scores in this sample, so the table shows the record condition; the corpus condition has the same values. (Corpus-wide search means searching all available graph data; record search is constrained to the question’s record.)

| Measure | VectorRAG | GraphRAG | What it says here |
|---|---:|---:|---|
| Answered / abstained | 1 / 2 | 1 / 2 | Both answered one question and abstained on two. |
| Gold exact answer accuracy | 1/3 (0.33) | 0/3 (0.00) | VectorRAG got the one answerable question right; GraphRAG did not. |
| Answer F1 | 0.33 | 0.08 | Token overlap with the gold answer; GraphRAG’s response had only partial overlap. |
| Retrieved supporting-paragraph recall | 0.50 | 0.50 | Both retrieved half of the labeled supporting paragraphs on average. |
| Retrieved supporting-paragraph precision | 0.20 | 0.20 | Only one fifth of retrieved paragraphs were gold supporting paragraphs on average. |
| Decomposition-hop coverage | 0.50 | 0.50 | Half of the hops’ labeled source paragraphs were retrieved on average. |
| RAGAS faithfulness (answered only) | 1.00 | 0.00 | Vector answer was judged supported; Graph answer was not. One scored answer each. |
| RAGAS answer relevance (answered only) | 0.00 | 0.98 | Opposite of gold correctness in this sample; see the investigation below. |
| RAGAS context relevance | 0.33 | 0.33 | Same average for both; context relevance is scored across all three rows. |
| Mean answer-context size | about 546 tokens | about 1,146 tokens | Graph context was roughly twice as large. These are `cl100k` estimates, not the model’s exact token counts. |

There were no query failures, RAGAS metric failures, or incomplete graph traversals. The before/after graph fingerprints match, so the graph did not change during the run. These are useful checks that the run completed and the comparison was internally consistent; they do not mean the answers were good.

## Good findings

- The pairing worked as intended: both backends used identical vector seed IDs for a given question and condition. That makes the comparison fairer.
- Citations that were returned for the answered row pointed to gold supporting paragraphs (citation gold-support precision 1.0). This is only one answered case, though, and the measure checks paragraph labels, not whether the cited sentence proves every part of the answer.
- The VectorRAG answer to “Which company owns the manufacturer of Learjet 60?” was **“Bombardier Inc.”**, matching the gold answer exactly. It had exact accuracy and F1 of 1.0 on that row.
- The run completed all 12 expected rows with a consistent graph snapshot and no metric-call errors.

## Negative findings

- Two of three questions were marked `insufficient_evidence` by both backends. The abstention rate is 2/3, even though these selected MuSiQue questions are answerable. The answerability accuracy is 1/3: the system abstained on two answerable questions and answered one.
- Neither backend retrieved all gold support reliably: supporting recall and hop coverage were 0.5. Supporting precision was 0.2, meaning most retrieved paragraphs were not labeled as supporting evidence for these questions.
- GraphRAG did not improve support recall, hop coverage, or context relevance over VectorRAG. It used substantially more context and was slightly slower in the record condition (about 4.14s versus 3.97s average; corpus timings are noisy and not representative at this sample size).
- On the one answered question, GraphRAG said **“Bombardier Aerospace owns the manufacturer of the Learjet 60.”** The gold answer is **“Bombardier Inc.”** The answer is not just a different wording: the cited passage says Learjet 60 is manufactured by Bombardier Aerospace, and a separate supporting passage identifies Bombardier Aerospace as the aerospace division of Bombardier Inc. It does not say Bombardier Aerospace owns the manufacturer. The report therefore marks the answer incorrect (exact match 0; F1 0.25).

## Weird results and what they may mean

### VectorRAG answer relevance is 0 even though its answer is correct

For the only answered question, VectorRAG returned the exact gold answer, “Bombardier Inc.” RAGAS still gave it answer relevance 0.0. This metric is a model-and-embedding-based score; it does not compare the answer with MuSiQue’s gold label. The report shows the score and no metric execution failure, but it does not provide enough evidence to identify the precise reason for the zero. It could reflect how the RAGAS judge generated or embedded reverse questions for this very short answer, or a judge/model behavior issue. It is not evidence that this answer failed to answer the question. The contradiction with exact gold correctness is a red flag to investigate in the saved metric trace or by manually checking RAGAS inputs and judge outputs.

### GraphRAG answer relevance is high while faithfulness and gold correctness are zero

The GraphRAG response is phrased as a direct answer to the question, which likely explains its high answer-relevance score (0.982). But being on-topic is different from being supported or correct. It asserted an ownership relation absent from the text, so faithfulness was 0 and the gold answer check failed. These metrics are measuring different things, not necessarily contradicting each other.

### The graph output contains dubious extracted relations

In GraphRAG’s rendered evidence for this question, the text says the Learjet is made by Bombardier Aerospace. The graph lines include `Bombardier Aerospace -[MANUFACTURES]-> Learjet 60`, which is directionally plausible, but graph extraction can also introduce noisy entity or relation labels. The graph rendering increases the answer context from about 546 to 1,146 tokens without raising gold support coverage. In this example, added graph text did not prevent the unsupported “owns” claim.

### The report provides clues, but not a complete root-cause diagnosis

The HTML and JSONL retain the question, gold answer and supporting paragraphs, exact retrieval evidence, exact answer context and prompt, graph nodes and relationships, answer, citations, RAGAS values/statuses, and retrieval/answer-context support calculations. This is enough to inspect whether evidence was retrieved, whether it survived context selection, and what graph facts were shown to the answer model. It is not enough to establish why the judge gave VectorRAG relevance 0: the summary has the score and failure status, but RAGAS did not return an explanatory reason. Nor can one sample establish whether the graph extractor or answer model caused a general pattern. The next debugging step is to inspect the affected row’s full RAGAS fields and re-score the same saved inputs with a second judge or manual review.

## What each metric means

### RAGAS Triad

| Metric | What it measures | How to read this run |
|---|---|---|
| **Faithfulness** | Whether the answer’s claims are supported by the supplied answer context. | VectorRAG 1.0 and GraphRAG 0.0 on their one answered row. The graph answer made an ownership claim the context did not support. |
| **Answer relevance** | Whether the answer addresses the question, judged using RAGAS’s answer-relevance procedure. | VectorRAG 0.0 is suspicious because its answer exactly matches gold; GraphRAG 0.982 shows it was on-topic, not that it was true. |
| **Context relevance** | Whether the supplied context is relevant to the question. | 0.33 average for both. It is an LLM-judged score, and here it includes abstained rows with available context. |

Faithfulness and answer relevance are averaged over **answered rows only**. Abstentions are N/A, not zeros. Context relevance can be scored for abstentions as long as context exists. The count beside each mean matters: the answer metrics each have count 1, while context relevance has count 3.

The report also shows **retrieval-context relevance**, which scores all retrieved evidence before the final answer-context selection. It is 0.33 for both in this sample. This helps distinguish poor initial retrieval from context-selection effects.

### Gold-answer and evidence metrics

| Metric | What it measures | How to read it |
|---|---|---|
| **Answer accuracy** | Exact normalized match to the gold answer or an accepted alias; incorrect answers and abstentions on answerable questions count as wrong. | 0.33 VectorRAG and 0 GraphRAG. This is the clearest answer-correctness measure in this small run. |
| **Answer F1** | Word overlap between the generated answer and the gold answer, after simple normalization. | 0.33 vs. 0.08. Partial overlap can score above zero even when exact accuracy is zero. |
| **Supporting recall (retrieved)** | Fraction of MuSiQue’s labeled supporting paragraphs present in all retrieved evidence. | 0.5 for both: half of the required support was found on average. |
| **Supporting precision (retrieved)** | Fraction of retrieved paragraphs that are labeled as supporting. | 0.2 for both: most retrieved paragraphs were distractors for these questions. |
| **Hop coverage** | Fraction of known question-decomposition hops whose labeled support paragraph was retrieved. | 0.5 for both. This checks source coverage, not whether the model actually reasoned through each hop. |
| **Supporting recall (answer context)** | Fraction of labeled support that remained in the bounded context shown to the answer model. | 0.5 for both; no average support loss from retrieval to answer context is visible here. |
| **Citation gold-support precision** | Fraction of cited paragraphs labeled as supporting paragraphs. | 1.0 for the one answered case. It is a label match, not a semantic entailment check. |
| **Answerability accuracy** | Whether the system answered answerable questions and abstained on unanswerable ones. | 0.33. All three selected examples are answerable, so the two abstentions count against it. |

### Operational measurements

- **Abstention rate:** share of queries returned as insufficient evidence. Here 2/3 for both systems.
- **Failure rate:** share of query executions that failed, separate from legitimate abstentions. Here 0.
- **Traversal incomplete:** how often graph traversal hit a limit and could not claim a complete component. Here 0.
- **Context tokens:** estimated size of the answer context, counted with a comparison tokenizer. GraphRAG supplies about twice as much text here; this may affect cost and noise.
- **Latency:** query elapsed time reported for each run. Tiny samples and local-model variation make these unsuitable for a performance claim.
- **Metric failures/count:** distinguish a failed judge call from a scored zero, and show how many rows contributed to each average. There were no metric failures in this run.
- **Paired delta:** GraphRAG score minus VectorRAG score for matched cases. It helps compare backends on the same question, but here the answer deltas are based on one scored answer.

## Why is GraphRAG faithfulness 0?

Faithfulness asks whether the answer follows from the text and graph facts supplied to the answer model. The relevant source paragraph says the Learjet 60 is **manufactured by Bombardier Aerospace**. The second supporting paragraph describes Bombardier Aerospace as an aerospace division of Bombardier Inc. GraphRAG answered that **Bombardier Aerospace owns the manufacturer**. That ownership relation is not supported by those passages and changes the relation asked about. The judge scored the answer unfaithful, and MuSiQue’s gold check also rejects it.

This is an answer-generation/evidence-use problem for this row, rather than a traversal timeout: traversal is marked complete, VectorRAG and GraphRAG start from the same seeds, and graph expansion did not improve source-paragraph support recall. The saved graph lines and full prompt/context make it possible to inspect whether the extra graph representation influenced the wording. They do not prove the model’s exact internal cause.

## Why is VectorRAG answer relevance 0?

The VectorRAG answer is exactly **“Bombardier Inc.”**, the gold answer, and has answer accuracy/F1 of 1.0 for that question. RAGAS answer relevance is a separate, reference-free judge score, not a gold-answer comparison. A zero therefore means the RAGAS answer-relevance computation judged the answer/question relationship at the bottom of its scale; it does **not** mean the gold check found the answer irrelevant.

The report records this as a successfully scored metric, so it was not a timeout or missing value. The report alone does not expose the judge’s reasoning for the zero. Given the obvious conflict with the exact gold match, treat it as a metric anomaly to investigate. Check the row’s RAGAS payload and the installed RAGAS 0.4.3 `AnswerRelevancy` inputs/outputs, then compare against a second evaluator or a small manually reviewed sample before presenting this score as a reliable result.

## Can we investigate the red flags and improve results?

Yes, the artifacts make the main pipeline stages inspectable. For each row, compare `retrieval_evidence`, `answer_context`, `answer_prompt`, `response`, `ragas`, `retrieval_gold`, and `answer_context_gold` in `records.jsonl`; the HTML report provides a friendlier view. That can tell whether missing support was never retrieved or was dropped before answering, and show the exact material given to the model. It can also expose noisy graph facts next to the source text.

The most promising next improvements suggested by this run are:

1. **Investigate the two abstentions first.** The traces show questions, ranked evidence, and gold supporting paragraphs. Find whether the missing evidence is absent from the index/graph, ranked below the retrieval limit, or not recognized as enough to answer. Improve search, query decomposition, or answerability policy based on that diagnosis.
2. **Improve graph extraction and graph-to-answer rendering.** The graph did not recover extra gold paragraphs here, used more than twice the context, and still produced an unsupported relation. Validate extracted entities and relation direction/meaning against source text; consider sending only relevant graph facts alongside their source paragraph.
3. **Investigate answer-relevance judging.** Re-run the saved VectorRAG answer/context with a second judge and inspect RAGAS’s generated questions/embeddings for this score. Keep zero as a recorded result until understood; do not silently overwrite it.
4. **Tune context selection for precision and size.** Low supporting precision suggests many retrieved paragraphs are not useful. Better ranking or a smaller, evidence-focused context could reduce noise and model input while preserving support.
5. **Run a larger paired cohort before deciding whether GraphRAG helps.** The architectural goal depends on useful connections between separate support paragraphs. Current per-paragraph graph components cannot add those paragraphs. Evaluate enough multi-hop records after cross-paragraph entity linking/resolution exists, and report retrieval support separately from answer quality.

These are hypotheses and next experiments, not proven fixes. This run has no result showing that graph expansion improves answer quality, and its three-question sample cannot support broad performance or latency claims.

## What should be presented?

For a project presentation, use the report’s overview and per-record comparison, but be explicit that this is a **three-question pilot/debug run**. The most defensible points are:

- both backends shared the same seeds and graph snapshot;
- both abstained on two of three answerable questions;
- VectorRAG answered the one answered question correctly, while GraphRAG gave an unsupported answer;
- both had the same gold support coverage, and GraphRAG used more context;
- the RAGAS answer-relevance score conflicts with VectorRAG’s exact gold match and needs validation.

Use the condition charts only with their scored counts visible. For the answer metrics, say “one answered case” aloud or in the slide. Include a single detailed row showing the gold answer, generated answer, supporting passage, and graph facts to make the faithfulness issue concrete. Avoid presenting the 0.98 GraphRAG answer-relevance score alone as success, or claiming that this run proves VectorRAG is better in general. A larger paired run is needed for that conclusion.

## Files to inspect alongside the report

- `runs/eval-prepared-4-to-6-part-3/report.html` — interactive overview and row-level report.
- `runs/eval-prepared-4-to-6-part-3/summary.json` — aggregate values and denominators.
- `runs/eval-prepared-4-to-6-part-3/records.jsonl` — full per-question traces and evidence.
- `runs/eval-prepared-4-to-6-part-3/manifest.json` — dataset fingerprint, selection, model and index configuration.
- `vectorrag_vs_graphrag_evaluation_ragas.md` — evaluation workflow, score definitions, and operational caveats.
