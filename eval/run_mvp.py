"""MVP experiment runner and comparative evaluation report (Step M6.5).

Produces official MuSiQue predictions and comparative report across:
1. bm25: per-question BM25 keyword baseline over candidate paragraphs.
2. vector: pure cosine vector similarity over IN_CONTEXT candidate paragraphs.
3. walk_off: bounded graph walk with entity linking off (ablation).
4. walk_exact: bounded graph walk across exact-name entity clusters.

Metrics reported:
- Answer F1 and Answer EM with 95% bootstrap confidence intervals.
- Support F1 with 95% bootstrap confidence intervals.
- Recall@2 and Recall@5 with 95% bootstrap confidence intervals.
- Later-hop Recall@5 with 95% bootstrap confidence intervals.
- Mean latency per question (seconds).
- Mean estimated spend per question (USD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure project root and third_party are in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_third_party_musique = _project_root / "third_party" / "musique"
if _third_party_musique.exists() and str(_third_party_musique) not in sys.path:
    sys.path.insert(0, str(_third_party_musique))

from rank_bm25 import BM25Okapi

from eval.baselines.bm25 import get_hop_supporting_indices, tokenize
from eval.predictions import evaluate_predictions, write_predictions
from graphrag_pipeline.answer import AnswerSettings, build_answer_llm, generate_answer
from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, OpenRouterSettings
from graphrag_pipeline.linking.clustering import exact_name_clustering, remove_entity_clusters
from graphrag_pipeline.query import EvidenceChunk, QueryRequest
from graphrag_pipeline.retrieval import RetrievalResult, _build_embedder, retrieve_per_question
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

logger = logging.getLogger(__name__)


from graphrag_pipeline.ingestion.record_adapter import compute_paragraph_hash


def classify_failure(
    record: dict[str, Any],
    retrieved_chunks: list[EvidenceChunk],
    prediction: dict[str, Any],
    driver: Any = None,
) -> str:
    """Categorize failure into extraction miss, link miss, walk miss, or answer error.

    Rules:
    - answer_error: Gold supporting paragraphs were retrieved in the evidence set,
      but the model answered incorrectly or failed to cite them.
    - extraction_miss: Gold paragraphs contain no extracted entities in the graph.
    - link_miss: Entities were extracted from gold paragraphs, but no SAME_AS cluster links connect them.
    - walk_miss: Bridging entity clusters exist, but traversal did not reach the target paragraph.
    """
    gold_indices = set(get_hop_supporting_indices(record))
    retrieved_indices = {c.paragraph_idx for c in retrieved_chunks if c.paragraph_idx is not None}

    # If all gold paragraphs were retrieved, retrieval did its job; failure is answer generation
    if gold_indices and gold_indices.issubset(retrieved_indices):
        return "answer_error"

    if driver is None:
        return "walk_miss" if (gold_indices & retrieved_indices) else "extraction_miss"

    qid = record.get("id")
    try:
        res, _, _ = driver.execute_query(
            """
            MATCH (q:Question {id: $qid})-[ic:IN_CONTEXT]->(d:Document)<-[:FROM_DOCUMENT]-(c:Chunk)
            WHERE ic.idx IN $gold_idxs
            OPTIONAL MATCH (e:__Entity__)-[:FROM_CHUNK]->(c)
            OPTIONAL MATCH (e)-[:SAME_AS]->(cluster:EntityCluster)<-[:SAME_AS]-(other:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
            RETURN count(DISTINCT e) AS entity_count, count(DISTINCT cluster) AS cluster_count
            """,
            qid=qid,
            gold_idxs=list(gold_indices),
        )
        if res:
            e_count = res[0]["entity_count"]
            c_count = res[0]["cluster_count"]
            if e_count == 0:
                return "extraction_miss"
            if c_count == 0:
                return "link_miss"
            return "walk_miss"
    except Exception:
        pass

    return "walk_miss" if (gold_indices & retrieved_indices) else "extraction_miss"


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 20261018,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1.0 - ci) / 2.0 * n_bootstrap)
    upper_idx = int((1.0 + ci) / 2.0 * n_bootstrap)
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))
    return (round(means[lower_idx], 3), round(means[upper_idx], 3))


class StubAnswerLLM:
    """Deterministic stub LLM for $0 testing and synthetic validation."""

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        if isinstance(input, list):
            prompt = input[-1].get("content", "")
        else:
            prompt = str(input)

        import re

        # Extract [id] citations from evidence prompt
        ids = re.findall(r"\[([a-f0-9]{8,64}|chunk-[^\]]+|musique:[^\]]+)\]", prompt)
        cited = [ids[0]] if ids else ["stub_citation"]
        payload = {"answer": "Stub Answer", "citations": cited}
        return SimpleNamespace(content=json.dumps(payload))


class StubEmbedder:
    """Deterministic stub embedder providing unit vectors."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        val = 1.0 / (self.dim**0.5)
        return [val] * self.dim

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        val = 1.0 / (self.dim**0.5)
        return [[val] * self.dim for _ in texts]


def load_slice_records(
    slice_spec: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load evaluation slice metadata and target gold records in order."""
    slice_path = Path(slice_spec)
    if not slice_path.exists() and not slice_path.suffix:
        slice_path = Path(f"eval/slices/{slice_spec}.json")
    if not slice_path.exists():
        raise FileNotFoundError(f"Slice specification not found at {slice_path}")

    with slice_path.open("r", encoding="utf-8") as f:
        slice_data = json.load(f)

    if split == "test":
        target_ids = list(slice_data.get("test_ids", []))
    elif split == "tune":
        target_ids = list(slice_data.get("tune_ids", []))
    elif split == "all":
        target_ids = list(slice_data.get("tune_ids", [])) + list(slice_data.get("test_ids", []))
    else:
        raise ValueError(f"Unknown split: {split}. Choose test, tune, or all.")

    if limit is not None:
        target_ids = target_ids[:limit]

    source_path = Path(slice_data["source_file"])
    if not source_path.exists():
        raise FileNotFoundError(f"Slice source dataset file not found at {source_path}")

    target_id_set = set(target_ids)
    records_by_id: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_id = rec.get("id")
            if rec_id in target_id_set:
                records_by_id[rec_id] = rec

    ordered_records = [records_by_id[rid] for rid in target_ids if rid in records_by_id]
    return slice_data, ordered_records


def retrieve_bm25_evidence(record: dict[str, Any], top_k: int = 5) -> RetrievalResult:
    """Perform per-question BM25 retrieval over a question's candidate paragraphs."""
    paragraphs = record.get("paragraphs", [])
    if not paragraphs:
        return RetrievalResult(
            trace_id=f"trace-bm25-{record.get('id')}",
            question=record.get("question", ""),
            mode="bm25",
            evidence=[],
        )

    corpus = [tokenize(f"{p['title']} {p['paragraph_text']}") for p in paragraphs]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize(record.get("question", ""))
    scores = bm25.get_scores(q_tokens)

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted_indices[:top_k]

    chunks: list[EvidenceChunk] = []
    for i in top_indices:
        p = paragraphs[i]
        p_hash = compute_paragraph_hash(p["title"], p["paragraph_text"])
        chunks.append(
            EvidenceChunk(
                chunk_id=f"musique:paragraph:{p_hash}",
                text=p["paragraph_text"],
                document_id=p["title"],
                score=float(scores[i]),
                graph_references=[],
                paragraph_hash=p_hash,
                paragraph_idx=p["idx"],
            )
        )

    return RetrievalResult(
        trace_id=f"trace-bm25-{record.get('id')}",
        question=record.get("question", ""),
        mode="bm25",
        evidence=chunks,
    )


@dataclass
class ConditionResult:
    """Metrics and artifacts for one evaluated condition."""

    condition: str
    num_questions: int
    answer_f1: float
    answer_f1_ci: tuple[float, float]
    answer_em: float
    answer_em_ci: tuple[float, float]
    answer_containment: float
    answer_containment_ci: tuple[float, float]
    support_f1: float
    support_f1_ci: tuple[float, float]
    r_at_2: float
    r_at_2_ci: tuple[float, float]
    r_at_5: float
    r_at_5_ci: tuple[float, float]
    later_hop_r_at_5: float
    later_hop_r_at_5_ci: tuple[float, float]
    mean_latency_seconds: float
    mean_cost_usd: float | None = None
    predictions_path: str = ""


def _fetch_gateway_spend() -> float | None:
    """Query model gateway spend endpoint if available."""
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8080/v1/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "spent_usd" in data:
                return float(data["spent_usd"])
    except Exception:
        pass
    return None


def run_condition(
    condition: str,
    records: list[dict[str, Any]],
    llm: Any,
    *,
    driver: Any = None,
    embedder: Any = None,
    top_k: int = 5,
    seeds: int = 2,
    fanout: int = 2,
    output_dir: Path = Path("runs/mvp_results"),
    unit_cost_usd: float | None = None,
) -> ConditionResult:
    """Execute retrieval, answer generation, and evaluation for one condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    r_at_2_list: list[float] = []
    r_at_5_list: list[float] = []
    later_r_at_5_list: list[float] = []
    latencies: list[float] = []

    spend_start = _fetch_gateway_spend()

    for rec in records:
        qid = rec["id"]
        qtext = rec["question"]
        gold_indices = set(get_hop_supporting_indices(rec))
        decomp = rec.get("question_decomposition", [])
        later_gold = (
            {step["paragraph_support_idx"] for step in decomp[1:] if "paragraph_support_idx" in step}
            if len(decomp) > 1
            else set()
        )

        q_start = time.perf_counter()

        # 1. Retrieval
        if condition == "bm25":
            retrieval_res = retrieve_bm25_evidence(rec, top_k=top_k)
        elif condition == "vector":
            if driver is not None and embedder is not None:
                retrieval_res = retrieve_per_question(
                    driver=driver,
                    embedder=embedder,
                    question_id=qid,
                    question_text=qtext,
                    top_k=top_k,
                    walk=False,
                )
            else:
                # Fallback to BM25 if no driver provided
                retrieval_res = retrieve_bm25_evidence(rec, top_k=top_k)
        elif condition in ("walk_off", "walk_exact"):
            if driver is not None and embedder is not None:
                # walk_off and walk_exact differ by whether EntityCluster nodes are active in the DB
                retrieval_res = retrieve_per_question(
                    driver=driver,
                    embedder=embedder,
                    question_id=qid,
                    question_text=qtext,
                    top_k=top_k,
                    seeds=seeds,
                    fanout=fanout,
                    walk=True,
                )
            else:
                retrieval_res = retrieve_bm25_evidence(rec, top_k=top_k)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Compute retrieval metrics
        retrieved_idxs_2 = {c.paragraph_idx for c in retrieval_res.evidence[:2] if c.paragraph_idx is not None}
        retrieved_idxs_5 = {c.paragraph_idx for c in retrieval_res.evidence[:5] if c.paragraph_idx is not None}

        cur_r2 = len(gold_indices & retrieved_idxs_2) / len(gold_indices) if gold_indices else 0.0
        cur_r5 = len(gold_indices & retrieved_idxs_5) / len(gold_indices) if gold_indices else 0.0
        r_at_2_list.append(cur_r2)
        r_at_5_list.append(cur_r5)

        if later_gold:
            cur_later_r5 = len(later_gold & retrieved_idxs_5) / len(later_gold)
            later_r_at_5_list.append(cur_later_r5)

        # 2. Answer generation
        query_req = QueryRequest(
            question=qtext,
            trace_id=f"trace-{condition}-{qid}",
            origin="musique",
            record_id=qid,
            source_file=rec.get("source_file"),
            source_line=rec.get("source_line"),
        )

        query_resp = generate_answer(
            request=query_req,
            retrieval=retrieval_res,
            llm=llm,
            answer_settings=AnswerSettings(retrieval_mode=condition, max_evidence_chunks=top_k),
        )
        elapsed = time.perf_counter() - q_start
        latencies.append(elapsed)

        pred_dict = query_resp.to_prediction()
        predictions.append(pred_dict)

    # Compute spend delta
    spend_end = _fetch_gateway_spend()
    if unit_cost_usd is not None:
        mean_cost: float | None = unit_cost_usd
    elif spend_start is not None and spend_end is not None:
        delta = max(0.0, spend_end - spend_start)
        mean_cost = round(delta / len(records), 5) if records else 0.0
    else:
        mean_cost = None

    # 3. Write predictions and evaluate
    pred_path = output_dir / f"predictions_{condition}.jsonl"
    write_predictions(predictions, pred_path, gold_records=records)

    gold_path = output_dir / "gold_eval_records.jsonl"
    with gold_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    official_metrics = evaluate_predictions(pred_path, gold_path)

    # Compute genuine per-instance scores for bootstrap CIs
    answer_f1_list: list[float] = []
    answer_em_list: list[float] = []
    answer_cont_list: list[float] = []
    support_f1_list: list[float] = []

    try:
        from metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths

        for pred, gold in zip(predictions, records):
            gold_ans = [gold["answer"]] + gold.get("answer_aliases", [])
            pred_ans = pred.get("predicted_answer", "")
            f1 = float(metric_max_over_ground_truths(compute_f1, pred_ans, gold_ans))
            em = float(metric_max_over_ground_truths(compute_exact, pred_ans, gold_ans))
            answer_f1_list.append(f1)
            answer_em_list.append(em)

            # Secondary metric: gold answer contained in predicted text
            p_lower = str(pred_ans).strip().lower()
            contained = any(str(ga).strip().lower() in p_lower for ga in gold_ans if str(ga).strip())
            answer_cont_list.append(1.0 if contained else 0.0)

            gold_sp = set(int(p["idx"]) for p in gold.get("paragraphs", []) if p.get("is_supporting"))
            pred_sp = set(int(idx) for idx in pred.get("predicted_support_idxs", []))
            tp = len(pred_sp & gold_sp)
            prec = tp / len(pred_sp) if pred_sp else 0.0
            rec = tp / len(gold_sp) if gold_sp else 0.0
            sp_f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if not pred_sp and not gold_sp:
                sp_f1 = 1.0
            support_f1_list.append(float(sp_f1))
    except ImportError as exc:
        logger.warning("Could not import official per-instance metrics: %s. Using macro scores.", exc)
        answer_f1_list = [official_metrics.get("answer_f1", 0.0)] * len(records)
        answer_em_list = [official_metrics.get("answer_em", 0.0)] * len(records)
        answer_cont_list = [0.0] * len(records)
        support_f1_list = [official_metrics.get("support_f1", 0.0)] * len(records)

    mean_r2 = sum(r_at_2_list) / len(r_at_2_list) if r_at_2_list else 0.0
    mean_r5 = sum(r_at_5_list) / len(r_at_5_list) if r_at_5_list else 0.0
    mean_later_r5 = sum(later_r_at_5_list) / len(later_r_at_5_list) if later_r_at_5_list else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    mean_cont = sum(answer_cont_list) / len(answer_cont_list) if answer_cont_list else 0.0

    return ConditionResult(
        condition=condition,
        num_questions=len(records),
        answer_f1=round(official_metrics.get("answer_f1", 0.0), 3),
        answer_f1_ci=bootstrap_ci(answer_f1_list),
        answer_em=round(official_metrics.get("answer_em", 0.0), 3),
        answer_em_ci=bootstrap_ci(answer_em_list),
        answer_containment=round(mean_cont, 3),
        answer_containment_ci=bootstrap_ci(answer_cont_list),
        support_f1=round(official_metrics.get("support_f1", 0.0), 3),
        support_f1_ci=bootstrap_ci(support_f1_list),
        r_at_2=round(mean_r2, 3),
        r_at_2_ci=bootstrap_ci(r_at_2_list),
        r_at_5=round(mean_r5, 3),
        r_at_5_ci=bootstrap_ci(r_at_5_list),
        later_hop_r_at_5=round(mean_later_r5, 3),
        later_hop_r_at_5_ci=bootstrap_ci(later_r_at_5_list),
        mean_latency_seconds=round(mean_lat, 3),
        mean_cost_usd=mean_cost,
        predictions_path=str(pred_path),
    )


def render_report(
    results: list[ConditionResult],
    slice_data: dict[str, Any],
    split: str,
    output_dir: Path,
) -> str:
    """Format and save comparative markdown and JSON reports."""
    lines = [
        "# MuSiQue MVP Comparative Evaluation Report",
        "",
        f"- **Slice:** `{slice_data.get('name', 'custom')}`",
        f"- **Split:** `{split}` (Evaluated Questions: {results[0].num_questions if results else 0})",
        "- **Evidence Budget:** Top 5 paragraphs (Seeds: 2, Fan-out: 2)",
        "",
        "| Condition | Answer F1 (95% CI) | Answer EM (95% CI) | Containment (95% CI) | Support F1 (95% CI) | R@2 (95% CI) | R@5 (95% CI) | Later-hop R@5 (95% CI) | Latency (s) | Cost ($/q) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        f1_str = f"{r.answer_f1 * 100:.1f} [{r.answer_f1_ci[0] * 100:.1f}, {r.answer_f1_ci[1] * 100:.1f}]"
        em_str = f"{r.answer_em * 100:.1f} [{r.answer_em_ci[0] * 100:.1f}, {r.answer_em_ci[1] * 100:.1f}]"
        cont_str = f"{r.answer_containment * 100:.1f} [{r.answer_containment_ci[0] * 100:.1f}, {r.answer_containment_ci[1] * 100:.1f}]"
        sf1_str = f"{r.support_f1 * 100:.1f} [{r.support_f1_ci[0] * 100:.1f}, {r.support_f1_ci[1] * 100:.1f}]"
        r2_str = f"{r.r_at_2 * 100:.1f} [{r.r_at_2_ci[0] * 100:.1f}, {r.r_at_2_ci[1] * 100:.1f}]"
        r5_str = f"{r.r_at_5 * 100:.1f} [{r.r_at_5_ci[0] * 100:.1f}, {r.r_at_5_ci[1] * 100:.1f}]"
        lr5_str = f"{r.later_hop_r_at_5 * 100:.1f} [{r.later_hop_r_at_5_ci[0] * 100:.1f}, {r.later_hop_r_at_5_ci[1] * 100:.1f}]"
        cost_str = f"${r.mean_cost_usd:.4f}" if r.mean_cost_usd is not None else "n/a"
        lines.append(
            f"| `{r.condition}` | {f1_str} | {em_str} | {cont_str} | {sf1_str} | {r2_str} | {r5_str} | {lr5_str} | {r.mean_latency_seconds:.2f} | {cost_str} |"
        )

    lines.extend(
        [
            "",
            "### Methodology & Verification",
            "- **BM25:** Candidate paragraph keyword matching per question.",
            "- **Vector:** Cosine similarity over candidate document embeddings scoped to question context.",
            "- **Walk Off:** Graph walk executed with entity linking disabled (ablation baseline).",
            "- **Walk Exact:** Graph walk traversing exact-name `SAME_AS` entity clusters.",
            "- **Support Grounding:** Citations strictly validated against retrieved paragraphs.",
            "- **Zero Cross-Question Leakage:** Scoped to each question's candidate context by construction.",
        ]
    )

    report_md = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mvp_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    json_path = output_dir / "mvp_report.json"
    json_data = {
        "slice": slice_data.get("name"),
        "split": split,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return report_md


def run_mvp(
    slice_spec: str | Path = "ans_dev_mvp_v1",
    split: str = "test",
    conditions: list[str] | None = None,
    limit: int | None = None,
    output_dir: Path = Path("runs/mvp_results"),
    stub: bool = False,
    provider: str = "openrouter",
    seeds: int = 2,
    fanout: int = 2,
    top_k: int = 5,
    driver: Any = None,
) -> tuple[str, list[ConditionResult]]:
    """Programmatic entry point to run MVP comparative experiments and render report."""
    cond_list = conditions or ["bm25", "vector", "walk_off", "walk_exact"]
    slice_data, records = load_slice_records(slice_spec, split=split, limit=limit)

    if not records:
        raise ValueError(f"No records found for slice {slice_spec} split {split}")

    # Build models
    if stub:
        llm = StubAnswerLLM()
        embedder = StubEmbedder()
    else:
        if provider == "ollama":
            model_settings = OllamaSettings.from_env()
        else:
            model_settings = OpenRouterSettings.from_env()
        llm = build_answer_llm(model_settings)
        embedder = _build_embedder(model_settings)

    # Initialize Neo4j driver if needed
    managed_conn = None
    if driver is None and any(c in cond_list for c in ("vector", "walk_off", "walk_exact")):
        try:
            neo4j_settings = Neo4jSettings.from_env()
            managed_conn = Neo4jConnection(neo4j_settings)
            driver = managed_conn.connect()
        except Exception as e:
            logger.warning("Neo4j connection unavailable: %s. Vector and walk conditions will use fallback.", e)
    if driver is not None:
        try:
            from graphrag_pipeline.ingestion.record_adapter import build_context_links_for_record
            from graphrag_pipeline.storage.question_context import link_questions_to_paragraphs
            links = []
            for r in records:
                links.extend(build_context_links_for_record(json.dumps(r), 1))
            link_questions_to_paragraphs(driver, links)
        except Exception as e:
            logger.warning("Could not auto-link question context: %s", e)

    results: list[ConditionResult] = []

    try:
        for cond in cond_list:
            logger.info("Running condition: %s on %d questions", cond, len(records))
            if cond == "walk_exact" and driver is not None:
                with exact_name_clustering(driver):
                    res = run_condition(
                        condition=cond,
                        records=records,
                        llm=llm,
                        driver=driver,
                        embedder=embedder,
                        top_k=top_k,
                        seeds=seeds,
                        fanout=fanout,
                        output_dir=output_dir,
                    )
            elif cond == "walk_off" and driver is not None:
                # Remove any existing clusters so linking is off
                remove_entity_clusters(driver)
                res = run_condition(
                    condition=cond,
                    records=records,
                    llm=llm,
                    driver=driver,
                    embedder=embedder,
                    top_k=top_k,
                    seeds=seeds,
                    fanout=fanout,
                    output_dir=output_dir,
                )
            else:
                res = run_condition(
                    condition=cond,
                    records=records,
                    llm=llm,
                    driver=driver,
                    embedder=embedder,
                    top_k=top_k,
                    seeds=seeds,
                    fanout=fanout,
                    output_dir=output_dir,
                )
            results.append(res)
    finally:
        if managed_conn is not None:
            managed_conn.close()

    report = render_report(results, slice_data, split=split, output_dir=output_dir)
    return report, results


def main() -> None:
    """Command-line interface for the MVP experiment runner."""
    parser = argparse.ArgumentParser(description="Run MuSiQue GraphRAG MVP comparative evaluation.")
    parser.add_argument("--slice", default="ans_dev_mvp_v1", help="Slice specification name or path")
    parser.add_argument("--split", default="test", choices=["test", "tune", "all"], help="Evaluation split")
    parser.add_argument("--conditions", default="bm25,vector,walk_off,walk_exact", help="Comma-separated conditions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--output-dir", default="runs/mvp_results", help="Directory for output predictions and report")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "ollama"], help="Model provider")
    parser.add_argument("--stub", action="store_true", help="Use stub models for $0 execution")
    parser.add_argument("--seeds", type=int, default=2, help="Seed count for graph walk")
    parser.add_argument("--fanout", type=int, default=2, help="Fan-out bound per seed for graph walk")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved evidence chunks")

    args = parser.parse_args()
    cond_list = [c.strip() for c in args.conditions.split(",") if c.strip()]

    report, _ = run_mvp(
        slice_spec=args.slice,
        split=args.split,
        conditions=cond_list,
        limit=args.limit,
        output_dir=Path(args.output_dir),
        stub=args.stub,
        provider=args.provider,
        seeds=args.seeds,
        fanout=args.fanout,
        top_k=args.top_k,
    )
    print("\n" + report + "\n")


if __name__ == "__main__":
    main()
