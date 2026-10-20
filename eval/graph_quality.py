"""Graph quality and bridge coverage evaluation metrics (Step M5.3).

Evaluates the structural quality of an extracted Neo4j knowledge graph:
- entities per paragraph distribution (mean, min, max)
- entity type mix
- paragraphs with zero entities
- relationship type histogram
- largest name groups (most frequent entity names)
- bridge coverage: whether an entity matching the gold MuSiQue bridge string
  exists in both linked paragraphs (the upper bound for multi-hop graph walks).

Note: This evaluation module is located under eval/ and uses gold decompositions
exclusively for offline graph assessment. It must never be imported by runtime
graphrag_pipeline/ modules.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neo4j import Driver

logger = logging.getLogger(__name__)


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for robust matching: lowercase, strip articles and punctuation."""
    s = name.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def extract_bridge_links(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract gold bridge links from a MuSiQue record decomposition.

    Each hop k referencing #{j} defines a bridge: the answer of hop j must connect
    hop j's supporting paragraph to hop k's supporting paragraph.
    """
    decomp = record.get("question_decomposition", [])
    bridges: list[dict[str, Any]] = []

    for k, hop in enumerate(decomp):
        refs = re.findall(r"#(\d+)", hop.get("question", ""))
        for ref_str in refs:
            j = int(ref_str)
            if 1 <= j <= len(decomp):
                source_hop = decomp[j - 1]
                bridge_answer = source_hop.get("answer")
                p_src = source_hop.get("paragraph_support_idx")
                p_tgt = hop.get("paragraph_support_idx")
                if bridge_answer and p_src is not None and p_tgt is not None:
                    bridges.append(
                        {
                            "record_id": record.get("id", ""),
                            "bridge_entity": bridge_answer,
                            "hop_source_idx": int(p_src),
                            "hop_target_idx": int(p_tgt),
                            "source_hop_id": source_hop.get("id"),
                            "target_hop_id": hop.get("id"),
                        }
                    )
    return bridges


@dataclass
class GraphQualityReport:
    """Structured graph quality and bridge coverage report."""

    total_paragraphs: int
    paragraphs_with_zero_entities: int
    zero_entities_ratio: float
    entities_per_paragraph: dict[str, float]
    type_mix: dict[str, int]
    relation_histogram: dict[str, int]
    largest_name_groups: list[dict[str, Any]]
    total_bridge_links: int
    covered_bridge_links: int
    bridge_coverage_ratio: float
    bridge_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_graph_quality(
    driver: Driver,
    dataset_records: list[dict[str, Any]] | None = None,
    database: str = "neo4j",
) -> GraphQualityReport:
    """Analyze the extracted graph in Neo4j and compute quality and bridge coverage metrics.

    Args:
        driver: Active Neo4j driver.
        dataset_records: Optional list of MuSiQue records with gold decompositions.
            When provided, calculates bridge coverage for the questions in the graph.
        database: Target Neo4j database name.

    Returns:
        GraphQualityReport containing all summary metrics and bridge breakdowns.
    """
    # 1. Paragraph counts and entity distribution
    doc_count_query = """
    MATCH (d:Document)
    OPTIONAL MATCH (e:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(d)
    RETURN d.paragraph_hash as hash, count(distinct e) as entity_count
    """
    doc_records, _, _ = driver.execute_query(doc_count_query, database_=database)

    total_paragraphs = len(doc_records)
    counts = [r["entity_count"] for r in doc_records] if doc_records else [0]
    zero_counts = sum(1 for c in counts if c == 0)
    zero_ratio = (zero_counts / total_paragraphs) if total_paragraphs > 0 else 0.0

    entities_dist = {
        "mean": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "min": float(min(counts)) if counts else 0.0,
        "max": float(max(counts)) if counts else 0.0,
    }

    # 2. Entity type mix
    type_query = """
    MATCH (e:__Entity__)
    UNWIND [l in labels(e) WHERE not l in ['__KGBuilder__', '__Entity__']] as lbl
    RETURN lbl, count(e) as count ORDER BY count DESC
    """
    type_records, _, _ = driver.execute_query(type_query, database_=database)
    type_mix = {r["lbl"]: int(r["count"]) for r in type_records}

    # 3. Relationship histogram
    rel_query = """
    MATCH (:__Entity__)-[r]->(:__Entity__)
    RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC
    """
    rel_records, _, _ = driver.execute_query(rel_query, database_=database)
    relation_histogram = {r["rel_type"]: int(r["count"]) for r in rel_records}

    # 4. Largest name groups
    name_query = """
    MATCH (e:__Entity__)
    RETURN e.name as name, count(e) as count ORDER BY count DESC LIMIT 10
    """
    name_records, _, _ = driver.execute_query(name_query, database_=database)
    largest_name_groups = [
        {"name": r["name"], "count": int(r["count"])}
        for r in name_records
        if r["name"] is not None
    ]

    # 5. Bridge coverage
    total_bridges = 0
    covered_bridges = 0
    bridge_details: list[dict[str, Any]] = []

    if dataset_records:
        entity_map_query = """
        MATCH (e:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(d:Document)
        WHERE d.record_id is not null AND d.paragraph_idx is not null
        RETURN d.record_id as record_id, toString(d.paragraph_idx) as paragraph_idx, collect(distinct e.name) as entities
        """
        map_records, _, _ = driver.execute_query(entity_map_query, database_=database)
        entity_map: dict[tuple[str, str], list[str]] = {
            (r["record_id"], r["paragraph_idx"]): [
                normalize_entity_name(x) for x in r["entities"] if x
            ]
            for r in map_records
        }

        for record in dataset_records:
            rec_id = record.get("id", "")
            bridges = extract_bridge_links(record)
            for b in bridges:
                total_bridges += 1
                norm_b = normalize_entity_name(b["bridge_entity"])
                src_key = (rec_id, str(b["hop_source_idx"]))
                tgt_key = (rec_id, str(b["hop_target_idx"]))

                src_entities = entity_map.get(src_key, [])
                tgt_entities = entity_map.get(tgt_key, [])

                src_match = any(
                    norm_b == e or norm_b in e or e in norm_b for e in src_entities
                )
                tgt_match = any(
                    norm_b == e or norm_b in e or e in norm_b for e in tgt_entities
                )
                is_covered = src_match and tgt_match
                if is_covered:
                    covered_bridges += 1

                bridge_details.append(
                    {
                        "record_id": rec_id,
                        "bridge_entity": b["bridge_entity"],
                        "hop_source_idx": b["hop_source_idx"],
                        "hop_target_idx": b["hop_target_idx"],
                        "source_has_entity": src_match,
                        "target_has_entity": tgt_match,
                        "covered": is_covered,
                    }
                )

    bridge_coverage_ratio = (
        (covered_bridges / total_bridges) if total_bridges > 0 else 0.0
    )

    return GraphQualityReport(
        total_paragraphs=total_paragraphs,
        paragraphs_with_zero_entities=zero_counts,
        zero_entities_ratio=round(zero_ratio, 4),
        entities_per_paragraph=entities_dist,
        type_mix=type_mix,
        relation_histogram=relation_histogram,
        largest_name_groups=largest_name_groups,
        total_bridge_links=total_bridges,
        covered_bridge_links=covered_bridges,
        bridge_coverage_ratio=round(bridge_coverage_ratio, 4),
        bridge_details=bridge_details,
    )


def render_graph_quality_report(report: GraphQualityReport) -> str:
    """Render a human-readable markdown summary table of graph quality metrics."""
    lines = [
        "# Graph Quality & Bridge Coverage Report",
        "",
        "## Summary",
        f"- **Total Paragraphs (Documents):** {report.total_paragraphs}",
        f"- **Paragraphs with 0 Entities:** {report.paragraphs_with_zero_entities} ({report.zero_entities_ratio * 100:.1f}%)",
        f"- **Entities per Paragraph:** mean={report.entities_per_paragraph['mean']}, min={report.entities_per_paragraph['min']}, max={report.entities_per_paragraph['max']}",
        f"- **Bridge Links Evaluated:** {report.total_bridge_links}",
        f"- **Bridge Links Covered:** {report.covered_bridge_links} ({report.bridge_coverage_ratio * 100:.1f}%)",
        "",
        "## Entity Type Mix",
        "| Entity Type | Count | Share |",
        "|-------------|-------|-------|",
    ]
    total_types = sum(report.type_mix.values()) or 1
    for t_name, count in sorted(report.type_mix.items(), key=lambda x: x[1], reverse=True):
        share = count / total_types * 100
        lines.append(f"| {t_name} | {count} | {share:.1f}% |")

    lines.extend([
        "",
        "## Relationship Histogram",
        "| Relationship Type | Count |",
        "|-------------------|-------|",
    ])
    for r_name, count in sorted(report.relation_histogram.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {r_name} | {count} |")

    lines.extend([
        "",
        "## Largest Name Groups",
        "| Entity Name | Frequency |",
        "|-------------|-----------|",
    ])
    for ng in report.largest_name_groups:
        lines.append(f"| {ng['name']} | {ng['count']} |")

    return "\n".join(lines)
