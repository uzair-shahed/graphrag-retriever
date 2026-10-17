"""Shared answer prompt policy and model-visible evidence rendering.

Last modified: 2026-10-17

Purpose of this file:
    Define the single answer instructions, JSON output contract wording, and
    evidence-to-text rendering used by both retrieval backends, so VectorRAG
    (application-owned prompt) and GraphRAG (Neo4j-managed ``GraphRAG.search``)
    show the answer model the same instructions and the same context format.

Files that use this file:
    ``graphrag_pipeline.answer`` builds the VectorRAG prompt from these
    helpers. ``graphrag_pipeline.retrieval`` uses them as the GraphRAG
    ``RagTemplate`` and to render each retrieved row's model-visible content.

Caveats:
    The template must not contain literal braces because it is formatted with
    ``str.format``. Rendering lists graph entities and relationships by display
    name only; it never includes MuSiQue gold answers or supporting labels.
    The contract has no field for the model to declare "insufficient
    evidence"; that status is assigned only when retrieval returns nothing.
"""

from __future__ import annotations

from graphrag_pipeline.query import EvidenceChunk

ANSWER_SYSTEM_INSTRUCTIONS = (
    "You answer questions using only the supplied evidence and respond with "
    "valid JSON only."
)

ANSWER_PROMPT_TEMPLATE = (
    "Answer the question using only the graph evidence below. "
    "Return valid JSON with exactly two fields: answer (string) and "
    "citations (array of chunk_id strings). Every citation must be a chunk_id "
    "shown in square brackets in the evidence and must identify evidence "
    "actually used in the answer; do not cite chunks that were not used. "
    "If the evidence does not support an answer, do not guess.\n\n"
    "Question: {query_text}\n\n"
    "Graph evidence:\n{context}"
)


def _entity_label(node_id: str, names: dict[str, str]) -> str:
    """Return the display label for a graph node, or a marker if it is unknown."""
    return names.get(node_id, "unknown entity")


def render_evidence_chunk(chunk: EvidenceChunk) -> str:
    """Render one evidence chunk, with its entity subgraph when present.

    Args:
        chunk: Evidence chunk whose ``chunk_id`` is the only valid citation key.

    Returns:
        A bracketed chunk ID and text, followed by graph entity and relationship
        lines when the chunk carries them. A truncated component is labelled so
        the model is never told it saw a complete component.
    """
    lines = [f"[{chunk.chunk_id}] {chunk.text}"]
    names = {
        node.node_id: node.name or (node.labels[0] if node.labels else "entity")
        for node in chunk.graph_nodes
    }
    if chunk.graph_nodes:
        entities = [
            f"{names[node.node_id]} ({', '.join(node.labels)})" if node.labels else names[node.node_id]
            for node in chunk.graph_nodes
        ]
        lines.append(f"Entities in this chunk: {'; '.join(entities)}")
    if chunk.graph_relationships:
        relations = [
            f"{_entity_label(rel.start_node_id, names)} -[{rel.type}]-> {_entity_label(rel.end_node_id, names)}"
            for rel in chunk.graph_relationships
        ]
        lines.append(f"Relationships in this chunk: {'; '.join(relations)}")
    if (chunk.graph_nodes or chunk.graph_relationships) and not chunk.graph_component_complete:
        lines.append("Note: the entity graph for this chunk was truncated by a size limit.")
    return "\n".join(lines)


def render_evidence_context(evidence: list[EvidenceChunk]) -> str:
    """Render evidence exactly as the answer model sees it.

    Args:
        evidence: Bounded evidence list; order is preserved.

    Returns:
        One block per chunk, each ending in a newline so that joining blocks with
        a single newline (as ``GraphRAG.search`` does) yields blank-line
        separation identical to the VectorRAG prompt.
    """
    return "\n".join(f"{render_evidence_chunk(chunk)}\n" for chunk in evidence)
