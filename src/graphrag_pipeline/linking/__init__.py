"""Entity linking and cluster management for GraphRAG.

Provides reversible exact-name clustering for entity resolution and graph walks.
"""

from __future__ import annotations

from graphrag_pipeline.linking.clustering import (
    ClusterSummary,
    create_exact_name_clusters,
    exact_name_clustering,
    remove_entity_clusters,
)

__all__ = [
    "ClusterSummary",
    "create_exact_name_clusters",
    "exact_name_clustering",
    "remove_entity_clusters",
]
