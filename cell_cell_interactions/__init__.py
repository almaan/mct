"""Multicellular Transformer package."""

__version__ = "0.0.1"

from cell_cell_interactions.api import (
    AdataInferencePreset,
    CellarrInferencePreset,
    GeneEmbedding,
    MCTIndex,
    MCTModel,
    SpatialInferencePreset,
    calculate_condition_specificity,
    embed_adata,
    embed_dataloader,
    get_context_aware_gene_embeddings,
    get_most_common_neighbors,
    jaccard_similarity_across_items,
    load_model,
    plot_gene_set,
    plot_similarity_across_conditions_heatmap,
)

__all__ = [
    "__version__",
    "AdataInferencePreset",
    "CellarrInferencePreset",
    "GeneEmbedding",
    "MCTIndex",
    "MCTModel",
    "SpatialInferencePreset",
    "calculate_condition_specificity",
    "embed_adata",
    "embed_dataloader",
    "get_context_aware_gene_embeddings",
    "get_most_common_neighbors",
    "jaccard_similarity_across_items",
    "load_model",
    "plot_gene_set",
    "plot_similarity_across_conditions_heatmap",
]
