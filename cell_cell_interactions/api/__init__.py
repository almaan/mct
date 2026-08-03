"""Public API for the Multicellular Transformer package.

Import from here rather than reaching into ``cell_cell_interactions.modeling``
or ``cell_cell_interactions.pp`` directly. The lower-level modules will keep
working but their signatures are subject to internal refactoring.

Example
-------
>>> from cell_cell_interactions.api import load_model, CellarrInferencePreset
>>> model = load_model("checkpoints/multicellular_transformer.ckpt")
>>> embeddings = model.gene_embeddings(["EGFR", "TP53"])
>>> preset = CellarrInferencePreset(
...     cell_tdb_uri="/path/to/cell_metadata",
...     gene_tdb_uri="/path/to/gene_metadata",
...     counts_tdb_uri="/path/to/counts",
... )
>>> from cell_cell_interactions.api import embed_adata
>>> results = embed_adata(model, preset, max_batches=5)
"""

from cell_cell_interactions.api.analysis import (
    GeneEmbedding,
    calculate_condition_specificity,
    get_most_common_neighbors,
    jaccard_similarity_across_items,
)
from cell_cell_interactions.api.data import (
    AdataInferencePreset,
    CellarrInferencePreset,
    SpatialInferencePreset,
)
from cell_cell_interactions.api.inference import (
    embed_adata,
    embed_dataloader,
    get_context_aware_gene_embeddings,
)
from cell_cell_interactions.api.model import (
    MCTIndex,
    MCTModel,
    load_model,
)
from cell_cell_interactions.api.plotting import (
    plot_gene_set,
    plot_similarity_across_conditions_heatmap,
)

__all__ = [
    # model
    "MCTIndex",
    "MCTModel",
    "load_model",
    # data
    "AdataInferencePreset",
    "CellarrInferencePreset",
    "SpatialInferencePreset",
    # inference
    "embed_adata",
    "embed_dataloader",
    "get_context_aware_gene_embeddings",
    # analysis
    "GeneEmbedding",
    "calculate_condition_specificity",
    "get_most_common_neighbors",
    "jaccard_similarity_across_items",
    # plotting
    "plot_gene_set",
    "plot_similarity_across_conditions_heatmap",
]
