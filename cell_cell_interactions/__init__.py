"""Multicellular Transformer package.

Scanpy-style entry points::

    import cell_cell_interactions as cci
    mct = cci.load("/models/mct_v1")
    emb = mct.gene_embeddings()
    ranking = cci.tl.rank_similar_genes(emb, "EGFR")
    cci.pl.similarity_ranking(ranking)
"""

__version__ = "0.0.1"

from cell_cell_interactions.api import pl, pp, tl
from cell_cell_interactions.api.model import (
    MASK_TOKEN_ID,
    N_SPECIAL_TOKENS,
    PAD_TOKEN_ID,
    MCTIndex,
    MCTModel,
    load,
)

__all__ = [
    "__version__",
    "MCTIndex",
    "MCTModel",
    "load",
    "pl",
    "pp",
    "tl",
    "PAD_TOKEN_ID",
    "MASK_TOKEN_ID",
    "N_SPECIAL_TOKENS",
]
