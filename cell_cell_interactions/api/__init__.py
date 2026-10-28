"""Public API for the Multicellular Transformer package (scanpy-style).

Users interact with three namespaces:

- :func:`cell_cell_interactions.load` — one call to get an :class:`MCTModel`.
- :mod:`cell_cell_interactions.tl` — tools that turn embeddings into tables.
- :mod:`cell_cell_interactions.pl` — plots.
- :mod:`cell_cell_interactions.pp` — preprocessing helpers for AnnData input.

Quick tour
----------
>>> import cell_cell_interactions as cci
>>> mct = cci.load("/models/mct_v1", device="cuda")
>>> emb = mct.gene_embeddings()                            # DataFrame (genes x dims)
>>> ranking = cci.tl.rank_similar_genes(emb, "EGFR", 20)   # DataFrame
>>> ax = cci.pl.similarity_ranking(ranking)                # Axes
>>> sample_embs = mct.embed(adata,                         # AnnData in
...                         cell_type_key="cell_type",
...                         sample_key="donor_id")
>>> per_donor_egfr = {k: df.loc["EGFR"] for k, df in sample_embs.items()}
"""

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
