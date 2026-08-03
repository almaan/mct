"""Plotting utilities.

Every function returns the ``matplotlib.axes.Axes`` (or list of Axes) it
drew on, and never calls ``plt.show()`` — scanpy-style, so users compose
subplots and control display themselves.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

# Re-export the legacy plot functions so they show up in one place.
from cell_cell_interactions.pl.condition_specific_genes import (
    plot_similarity_across_conditions_heatmap,
)
from cell_cell_interactions.pl.grn import plot_gene_set

__all__ = [
    "plot_gene_set",
    "plot_similarity_across_conditions_heatmap",
    "similarity_ranking",
    "similarity_heatmap",
]


def similarity_ranking(
    ranking: pd.DataFrame,
    top_n: int = 20,
    ax=None,
    palette: str = "magma_r",
    gene_col: str = "gene",
    score_col: str = "similarity",
    title: str | None = None,
):
    """Bar plot of the top-N most-similar genes to a query.

    Parameters
    ----------
    ranking:
        DataFrame produced by ``cci.tl.rank_similar_genes``.
    top_n:
        Number of rows to show.
    ax:
        Existing ``matplotlib.axes.Axes``. If None, creates a new figure.
    palette:
        Seaborn palette name.
    gene_col, score_col:
        Column names in ``ranking``.
    title:
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> emb = mct.gene_embeddings()
    >>> ranking = cci.tl.rank_similar_genes(emb, "EGFR", top_n=20)
    >>> ax = cci.pl.similarity_ranking(ranking, title="EGFR neighbors")
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    top = ranking.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, max(2, 0.3 * top_n)))
    sns.barplot(data=top, x=score_col, y=gene_col, palette=palette, ax=ax)
    if title:
        ax.set_title(title)
    return ax


def similarity_heatmap(
    emb: pd.DataFrame,
    genes: Iterable[str] | None = None,
    ax=None,
    cmap: str = "vlag",
    figsize: tuple[float, float] = (6, 5),
    annotate: bool = False,
):
    """Cosine-similarity heatmap over a set of genes.

    Parameters
    ----------
    emb:
        Embedding matrix.
    genes:
        Iterable of gene symbols to plot. Defaults to the full index (only
        practical for small sets).
    ax:
        Existing Axes; if None, a new figure is created.
    cmap, figsize, annotate:
        Passed to seaborn / matplotlib.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> cci.pl.similarity_heatmap(emb, genes=["EGFR", "ERBB2", "ERBB3", "ERBB4"])
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    if genes is not None:
        keep = [g for g in genes if g in emb.index]
        emb = emb.loc[keep]

    if emb.shape[0] > 200:
        warnings.warn(
            f"similarity_heatmap with {emb.shape[0]} genes; consider subsetting.",
            stacklevel=2,
        )

    sim = cosine_similarity(emb.values)
    sim_df = pd.DataFrame(sim, index=emb.index, columns=emb.index)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(sim_df, cmap=cmap, center=0, annot=annotate, ax=ax, cbar_kws={"label": "cosine"})
    return ax
