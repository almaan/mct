"""Tool functions — analyses over gene embeddings.

Scanpy-style: functions take a DataFrame (or AnnData) and return one.
Nothing here touches the model directly; call ``cci.load(...).gene_embeddings()``
or ``cci.load(...).embed(adata)`` first, then feed the result into these.

Every function ships with a runnable ``Examples`` block in its docstring.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity_full
from sklearn.neighbors import NearestNeighbors

# Downstream tools re-exported so users can find them in one namespace.
from cell_cell_interactions.tl.condition_specific_genes import (
    calculate_condition_specificity,
    get_most_common_neighbors,
    jaccard_similarity_across_items,
)
from cell_cell_interactions.tl.grn import GeneEmbedding

__all__ = [
    # Reused paper utilities
    "GeneEmbedding",
    "calculate_condition_specificity",
    "get_most_common_neighbors",
    "jaccard_similarity_across_items",
    # New scanpy-style helpers
    "pairwise_cosine_similarity",
    "rank_similar_genes",
    "gene_neighbors",
    "regress_out",
]


# ---------------------------------------------------------------------- #
# Pairwise similarity
# ---------------------------------------------------------------------- #


def pairwise_cosine_similarity(
    emb: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    pairs_a: list[str] | None = None,
    pairs_b: list[str] | None = None,
) -> pd.DataFrame:
    """Score cosine similarity between named pairs of embeddings.

    Two calling styles:

    - ``pairs=[("EGFR", "EGF"), ("KIT", "KITLG"), ...]``
    - ``pairs_a=["EGFR", "KIT"], pairs_b=["EGF", "KITLG"]``

    Missing symbols on either side of a pair are silently dropped.

    Parameters
    ----------
    emb:
        Embedding matrix. Rows are entities (typically genes), columns are
        embedding dimensions.
    pairs, pairs_a, pairs_b:
        See above.

    Returns
    -------
    pandas.DataFrame
        Columns ``a``, ``b``, ``pair`` (``"{a}_{b}"``), ``score``.

    Examples
    --------
    >>> import cell_cell_interactions as cci
    >>> mct = cci.load("/models/mct_v1")
    >>> emb = mct.gene_embeddings()
    >>> cci.tl.pairwise_cosine_similarity(
    ...     emb,
    ...     pairs=[("EGFR", "EGF"), ("KIT", "KITLG")],
    ... )
          a      b        pair     score
    0  EGFR    EGF    EGFR_EGF  0.63...
    1   KIT  KITLG   KIT_KITLG  0.71...
    """
    if pairs is None:
        if pairs_a is None or pairs_b is None:
            raise ValueError("Provide either `pairs` or both `pairs_a` and `pairs_b`.")
        if len(pairs_a) != len(pairs_b):
            raise ValueError("`pairs_a` and `pairs_b` must have the same length.")
        pairs = list(zip(pairs_a, pairs_b))

    kept = [(a, b) for a, b in pairs if a in emb.index and b in emb.index]
    if not kept:
        return pd.DataFrame(columns=["a", "b", "pair", "score"])

    a_list, b_list = map(list, zip(*kept))
    a_mat = emb.loc[a_list].values
    b_mat = emb.loc[b_list].values
    score = _rowwise_cosine(a_mat, b_mat)
    return pd.DataFrame(
        {
            "a": a_list,
            "b": b_list,
            "pair": [f"{a}_{b}" for a, b in kept],
            "score": score,
        }
    )


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    out = dot / denom
    return np.nan_to_num(out, nan=0.0)


# ---------------------------------------------------------------------- #
# Rank similar genes to a query
# ---------------------------------------------------------------------- #


def rank_similar_genes(
    emb: pd.DataFrame,
    query: str | np.ndarray | pd.Series,
    top_n: int = 20,
    subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rank genes by cosine similarity to a query gene or vector.

    Parameters
    ----------
    emb:
        Embedding matrix.
    query:
        Either a gene symbol present in ``emb.index`` or a raw vector of
        length ``emb.shape[1]``.
    top_n:
        Number of neighbors to return.
    subset:
        Optional iterable of gene symbols to restrict the ranking to.

    Returns
    -------
    pandas.DataFrame
        Columns ``gene``, ``similarity`` sorted descending.

    Examples
    --------
    >>> emb = mct.gene_embeddings()
    >>> cci.tl.rank_similar_genes(emb, "EGFR", top_n=5)
        gene  similarity
    0   EGFR    1.0000
    1  ERBB2    0.8123
    ...
    """
    if isinstance(query, str):
        if query not in emb.index:
            raise KeyError(f"query gene '{query}' not in embedding index")
        vec = emb.loc[query].values
    else:
        vec = np.asarray(query, dtype=np.float64)
        if vec.shape[-1] != emb.shape[1]:
            raise ValueError(
                f"query vector dim {vec.shape[-1]} != embedding dim {emb.shape[1]}"
            )

    target = emb.loc[[g for g in subset if g in emb.index]] if subset is not None else emb
    sims = _cosine_similarity_full(vec.reshape(1, -1), target.values)[0]
    out = pd.DataFrame({"gene": target.index.tolist(), "similarity": sims})
    return out.sort_values("similarity", ascending=False, kind="mergesort").head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------- #
# k-nearest-neighbor gene graph
# ---------------------------------------------------------------------- #


def gene_neighbors(
    emb: pd.DataFrame,
    n_neighbors: int = 20,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Compute the k-nearest-neighbor gene table.

    Parameters
    ----------
    emb:
        Embedding matrix.
    n_neighbors:
        Number of neighbors per gene (excludes self).
    metric:
        Any distance metric supported by ``sklearn.neighbors.NearestNeighbors``.

    Returns
    -------
    pandas.DataFrame
        Long-form frame with columns ``gene``, ``neighbor``, ``rank``,
        ``distance``. ``rank`` is 1-indexed (self is not included).

    Examples
    --------
    >>> emb = mct.gene_embeddings()
    >>> nbrs = cci.tl.gene_neighbors(emb, n_neighbors=10)
    >>> nbrs[nbrs["gene"] == "EGFR"].head()
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(emb.values)
    distances, indices = nn.kneighbors(emb.values)
    rows = []
    genes = emb.index.tolist()
    for i, gene in enumerate(genes):
        for rank in range(1, n_neighbors + 1):  # skip self at rank 0
            j = int(indices[i, rank])
            rows.append(
                {
                    "gene": gene,
                    "neighbor": genes[j],
                    "rank": rank,
                    "distance": float(distances[i, rank]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Regress out batch/study covariates from a score column
# ---------------------------------------------------------------------- #


def regress_out(
    scores: pd.DataFrame,
    regress_out: str | list[str],
    keep: list[str],
    score_col: str = "score",
    adjusted_col: str = "score_adj",
    fisher_transform: bool = True,
) -> pd.DataFrame:
    """Regress out categorical batch/study effects from a score column.

    Fits OLS on a Fisher-transformed score (default) against covariates
    that have been orthogonalized against ``keep``. Returns a copy of
    ``scores`` with ``adjusted_col`` added.

    Parameters
    ----------
    scores:
        DataFrame containing ``score_col``, every ``regress_out`` column,
        and every ``keep`` column.
    regress_out:
        Categorical column(s) whose effect should be removed.
    keep:
        Categorical column(s) to control for (kept in the model).
    score_col:
        Name of the column with the raw scores.
    adjusted_col:
        Name of the new column to add with adjusted scores.
    fisher_transform:
        Fisher-z the input before regression and inverse-tanh the output.
        Appropriate when ``score_col`` is a correlation in ``[-1, 1]``.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> adjusted = cci.tl.regress_out(
    ...     scores, regress_out="study", keep=["tissue"],
    ... )
    """
    try:
        import statsmodels.api as sm
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "cci.tl.regress_out requires statsmodels. Install it with "
            "`pip install statsmodels`."
        ) from err

    out = scores.copy()
    if isinstance(regress_out, str):
        regress_out = [regress_out]

    batch_dummies = pd.get_dummies(out[regress_out], drop_first=True).fillna(0).astype(float)
    keep_dummies = pd.get_dummies(out[keep], drop_first=True).fillna(0).astype(float)
    keep_design = sm.add_constant(keep_dummies)

    y = out[score_col].astype(float)
    if fisher_transform:
        eps = 1e-6
        y = np.arctanh(y.clip(-1 + eps, 1 - eps))

    Q, _ = np.linalg.qr(keep_design.to_numpy())
    batch_ortho = batch_dummies.to_numpy() - Q @ (Q.T @ batch_dummies.to_numpy())
    fit = sm.OLS(y, sm.add_constant(batch_ortho)).fit()
    y_adj = fit.resid + y.mean()

    out[adjusted_col] = np.tanh(y_adj) if fisher_transform else y_adj
    return out
