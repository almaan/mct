"""Preprocessing utilities for AnnData objects passed to ``MCTModel.embed``.

These are convenience wrappers around the same normalization the model was
trained with. They are pure DataFrame / AnnData in-place operations; nothing
here talks to the model.
"""

from __future__ import annotations

from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

__all__ = [
    "filter_to_model_vocabulary",
    "harmonize_cell_types",
    "log_normalize",
]


def filter_to_model_vocabulary(
    adata: ad.AnnData,
    gene2idx: dict[str, int],
    copy: bool = False,
) -> ad.AnnData:
    """Subset ``adata`` to genes present in the model's ``gene2idx``.

    Parameters
    ----------
    adata:
        Input AnnData. Gene symbols are read from ``adata.var_names``.
    gene2idx:
        Mapping from gene symbol to token id (as produced by
        ``mct.gene2idx``).
    copy:
        Return a copy rather than an in-memory view.

    Returns
    -------
    anndata.AnnData

    Examples
    --------
    >>> import cell_cell_interactions as cci
    >>> mct = cci.load("/models/mct_v1")
    >>> adata = cci.pp.filter_to_model_vocabulary(adata, mct.gene2idx)
    >>> adata.n_vars
    11743
    """
    keep = [g for g in adata.var_names if g in gene2idx]
    view = adata[:, keep]
    return view.copy() if copy else view


def harmonize_cell_types(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    mapping: dict[str, str] | None = None,
    drop_unmapped: bool = False,
    new_key: str | None = None,
) -> ad.AnnData:
    """Apply a ``{source_cell_type: harmonized_cell_type}`` mapping.

    Parameters
    ----------
    adata:
        Input AnnData.
    cell_type_key:
        Column in ``adata.obs`` to rewrite.
    mapping:
        Dict of remappings. Missing keys pass through unchanged (unless
        ``drop_unmapped=True``).
    drop_unmapped:
        If True, drop cells whose original label is absent from ``mapping``.
    new_key:
        If given, write the harmonized labels into this new column instead
        of overwriting ``cell_type_key``.

    Returns
    -------
    anndata.AnnData

    Examples
    --------
    >>> mct = cci.load("/models/mct_v1")
    >>> adata = cci.pp.harmonize_cell_types(
    ...     adata,
    ...     cell_type_key="cell_type",
    ...     mapping=mct.index.cell_type_map,
    ... )
    """
    if mapping is None:
        return adata
    target_key = new_key if new_key is not None else cell_type_key

    original = adata.obs[cell_type_key].astype(str)
    if drop_unmapped:
        keep = original.isin(mapping.keys())
        adata = adata[keep].copy()
        original = original[keep]

    adata.obs[target_key] = original.map(lambda x: mapping.get(x, x))
    return adata


def log_normalize(
    df_or_adata,
    target_sum: float = 1e4,
    copy: bool = False,
):
    """Log-normalize a pseudobulk DataFrame or an AnnData object.

    The normalization used by MCT during training is a global
    ``log1p(x / total * target_sum)`` where ``total`` is the sum of all
    counts in the input (per-sample). This helper exposes the same
    operation so users can preview what the model sees.

    Accepts either:

    - a ``pandas.DataFrame`` (returned normalized) — this is what
      ``MCTModel.embed`` uses internally on pseudobulk;
    - an ``anndata.AnnData`` (writes normalized ``.X`` in place unless
      ``copy=True``).

    Examples
    --------
    >>> df = pd.DataFrame({"c1": [1, 2, 0], "c2": [5, 0, 3]}, index=["A", "B", "C"])
    >>> cci.pp.log_normalize(df).round(3)
             c1     c2
    A     0.693  2.301
    B     1.099  0.000
    C     0.000  1.386
    """
    if isinstance(df_or_adata, pd.DataFrame):
        df = df_or_adata
        total = df.values.sum()
        if total <= 0:
            return df * 0.0
        return np.log1p(df / total * target_sum)

    adata = df_or_adata.copy() if copy else df_or_adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    total = X.sum()
    if total > 0:
        adata.X = np.log1p(X / total * target_sum)
    return adata
