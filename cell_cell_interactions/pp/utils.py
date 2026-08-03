import pandas as pd


def get_ct_indices(ct_list: list[str], indices: list[int], all_cts: list[str]) -> dict[str, list[int]]:
    """Get indices of cell types in a list of cell types in a single sample."""
    ct_indices = {ct: [] for ct in all_cts}
    for ct, idx in zip(ct_list, indices):
        if ct in ct_indices:
            ct_indices[ct].append(idx)
    return ct_indices


def get_cell_type_order(cell_type_proportions: dict[str, float], n_cells_bag: int) -> dict[str, tuple[int, int]]:
    """Get the order of cell types for the encoder input."""
    if not cell_type_proportions:
        raise ValueError("cell_type_proportions must contain at least one cell type.")

    start = 0
    cell_types = list(cell_type_proportions.keys())
    cell_type_input_order = {}
    for ct in cell_types[:-1]:
        ct_prop = cell_type_proportions[ct]
        end = start + round(ct_prop * n_cells_bag)
        cell_type_input_order[ct] = (start, end)
        start = end
    if start < n_cells_bag:
        cell_type_input_order[cell_types[-1]] = (start, n_cells_bag)
    return cell_type_input_order


def group_cell_types_by_sample(
    obs: pd.DataFrame, sample_key: str, cell_type_key: str, cell_types: list[str]
) -> pd.DataFrame:
    """Group cell types by sample."""
    if not cell_types:
        raise ValueError("cell_types must contain at least one cell type.")

    # add idx column
    obs = obs.copy()
    obs["idx"] = obs.index.tolist()
    # group cell types by sample
    cell_types_grouped_by_sample = obs.groupby(sample_key)[[cell_type_key, "idx"]].agg(list)

    # get indices of each cell to be able to later retrieve particular cells from the matrix
    cell_types_grouped_by_sample["ct_indices"] = cell_types_grouped_by_sample.apply(
        lambda row: get_ct_indices(row[cell_type_key], row["idx"], cell_types), axis=1
    )

    # explode the dictionary to columns
    cell_types_grouped_by_sample = pd.concat(
        [
            cell_types_grouped_by_sample.drop(["ct_indices"], axis=1),
            cell_types_grouped_by_sample["ct_indices"].apply(pd.Series).reindex(columns=cell_types),
        ],
        axis=1,
    ).drop(columns=[cell_type_key, "idx"])

    return cell_types_grouped_by_sample
