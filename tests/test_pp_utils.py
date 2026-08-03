import pandas as pd
import pytest

from cell_cell_interactions.pp.utils import get_cell_type_order, group_cell_types_by_sample


def test_get_cell_type_order_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one cell type"):
        get_cell_type_order({}, n_cells_bag=4)


def test_group_cell_types_by_sample_preserves_requested_columns_and_input_df():
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s1", "s2"],
            "cell_type": ["T", "B", "other", "B"],
        },
        index=[10, 11, 12, 13],
    )

    grouped = group_cell_types_by_sample(obs, "sample", "cell_type", ["B", "T"])

    assert "idx" not in obs.columns
    assert grouped.columns.tolist() == ["B", "T"]
    assert grouped.loc["s1", "B"] == [11]
    assert grouped.loc["s1", "T"] == [10]
    assert grouped.loc["s2", "B"] == [13]
    assert grouped.loc["s2", "T"] == []
