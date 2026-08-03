from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def jaccard_similarity(matrix: np.ndarray):
    """Calculate Jaccard similarity between rows of a matrix."""
    # Get the number of rows
    N = matrix.shape[0]

    # Initialize the N x N Jaccard similarity matrix
    jaccard_matrix = np.zeros((N, N))

    # Calculate Jaccard similarity for each pair of rows
    for i in range(N):
        for j in range(N):
            set_i = set(matrix[i])
            set_j = set(matrix[j])
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            if union == 0:
                jaccard_matrix[i, j] = 0
            elif i == j:
                jaccard_matrix[i, j] = 0
            else:
                jaccard_matrix[i, j] = intersection / union

    return jaccard_matrix


def jaccard_similarity_across_items(item: List[np.ndarray], k: int = 50):
    """Calculate the average Jaccard similarity"""
    item = np.stack(item)[:, :k]
    return np.mean(jaccard_similarity(item))


def get_most_common_neighbors(item: List[np.ndarray], k: int = 50):
    """Get the most common neighbors across items."""
    # Flatten all arrays into a single list
    all_items = np.concatenate(item)

    # Use Counter to find the frequency of each item
    item_counts = Counter(all_items)

    # Find the most common items
    most_common_indices = [idx for idx, c in item_counts.most_common()[:k]]
    return np.array(most_common_indices)


def calculate_condition_specificity(
    df: pd.DataFrame,  # dataframe with neighbors
    condition: str = "tissue",  # the column name for the condition
    min_n_conditions: int = 2,  # minimum number of samples per gene in a condition
    min_n_samples: int = 10,  # minimum number of samples per gene
    k_neighbors_within_condition: int = 20,  # number of neighbors to consider within a condition
    k_neighbors_across_conditions: int = 40,  # number of neighbors to consider across conditions
) -> pd.DataFrame:
    """Calculate condition-specificity by calculating the overlap across and within conditions."""
    # exclude genes which are in less than 1 condition
    genes = [gene for gene, count in df.groupby('gene_name')[condition].count().items() if count >= min_n_conditions]
    df = df[df.gene_name.isin(genes)]

    gdf = df.groupby(['gene_name', condition])['neighbors'].agg(list).to_frame()
    gdf['n_samples'] = gdf['neighbors'].apply(len)
    gdf = gdf[gdf['n_samples'] >= min_n_samples]

    gdf['overlap_within_condition'] = gdf['neighbors'].progress_apply(
        lambda x: jaccard_similarity_across_items(x, k=k_neighbors_within_condition))

    # filter out ones with no or max overlap
    gdf = gdf[gdf['overlap_within_condition'] > 0.]
    gdf = gdf[gdf['overlap_within_condition'] <= 1.]

    overlap_within_condition_df = gdf.reset_index().groupby(
        'gene_name')['overlap_within_condition'].mean().to_frame().reset_index()

    gdf['most_common_neighbors'] = gdf['neighbors'].progress_apply(
        lambda x: get_most_common_neighbors(x, k=k_neighbors_across_conditions))
    gdf = gdf.reset_index().groupby('gene_name')['most_common_neighbors'].agg(list).to_frame()

    gdf['overlap_across_conditions'] = gdf['most_common_neighbors'].progress_apply(
        lambda x: jaccard_similarity_across_items(x, k=k_neighbors_across_conditions))
    gdf = gdf[['overlap_across_conditions']].reset_index()

    output = pd.merge(overlap_within_condition_df, gdf, on='gene_name')
    # filter out ones with no or maximum overlap (noise)
    output = output[output['overlap_across_conditions'] > 0.]
    output = output[output['overlap_across_conditions'] <= 1.]

    output['condition_specificity_score'] = output['overlap_within_condition'] / output['overlap_across_conditions']
    output = output.sort_values(by='condition_specificity_score', ascending=False)
    return output
