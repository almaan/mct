from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cell_cell_interactions.tl.condition_specific_genes import jaccard_similarity


def plot_similarity_across_conditions_heatmap(
    gene_df: pd.DataFrame,  # dataframe with neighbors
    k: int = 20,  # number of neighbors to consider
    min_items: int = 2,  # minimum number of samples per gene in a condition
    condition: str = 'tissue',  # the column name for the condition
    labels_fontsize: int = 26,  # font size for labels
):
    # remove items with very low frequency
    items = [i for i, count in gene_df[condition].value_counts().items() if count >= min_items]
    gene_df = gene_df[gene_df[condition].isin(items)].sort_values(by=condition)

    data = np.stack(gene_df.neighbors.tolist())[:, :k]
    data = jaccard_similarity(data)

    # Define group labels and their positions

    key_indices = defaultdict(list)
    for i, k in enumerate(gene_df[condition].tolist()):
        key_indices[k].append(i)

    group_labels = list(key_indices.keys())
    group_boundaries = [min(v) for v in key_indices.values()] + [len(data)]  # Boundaries between groups

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, cbar=True, cmap="rocket_r")

    # Set custom x and y ticks
    group_midpoints = [(group_boundaries[i] + group_boundaries[i + 1]) // 2 for i in range(len(group_boundaries) - 1)]
    ax.set_xticks(group_midpoints)
    ax.set_yticks(group_midpoints)

    # Access the colorbar and set the number of ticks
    cbar = ax.collections[0].colorbar  # Get the colorbar associated with the heatmap
    cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 4))  # Increase the number of ticks to 10
    cbar.set_ticklabels([f'{i:.2f}' for i in np.linspace(cbar.vmin, cbar.vmax, 4)])  # Set custom tick labels
    cbar.ax.tick_params(labelsize=20)  # Adjust the font size here

    # Remove default tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide all ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)

    # Function to draw brackets
    def draw_bracket(ax, x0, x1, y, bracket_width=0.5):
        # Draw horizontal brackets at the top
        ax.plot([x0, x0, x1, x1], [y, y + bracket_width, y + bracket_width, y], color='black', lw=2, clip_on=False)

    def draw_vertical_bracket(ax, y0, y1, x, bracket_width=0.5):
        # Draw vertical brackets on the left
        ax.plot([x, x - bracket_width, x - bracket_width, x], [y0, y0, y1, y1], color='black', lw=2, clip_on=False)

    # Draw brackets for x-axis at the top
    for start, end in zip(group_boundaries[:-1], group_boundaries[1:]):
        draw_bracket(ax, start, end, y=ax.get_ylim()[0], bracket_width=0.5)  # Top bracket

    # Draw brackets for y-axis on the left
    for start, end in zip(group_boundaries[:-1], group_boundaries[1:]):
        draw_vertical_bracket(ax, start, end, x=ax.get_xlim()[0], bracket_width=0.5)  # Left bracket

    # Add custom y-tick labels manually to the left
    for i, label in enumerate(group_labels):
        plt.text(-6, group_midpoints[i], label, ha='center', va='center', fontsize=labels_fontsize, rotation=90)

    # Add custom x-tick labels manually to the bottom
    for i, label in enumerate(group_labels):
        plt.text(group_midpoints[i], ax.get_ylim()[0] + 6, label, ha='center', va='center', fontsize=labels_fontsize)

    plt.show()
