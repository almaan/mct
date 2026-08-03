import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot_gene_set(gene_set, embed, thresh=0.4, seed=10, edge_font_size=15):
    # Compute cosine similarities among genes in this gene program
    df_gs = pd.DataFrame(columns=['Gene', 'Similarity', 'Gene1'])
    for i in tqdm(gene_set):
        df = embed.compute_similarities(i, gene_set)
        df['Gene1'] = i
        df_gs = pd.concat([df_gs, df])
    df_gs = df_gs[df_gs['Similarity'] < 0.99].sort_values(by='Gene')  # Filter out edges from each gene to itself

    # Creates a graph from the cosine similarity network
    input_node_weights = [(row['Gene'], row['Gene1'], round(row['Similarity'], 2)) for i, row in df_gs.iterrows()]
    G = nx.Graph()
    G.add_weighted_edges_from(input_node_weights)

    # Plot the cosine similarity network; strong edges (> select threshold) are highlighted
    thresh = thresh
    plt.figure(figsize=(20, 20))
    widths = nx.get_edge_attributes(G, 'weight')

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > thresh]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= thresh]

    pos = nx.spring_layout(G, k=0.4, iterations=15, seed=seed)

    width_large = {}
    width_small = {}
    for i, v in enumerate(list(widths.values())):
        if v > thresh:
            width_large[list(widths.keys())[i]] = v * 10
        else:
            width_small[list(widths.keys())[i]] = max(v, 0) * 10

    nx.draw_networkx_edges(G, pos,
                           edgelist=width_small.keys(),
                           width=list(width_small.values()),
                           edge_color='lightblue',
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos,
                           edgelist=width_large.keys(),
                           width=list(width_large.values()),
                           alpha=0.5,
                           edge_color="blue",
                           )
    # node labels
    nx.draw_networkx_labels(G, pos, font_size=30, font_family="sans-serif")
    # edge weight labels
    d = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: d[k] for k in elarge}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=edge_font_size)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.show()
