import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram

def compare_models(names: list, params_dict_ls: list):
    scores = np.zeros([6, 4])
    for i, d in enumerate(params_dict_ls):
        for j, n in enumerate(names):
            scores[i, j] = d[n][0]
        
    for j, n in enumerate(names):
        plt.scatter(range(1, 7), scores[:, j], label=n, s=80)
        plt.plot(range(1, 7), scores[:, j])
        plt.xticks(range(1, 7), range(6))

        plt.xlabel('# of used freq')
        plt.ylabel('Silhouette euclidian score')
        plt.legend()
    plt.show()
    return

def plot_dendrogram(model, **kwargs):
    '''Source:
       https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html'''
    
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return

def dendogram_well_clust(X: np.array, hyperparams: dict):
    hp = hyperparams.copy()
    hp.pop('n_clusters')
    # model = AgglomerativeClustering(**hp, n_clusters=None, distance_threshold=0)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    model.fit(X)
    
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    return