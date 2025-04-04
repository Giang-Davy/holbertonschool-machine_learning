#!/usr/bin/env python3
"""12-agglomerative"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """sklearn agglomerative"""
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method="ward")
    dendrogram = scipy.cluster.hierarchy.dendrogram(
            linkage_matrix, color_threshold=dist)
    plt.show()

    return scipy.cluster.hierarchy.fcluster(Z=linkage_matrix,
                                            t=dist, criterion="distance")
