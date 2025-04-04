#!/usr/bin/env python3
"""10-kmeans"""


import sklearn.cluster


def kmeans(X, k):
    """kmeans en sklearn"""
    # Appliquer KMeans de sklearn
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    # C : centroïdes
    C = model.cluster_centers_

    # clss : indices des clusters
    clss = model.labels_

    return C, clss
