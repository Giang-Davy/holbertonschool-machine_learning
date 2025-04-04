#!/usr/bin/env python3
"""10-kmeans"""


from sklearn.cluster import KMeans


def kmeans(X, k):
    """kmeans en sklearn"""
    # Appliquer KMeans de sklearn
    model = KMeans(n_clusters=k)
    model.fit(X)

    # C : centroïdes
    C = model.cluster_centers_

    # clss : indices des clusters
    clss = model.labels_

    return C, clss
