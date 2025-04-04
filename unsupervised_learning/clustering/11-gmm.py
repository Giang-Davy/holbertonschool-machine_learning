#!/usr/bin/env python3
"""11-gmm.py"""


import sklearn.mixture


def gmm(X, k):
    """gmm en sklearn"""
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
