#!/usr/bin/env python3
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Propagation avant avec Dropout.
    
    X : numpy.ndarray, forme (nx, m) — données d'entrée
    weights : dictionnaire contenant les poids et biais
    L : int — nombre de couches
    keep_prob : probabilité de conserver un neurone
    """
    caches = {"A0": X}
    for couche_actuelle in range(1, L + 1):
        W = weights[f"W{couche_actuelle}"]
        b = weights[f"b{couche_actuelle}"]
        A_prev = caches[f"A{couche_actuelle - 1}"]
        output_before = np.dot(W, A_prev) + b
        
        if couche_actuelle == L:
            exp_output = np.exp(output_before - np.max(output_before, axis=0, keepdims=True))
            A = exp_output / np.sum(exp_output, axis=0, keepdims=True)
        else:
            A = np.tanh(output_before)
            D = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(float)  # Conversion en float (0. ou 1.)
            A = A * D
            A = A / keep_prob
        
        caches[f"A{couche_actuelle}"] = A
        caches[f"D{couche_actuelle}"] = D
        
    return caches
