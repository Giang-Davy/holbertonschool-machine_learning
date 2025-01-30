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
    caches = {"A0": X}  # Initialisation de la première couche avec les entrées
    for couche_actuelle in range(1, L + 1):
        # Récupérer les poids et biais pour la couche actuelle
        W = weights[f"W{couche_actuelle}"]
        b = weights[f"b{couche_actuelle}"]
        
        # Calcul de la sortie avant activation
        A_prev = caches[f"A{couche_actuelle - 1}"]  # Sortie de la couche précédente
        output_before = np.dot(W, A_prev) + b
        
        # Application de l'activation (tanh pour toutes sauf la dernière couche)
        if couche_actuelle == L:
            # Softmax pour la dernière couche
            exp_output = np.exp(output_before - np.max(output_before, axis=0, keepdims=True))  # Stabilité numérique
            A = exp_output / np.sum(exp_output, axis=0, keepdims=True)
        else:
            A = np.tanh(output_before)  # tanh pour les autres couches
        
        # Dropout : appliquer si ce n'est pas la dernière couche
        if couche_actuelle != L:
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = A * D  # Appliquer le masque de dropout
            A = A / keep_prob  # Normalisation du dropout
        
        # Stocker la sortie activée dans le dictionnaire
        caches[f"A{couche_actuelle}"] = A
        caches[f"D{couche_actuelle}"] = D  # Stocker également le masque de dropout
        
    return caches
