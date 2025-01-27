#!/usr/bin/env python3
"""fonction"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion à partir des labels et des logits.

    Args:
    - labels (numpy.ndarray): Tableau one-hot de formabels corrects.
    - lots (numpy.ndarray): Tableau one-hnant les labels prédits.

    Retourne :
    - confusion (numpy.ndarray): Matrice de confurme (classes, classes).
    """
    # Initialiser la matrice de confusion avec des zéros
    confusion = np.zeros((labels.shape[1], labels.shape[1]))

    # Parcourir tous les points de données
    for i in range(labels.shape[0]):
        # Trouver l'indice de la véritable étiquette et de l'étiquette prédite
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])

        # Incrémenter la cellule correspondante dans la matrice de confusion
        confusion[true_label][predicted_label] += 1

    return confusion
