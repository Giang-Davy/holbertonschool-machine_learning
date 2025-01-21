#!/usr/bin/env python3
"""fonction"""


import numpy as np


def moving_average(data, beta):
    """
    Calcule la moyenne mobile exponentielle avec correction du biais
    """
    v_prev = 0  # Initialiser la valeur précédente à 0 (ou à la première valeur de la série)
    moving_averages = []  # Liste pour stocker les moyennes mobiles
    
    for t, x_t in enumerate(data, 1):  # Commence à 1 pour faciliter la correction du biais
        v_t = beta * v_prev + (1 - beta) * x_t  # Calcul de la moyenne mobile
        v_t_corrected = v_t / (1 - beta**t)  # Correction du biais
        moving_averages.append(v_t_corrected)  # Ajouter la moyenne mobile corrigée à la liste
        v_prev = v_t  # Mettre à jour la valeur précédente pour la prochaine itération

    return moving_averages
