#!/usr/bin/env python3
"""fonction"""


import numpy as np


def moving_average(data, beta):
    """
    Calcule la moyenne mobile exponentielle avec correction du biais
    """
    v_prev = 0  # Initial� 0 (ou à la première valeur de la série)
    moving_averages = []  # Liste pour stocker les moyennes mobiles

    for t, x_t in enumerate(data, 1):  # Commence à rection du biais
        v_t = beta * v_prev + (1 - beta) * x_t  # Calcmobile
        v_t_corrected = v_t / (1 - beta**t)  # Correction du biais
        moving_averages.append(v_t_corrected)  # Ajoutrrigée à la liste
        v_prev = v_t  # Mettre à jour la valeur pré itération

    return moving_averages
