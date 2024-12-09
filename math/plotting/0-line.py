#!/usr/bin/env python3
"""ff"""


import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Args: ff

    Returns: ff
    """
    # Définir les valeurs de y (cube des nombres de 0 à 10)
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)  # Créer les valeurs de x de 0 à 10

    plt.figure(figsize=(6.4, 4.8))  # Taille de la figure

    # Tracer y comme une ligne rouge continue
    plt.plot(x, y, color='r', linestyle='-')

    # Définir la plage de l'axe des x de 0 à 10
    plt.xlim(0, 10)

    # Afficher le graphique
    plt.show()
