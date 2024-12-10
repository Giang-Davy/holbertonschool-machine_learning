#!/usr/bin/env python3
"""Module pour afficher les 4 graphiques en même temps"""

import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """
    Les 5 graphiques affichés
    """
    plt.figure(figsize=(6.4, 4.8))
    plt.suptitle('All in One', fontsize='x-large')

    # Graphique 0
    plt.subplot(3, 2, 1)
    y0 = np.arange(0, 11) ** 3
    x0 = np.arange(0, 11)
    plt.plot(x0, y0, color='r', linestyle='-')
    plt.xlim(0, 10)
    plt.yticks(np.arange(0, 1001, 500))

    # Graphique 1
    plt.subplot(3, 2, 2)
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    plt.scatter(x1, y1, color='magenta', s=1)
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')
    plt.title('Men\'s Height vs Weight', fontsize='x-small')

    # Graphique 2
    plt.subplot(3, 2, 3)
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    plt.plot(x2, y2)
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of C-14', fontsize='x-small')
    plt.yscale('log')
    plt.xlim(0, 28650)

    # Graphique 3
    plt.subplot(3, 2, 4)
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    plt.plot(x3, y31, color='red', linestyle='--', label='C-14')
    plt.plot(x3, y32, color='green', linestyle='-', label='Ra-226')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(fontsize='x-small')

    # Graphique 4 (occuper 2 colonnes)
    plt.subplot(3, 1, 3)
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.ylim(0, 30)
    plt.yticks(np.arange(0, 31, 10))
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))

    # Ajustement des espaces
    plt.tight_layout()
    plt.show()

all_in_one()
