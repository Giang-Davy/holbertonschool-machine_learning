#!/usr/bin/env python3*
"""Fonction"""


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    tactac
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    names = ['Farrah', 'Fred', 'Felicia']
    plt.figure(figsize=(6.4, 4.8))
    plt.bar(names, fruit[0], color='red',
            width=0.5, label='apples')
    plt.bar(names, fruit[1], color='yellow',
            width=0.5, label='bananas', bottom=fruit[0])
    plt.bar(names, fruit[2], color='#ff8000',
            width=0.5, label='oranges', bottom=fruit[0] + fruit[1])
    plt.bar(names, fruit[3], color='#ffe5b4',
            width=0.5, label='peaches', bottom=fruit[0] + fruit[1] + fruit[2])
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()


bars()
