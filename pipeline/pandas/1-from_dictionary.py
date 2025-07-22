#!/usr/bin/env python3
"""1-from_dictionary.py"""


import pandas as pd


dico = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
    }

lettres = ['A', 'B', 'C', 'D']
df = pd.DataFrame(data=dico, index=lettres)
