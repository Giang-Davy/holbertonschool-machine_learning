#!/usr/bin/env python3
"""10-index.py"""


def index(df):
    """mettre timestamp sur le colonne de la dataframe"""
    df = df.set_index("Timestamp")
    return df
