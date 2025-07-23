#!/usr/bin/env python3
"""4-array.py"""


def array(df):
    """prendre les 10 dernières valeurs de High et Close
    et le convertir en np"""
    valeur_select = df[["High", "Close"]].tail(10)
    return valeur_select.values
