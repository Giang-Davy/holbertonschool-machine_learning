#!/usr/bin/env python3
"""6-flip_switch.py"""


def flip_switch(df):
    """inverse le contenue"""
    df = df.iloc[::-1, ::-1]
    df_transpose = df.T
    return df_transpose
