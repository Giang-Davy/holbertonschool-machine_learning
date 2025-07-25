#!/usr/bin/env python3
"""13-analyze.py"""


def analyze(df):
    """analiser"""
    df = df.drop(columns=["Timestamp"])
    df = df.describe()
    return df
