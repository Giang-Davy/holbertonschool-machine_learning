#!/usr/bin/env python3
"""9-fill.py"""


def fill(df):
    """remplir"""
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].fillna(method='ffill')
    df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(df["Close"])
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(value=0)

    return df
