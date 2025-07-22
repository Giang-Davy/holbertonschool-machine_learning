#!/usr/bin/env python3
"""3-rename.py"""


import pandas as pd


def rename(df):
    """changer de nom"""

    df = pd.DataFrame(df[['Timestamp', 'Close']])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.rename(columns={'Timestamp': 'Datetime'})

    return df
