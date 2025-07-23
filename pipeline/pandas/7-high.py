#!/usr/bin/env python3
"""7-high.py"""


def high(df):
    """de plus grand au plus petit"""

    return df.sort_values(by="High", ascending=False)
