#!/usr/bin/env python3
"""8-prune.py"""


def prune(df):
    """enlève NaN de Close"""

    filtered = df[df['Close'].notna()]
    return filtered
