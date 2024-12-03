#!/usr/bin/env python3
"""fonction"""


def add_arrays(arr1, arr2):
    """ff"""
    add = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            add.append(arr1[i] + arr2[i])
        return add
    else:
        return None
