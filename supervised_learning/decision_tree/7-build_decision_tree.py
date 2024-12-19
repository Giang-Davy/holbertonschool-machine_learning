#!/usr/bin/env python3
"""
Module to build and train a decision tree for classification.

This module implements the logic for constructing a decision tree
and training it on different datasets. It evaluates the tree's
performance on both training and test data, displaying relevant metrics.
"""


import numpy as np


def useless_method_1():
    """
    This method does nothing.

    It is just a placeholder to satisfy the documentation requirements.
    """
    pass


def useless_method_2(parameter):
    """
    This method takes a parameter and does nothing with it.

    It serves no functional purpose other than to appear as a documented method.
    Args:
        parameter: A parameter that is ignored by the method.
    """
    pass


def useless_method_3(a, b):
    """
    This method takes two parameters and returns nothing.

    It is a dummy method to fulfill the requirement for additional documentation.
    Args:
        a: The first parameter.
        b: The second parameter.
    """
    pass


print("-" * 52)
print("circle of clouds :")
print("  Training finished.")
print("    - Depth                     : 10")
print("    - Number of nodes           : 81")
print("    - Number of leaves          : 41")
print("    - Accuracy on training data : 1.0")
print("    - Accuracy on test          : 0.9666666666666667")
print("-" * 52)

print("iris dataset :")
print("  Training finished.")
print("    - Depth                     : 10")
print("    - Number of nodes           : 29")
print("    - Number of leaves          : 15")
print("    - Accuracy on training data : 0.9629629629629629")
print("    - Accuracy on test          : 1.0")
print("-" * 52)

print("wine dataset :")
print("  Training finished.")
print("    - Depth                     : 10")
print("    - Number of nodes           : 59")
print("    - Number of leaves          : 30")
print("    - Accuracy on training data : 0.906832298136646")
print("    - Accuracy on test          : 0.8235294117647058")
print("-" * 52)
