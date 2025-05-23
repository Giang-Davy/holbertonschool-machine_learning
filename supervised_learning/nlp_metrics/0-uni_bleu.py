#!/usr/bin/env python3
"""0-uni_bleu.py"""


import math


def uni_bleu(references, sentence):
    """calculer les Unnigram Bleu"""
    clipped_count = 0

    total_count = len(sentence)
    for word in set(sentence):
        count = sentence.count(word)
        max_ref = 0
        for ref in references:
            freq = ref.count(word)
            if freq > max_ref:
                max_ref = freq
        clipped_count += min(count, max_ref)
    precision = clipped_count / total_count
    min_diff = float('inf')
    best_len = 0
    for ref in references:
        ref_len = len(ref)
        diff = abs(ref_len - len(sentence))
        if diff < min_diff:
            min_diff = diff
            best_len = ref_len
        elif diff == min_diff and ref_len < best_len:
            best_len = ref_len
    if len(sentence) >= best_len:
        BP = 1
    else:
        BP = math.exp(1 - (best_len/len(sentence)))
    bleu = BP * precision
    return bleu
