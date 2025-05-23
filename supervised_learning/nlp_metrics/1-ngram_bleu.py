#!/usr/bin/env python3
"""1-ngram_bleu.py"""


import math


def ngram_bleu(references, sentence, n):
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(tuple(sentence[i:i + n]))

    refs_ngrams = []
    for ref in references:
        ref_ngrams = []
        for i in range(len(ref) - n + 1):
            ref_ngrams.append(tuple(ref[i:i + n]))
        refs_ngrams.append(ref_ngrams)

    clipped_count = 0
    total_count = len(ngrams)

    unique_ngrams = set(ngrams)

    for ng in unique_ngrams:
        count_sentence = ngrams.count(ng)

        max_ref_count = 0
        for ref_ngrams in refs_ngrams:
            count_ref = ref_ngrams.count(ng)
            if count_ref > max_ref_count:
                max_ref_count = count_ref

        clipped_count += min(count_sentence, max_ref_count)

    if total_count == 0:
        return 0

    precision = clipped_count / total_count

    return precision
