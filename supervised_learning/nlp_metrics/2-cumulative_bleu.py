#!/usr/bin/env python3
"""2-cumulative_bleu.py"""


import math


def ngram_precision(references, sentence, n):
    """Calcule la précision n-gram sans BP"""
    ngrams = [tuple(sentence[i:i + n]) for i in range(
        len(sentence) - n + 1)]
    refs_ngrams = [
        [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
        for ref in references
    ]
    clipped_count = 0
    total_count = len(ngrams)
    unique_ngrams = set(ngrams)
    for ng in unique_ngrams:
        count_sentence = ngrams.count(ng)
        max_ref_count = max(
            ref_ngrams.count(ng) for ref_ngrams in refs_ngrams)
        clipped_count += min(count_sentence, max_ref_count)
    if total_count == 0:
        return 0
    return clipped_count / total_count


def cumulative_bleu(references, sentence, n):
    """cumulative calcul"""
    P = []
    for k in range(1, n + 1):
        P.append(ngram_precision(references, sentence, k))
    if any(p == 0 for p in P):
        score = 0
    else:
        score = math.exp((1 / n) * sum(math.log(p) for p in P))
    ref_lens = [len(ref) for ref in references]
    cand_len = len(sentence)
    r = min(ref_lens, key=lambda ref_len: (abs(
        ref_len - cand_len), ref_len))
    if cand_len > r:
        BP = 1
    else:
        BP = math.exp(1 - (r / cand_len))
    BLEU = BP * score
    return BLEU
