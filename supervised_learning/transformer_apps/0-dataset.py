#!/usr/bin/env python3
"""0-dataset.py"""


import tensorflow_datasets as tfds
import transformers


class Dataset:
    """classe de dataset"""
    def __init__(self):
        """initialisation"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """crée deux tokenizers avec vocabulaire 2**13"""
        pt_texts = []
        en_texts = []

        for pt, en in data:
            pt_texts.append(pt.numpy().decode("utf-8"))
            en_texts.append(en.numpy().decode("utf-8"))

        tokenizer_pt = transformers.BertTokenizerFast.train_new_from_iterator(
            pt_texts,
            vocab_size=2**13
        )
        tokenizer_en = transformers.BertTokenizerFast.train_new_from_iterator(
            en_texts,
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
