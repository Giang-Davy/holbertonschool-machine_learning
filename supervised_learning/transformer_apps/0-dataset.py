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
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_dataset(self, data):
        """Tokenize le dataset"""
        pt_texts = []
        en_texts = []
        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        tokenized_pt = self.tokenizer_pt(pt_texts, padding=True, truncation=True, return_attention_mask=False)["input_ids"]
        tokenized_en = self.tokenizer_en(en_texts, padding=True, truncation=True, return_attention_mask=False)["input_ids"]

        return tokenized_pt, tokenized_en
