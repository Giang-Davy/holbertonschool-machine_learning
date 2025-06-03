#!/usr/bin/env python3

import tensorflow_datasets as tfds
import transformers


class Dataset:
    def __init__(self):
        self.data_train, self.data_valid = tfds.load(
        'ted_hrlr_translate/pt_to_en',
        split=['train', 'validation'],
        as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        def tokenize_dataset(self, data):
            tokenizer_pt = transformers.BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
            tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
            return tokenizer_pt, tokenizer_en
