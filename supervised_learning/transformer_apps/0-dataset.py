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
        pt_sentences = []
        en_sentences = []

        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        pt_tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        en_tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        pt_tokenizer.train_new_from_iterator(pt_sentences, vocab_size=2**13)
        en_tokenizer.train_new_from_iterator(en_sentences, vocab_size=2**13)

        return pt_tokenizer, en_tokenizer
