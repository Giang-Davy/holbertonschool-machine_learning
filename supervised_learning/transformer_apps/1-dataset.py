#!/usr/bin/env python3
"""1-dataset.py"""


import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """classe"""
    def __init__(self):
        """initialisation"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """token"""
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

        pt_tokenizer = pt_tokenizer.train_new_from_iterator(
            pt_sentences, vocab_size=2**13)
        en_tokenizer = en_tokenizer.train_new_from_iterator(
            en_sentences, vocab_size=2**13)

        return pt_tokenizer, en_tokenizer

    def encode(self, pt, en):
        """encode"""
        if isinstance(pt, tf.Tensor):
            pt = pt.numpy().decode('utf-8')
        if isinstance(en, tf.Tensor):
            en = en.numpy().decode('utf-8')

        vocab_size_pt = len(self.tokenizer_pt.get_vocab())
        tokens_pt = self.tokenizer_pt.encode(pt, add_special_tokens=False)
        tokens_pt = [vocab_size_pt] + tokens_pt + [vocab_size_pt + 1]

        vocab_size_en = len(self.tokenizer_en.get_vocab())
        tokens_en = self.tokenizer_en.encode(en, add_special_tokens=False)
        tokens_en = [vocab_size_en] + tokens_en + [vocab_size_en + 1]

        return tokens_pt, tokens_en
