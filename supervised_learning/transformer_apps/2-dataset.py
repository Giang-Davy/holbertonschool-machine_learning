#!/usr/bin/env python3
"""2-dataset.py"""


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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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
        #  transforme le contenu de pt/en byte puis en chaine de caratères
        if isinstance(pt, tf.Tensor):
            pt = pt.numpy().decode('utf-8')
        if isinstance(en, tf.Tensor):
            en = en.numpy().decode('utf-8')

        vocab_size_pt = len(self.tokenizer_pt.get_vocab())
        #  transformer en valeurs ID
        tokens_pt = self.tokenizer_pt.encode(pt, add_special_tokens=False)
        tokens_pt = [vocab_size_pt] + tokens_pt + [vocab_size_pt + 1]

        vocab_size_en = len(self.tokenizer_en.get_vocab())
        tokens_en = self.tokenizer_en.encode(en, add_special_tokens=False)
        tokens_en = [vocab_size_en] + tokens_en + [vocab_size_en + 1]

        return tokens_pt, tokens_en

    def tf_encode(self, pt, en):
        """tf_encode"""

        result = tf.py_function(
            func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        #  None pour que la séquence soit variable
        #  pour que tensorflow puisse faire un graphe dynamique
        result[0].set_shape([None])
        result[1].set_shape([None])

        return result
