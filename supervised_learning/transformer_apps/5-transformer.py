#!/usr/bin/env python3
"""5-transformer.py"""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """encodement"""
    pos = np.arange(max_seq_len)
    dm_matrice = np.arange(0, dm, 2)
    pos = np.expand_dims(pos, axis=1)
    PE_1 = np.sin(pos/(10000**(dm_matrice/dm)))
    PE_2 = np.cos(pos/(10000**(dm_matrice/dm)))
    result = np.concatenate([PE_1, PE_2], axis=1)
    result[:, 0::2] = PE_1
    result[:, 1::2] = PE_2
    return result

def sdp_attention(Q, K, V, mask=None):
    """sdp attention calcule"""
    # Q, K, V: (..., seq_len_q, depth)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """classe"""
    def __init__(self, dm, h):
        """initialisation"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.depth = dm // h
        self.dm = dm
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """rappelle"""
        batch_size = tf.shape(Q)[0]
        q_seq_len = tf.shape(Q)[1]  # Ajouté

        # Projeter Q, K, V
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Reshape et transpose pour multi-head
        def split_heads(x):
            shape = tf.shape(x)
            batch_size = shape[0]
            seq_len = shape[1]
            x = tf.reshape(x, (batch_size, seq_len, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention
        output, weights = sdp_attention(q, k, v, mask)

        # Concaténer les têtes
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, q_seq_len, self.dm))  # Correction ici
        output = self.linear(output)

        return output, weights
    
class EncoderBlock(tf.keras.layers.Layer):
    """block encoder"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialisation"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """rappelle"""
        Q = x
        K = x
        V = x
        output, weights = self.mha(Q, K, V, mask)
        result = self.dropout1(output, training)
        add1 = result + x
        layernorm = self.layernorm1(add1)
        hidd1 = self.dense_hidden(layernorm)
        dense1 = self.dense_output(hidd1)
        result2 = self.dropout2(dense1, training)
        add2 = result2 + layernorm
        layernorm2 = self.layernorm2(add2)

        return layernorm2
    
class DecoderBlock(tf.keras.layers.Layer):
    """Classe decoderblock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialisation"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """rappelle"""
        Q = x
        K = x
        V = x
        output1, weights = self.mha1(Q, K, V, look_ahead_mask)
        result = self.dropout1(output1, training)
        add1 = result + x
        layernorm1 = self.layernorm1(add1)
        Q = layernorm1
        K = encoder_output
        V = encoder_output
        output2, weights = self.mha2(Q, K, V, padding_mask)
        result2 = self.dropout2(output2, training)
        add2 = layernorm1 + result2
        layernorm2 = self.layernorm2(add2)
        hidd = self.dense_hidden(layernorm2)
        final_dense = self.dense_output(hidd)
        result3 = self.dropout3(final_dense, training)
        add3 = layernorm2 + result3
        layernorm3 = self.layernorm3(add3)

        return layernorm3

class Encoder(tf.keras.layers.Layer):
    """Classe decoderblock"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialisation"""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(
            max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """rappelle"""
        lenght = tf.shape(x)[1]
        embed = self.embedding(x)
        embed = embed * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        encoding = self.positional_encoding[:lenght, :]
        encoding = tf.cast(encoding, tf.float32)
        encoding = tf.expand_dims(encoding, axis=0)
        somme = embed + encoding
        embed = self.dropout(somme, training)
        for bloc in self.blocks:
            embed = bloc(embed, training, mask)
        return embed
    
class Decoder(tf.keras.layers.Layer):
    """Decoder"""
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialisation"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(
            max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """rappelle"""
        length = tf.shape(x)[1]
        embed = self.embedding(x)
        embed = embed * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        encoding = self.positional_encoding[:length, :]
        encoding = tf.cast(encoding, tf.float32)
        encoding = tf.expand_dims(encoding, axis=0)
        somme = embed + encoding
        embed = self.dropout(somme, training)
        for bloc in self.blocks:
            embed = bloc(embed, encoder_output, training,
                         look_ahead_mask, padding_mask)
        return embed

class Transformer(tf.keras.Model):
    """transformer"""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """initialisation"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """rappelle"""
        encode = self.encoder(inputs, training, encoder_mask)
        decod = self.decoder(target, encode, training,
                             look_ahead_mask, decoder_mask)
        linear = self.linear(decod)

        return linear
