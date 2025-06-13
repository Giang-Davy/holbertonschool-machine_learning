#!/usr/bin/env python3
"""4-qa.py"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """question réponses avec plusieurs référence"""
    tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
    # On charge le modèle BERT de question/réponse depuis TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    while True:
        word = input("Q: ")
        word_lower = word.lower()
        verif = ['bye', 'exit', 'quit', 'goodbye']
        if word_lower in verif:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, word)

        # On tokenize la question et le texte de référence
        inputs = tokenizer(word, reference,
                           return_tensors="tf", truncation=True,
                           max_length=512)

        # On prépare les tenseurs d'entrée pour le modèle
        input_tensors = [
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"]
        ]

        # On passe les tenseurs au modèle et on récupère les logits de début
        # et de fin de réponse
        output = model(input_tensors)
        start_logits = output[0]
        end_logits = output[1]

        sequence_length = inputs["input_ids"].shape[1]

        # On cherche les indices de début
        # et de fin les plus probables pour la réponse
        start_index = tf.math.argmax(
            start_logits[0, 1:sequence_length - 1]) + 1
        end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

        # Si l'indice de fin est avant celui de début, on retourne None
        if end_index < start_index:
            print("Sorry, I do not understand your question.")
            continue
        # On récupère les tokens correspondant à la réponse
        answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]
        # On décode les tokens pour obtenir la réponse finale en texte
        answer = tokenizer.decode(answer_tokens,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)

        if not answer.strip():
            print("Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
