#!/usr/bin/env python3
"""0-qa.py"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Trouve un extrait de texte pour répondre à une question."""
    # On charge le tokenizer BERT pré-entraîné pour SQuAD
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    # On charge le modèle BERT de question/réponse depuis TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # On tokenize la question et le texte de référence
    inputs = tokenizer(question, reference,
                       return_tensors="tf", truncation=True, max_length=512)

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
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Si l'indice de fin est avant celui de début, on retourne None
    if end_index < start_index:
        return None

    # On récupère les tokens correspondant à la réponse
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # On décode les tokens pour obtenir la réponse finale en texte
    answer = tokenizer.decode(answer_tokens,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # Si la réponse est vide ou ne contient que des espaces, on retourne None
    if not answer.strip():
        return None

    return answer
