#!/usr/bin/env python3
"""2-qa.py"""


question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """question réponse avec chat"""
    while True:
        word = input("Q:")
        word_lower = word.lower()
        verif = ['bye', 'exit', 'quit', 'goodbye']
        if word_lower in verif:
            print("A: Goodbye")
            break
        if question_answer(word, reference) is None:
            print("Sorry, I do not understand your question.")
        else:
            print(f"A: {question_answer(word, reference)}")
