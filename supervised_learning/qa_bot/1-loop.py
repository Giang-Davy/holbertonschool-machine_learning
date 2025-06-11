#!/usr/bin/env python3
"""1-loop.py"""

"""
crée un code pour un chat, si on écrit "bye", "exit",
"quit" ou "goodbye" cela fait quitter le chat
"""
while True:
    word = input("Q:")
    word_lower = word.lower()
    verif = ['bye', 'exit', 'quit', 'goodbye']
    if word_lower in verif:
        print("A: Goodbye")
        break
    else:
        print("A:")
