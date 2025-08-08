#!/usr/bin/env python3
"""31-insert_school.py"""


def insert_school(mongo_collection, **kwargs):
    """insertion d'un nouveau document"""
    return mongo_collection.insert_one(kwargs).inserted_id
