#!/usr/bin/env python3
"""32-Update_topics.py"""

def update_topics(mongo_collection, name, topics):
    """change les sujets"""
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
