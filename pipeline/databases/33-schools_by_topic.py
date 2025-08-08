#!/usr/bin/env python3
"""33-schools_by_topic.py"""


def schools_by_topic(mongo_collection, topic):
    """retourne une liste avec un sujet prÉcis"""
    return list(mongo_collection.find({"topics": topic}))
