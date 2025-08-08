#!/usr/bin/env python3
"""30-all.py"""

def list_all(mongo_collection):
    """liste tout les documents"""
	return list(mongo_collection.find())
