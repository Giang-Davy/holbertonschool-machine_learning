#!/usr/bin/env python3
"""34-log_stats.py"""


from pymongo import MongoClient


def log_stats():
    """donne des stats"""
	client = MongoClient()
	collection = client.logs.nginx

	count = collection.count_documents({})
	print(f"{count} logs")

	print("Methods:")
	for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
		n = collection.count_documents({"method": method})
		print(f"\tmethod {method}: {n}")

	status_check = collection.count_documents(
            {"method": "GET", "path": "/status"})
	print(f"{status_check} status check")


if __name__ == "__main__":
	log_stats()
