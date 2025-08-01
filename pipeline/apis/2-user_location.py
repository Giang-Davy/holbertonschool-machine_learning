#!/usr/bin/env python3
"""2-user_location.py"""


import sys
import requests
import time
import math


def userlocation(url):
    """avoir la location d'un user en fonction du lien github"""
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data['location']
        return results
    if response.status_code == 403:
        reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
        reset_in = (reset_timestamp - time.time()) / 60
        print(f"Reset in {math.ceil(reset_in)} min")
        return
    if response.status_code == 404:
        return None


if __name__ == '__main__':
    location = userlocation(sys.argv[1])
    if location is None and len(sys.argv) > 1:
        response = requests.get(sys.argv[1])
        if response.status_code != 403:
            print("Not found")
    elif location is not None:
        print(location)
