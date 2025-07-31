#!/usr/bin/env python3
"""4-rocket_frequency.py"""


import requests


def frequencerockets():
    """compte le nombre de lancement par rockets"""
count = {}
response = requests.get('https://api.spacexdata.com/v4/launches')
launches = response.json()
for i in launches:
    rocket_id = i['rocket']
    if rocket_id in count:
        count[rocket_id] += 1
    else:
        count[rocket_id] = 1

response = requests.get('https://api.spacexdata.com/v4/rockets')
rockets = response.json()
id_to_name = {}
for r in rockets:
    id_to_name[r['id']] = r['name']

result = []
for rocket_id, freq in count.items():
    name = id_to_name.get(rocket_id, "")
    result.append((name, freq))

result.sort(key=lambda x: (-x[1], x[0]))

for name, freq in result:
    print(f"{name}: {freq}")
