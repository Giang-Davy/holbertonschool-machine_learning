#!/usr/bin/env python3
"""3-first_launch.py"""


import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    data = response.json()

    launch = sorted(data, key=lambda x: x['date_unix'])[0]

    rocket_id = launch['rocket']
    launchpad_id = launch['launchpad']

    rocket = requests.get(
        f'https://api.spacexdata.com/v4/rockets/{rocket_id}').json()
    launchpad = requests.get(
        f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}').json()

    name = launch['name']
    date = launch['date_local']
    rocket = rocket['name']
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']

    print(
        f"{name} ({date}) {rocket} - {launchpad_name} ({launchpad_locality})"
        )
