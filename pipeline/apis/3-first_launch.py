#!/usr/bin/env python3
"""3-first_launch.py"""


import requests


def get_first_launch_info():
    """Avoir les informations du premier lancement"""
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()
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
    rocket_name = rocket['name']
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']

    return (f"{name} ({date}) {rocket_name} - "
            f"{launchpad_name} ({launchpad_locality})")


if __name__ == '__main__':
    print(get_first_launch_info())
