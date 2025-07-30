#!/usr/bin/env python3
"""3-first_launch.py"""


import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()
    launches = response.json()

    # Sort by date_unix, stable sort keeps API order for ties
    first_launch = sorted(launches, key=lambda x: x['date_unix'])[0]

    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    rocket = requests.get(
        f'https://api.spacexdata.com/v4/rockets/{rocket_id}').json()
    launchpad = requests.get(
        f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}').json()

    print(
        f"{first_launch['name']} ({first_launch['date_local']}) "
        f"{rocket['name']} - {launchpad['name']} ({launchpad['locality']})"
        )
