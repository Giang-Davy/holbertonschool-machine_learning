#!/usr/bin/env python3
"""1-sentience.py"""


import requests


def sentientPlanets():
    """
    Afficher toutes les espèces "sentient" et leurs planètes d'origine
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    planetes = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            print("Erreur lors de la requête:", response.status_code)
            break

        data = response.json()
        results = data['results']
        for i in results:
            if (i['designation'].lower() == 'sentient' or i['classification'].lower() == 'sentient'):
                if i["homeworld"] is None:
                    continue
                else:
                    url_home = i["homeworld"]
                    response_home = requests.get(url_home)
                    data_home = response_home.json()
                    planetes.append(data_home['name'])
        url = data.get('next')

    return planetes
