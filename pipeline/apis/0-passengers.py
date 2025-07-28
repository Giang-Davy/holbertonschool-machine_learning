#!/usr/bin/env python3
"""0-passengers.py"""


import requests


def availableShips(passengerCount):
    """
    Connaitre le vaisseau par le nombre
    de passagers qu'il peut contenir
    """

    url = "https://swapi-api.hbtn.io/api/starships/"
    liste_vaisseaux = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            print("Erreur lors de la requête:", response.status_code)
            break

        data = response.json()
        results = data['results']
        for i in results:
            if i['passengers'] in ['n/a', 'unknown']:
                continue
            nombre_sans_virgules = i['passengers'].replace(",", "")
            nombre = int(nombre_sans_virgules)
            if nombre >= passengerCount:
                liste_vaisseaux.append(i['name'])

        url = data.get('next')

    return liste_vaisseaux
