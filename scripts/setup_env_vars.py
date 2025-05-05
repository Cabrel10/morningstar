#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour configurer les variables d'environnement nécessaires au fonctionnement du pipeline.
Ce script définit les clés API dans l'environnement sans les stocker dans un fichier.
"""

import os
import sys

# Définir les clés API
os.environ['GOOGLE_API_KEY_1'] = 'AIzaSyBxT1xuZf_3CsnEnQs__HZJX-S_Ak6NS0E'
os.environ['GOOGLE_API_KEY_2'] = 'AIzaSyAnWLOQ1NemZEi1YMOIk2NIjVyOgJ66XfQ'
os.environ['COINMARKETCAP_API_KEY'] = '2b7f5480-8df2-4c84-a22a-8d9f66d4d580'
os.environ['COINMARKETCAP_API_KEY_2'] = '0853fffa-27e7-45c7-b549-5e857416275f'

# Définir GEMINI_API_KEY comme étant égal à GOOGLE_API_KEY_1 pour la compatibilité
os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY_1']

print("Variables d'environnement configurées avec succès.")

# Exécuter la commande passée en argument si spécifiée
if len(sys.argv) > 1:
    command = ' '.join(sys.argv[1:])
    print(f"Exécution de la commande: {command}")
    os.system(command)
