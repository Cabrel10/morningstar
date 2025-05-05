#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour créer un logo simple pour Morningstar.
"""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# Définir le chemin du logo
LOGO_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "morningstar_logo.png"

# Créer une image avec un fond transparent
width, height = 500, 500
image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# Dessiner un cercle doré (représentant l'étoile du matin)
circle_center = (width // 2, height // 2)
circle_radius = 150
circle_color = (255, 215, 0, 255)  # Or
draw.ellipse(
    (circle_center[0] - circle_radius, circle_center[1] - circle_radius,
     circle_center[0] + circle_radius, circle_center[1] + circle_radius),
    fill=circle_color
)

# Dessiner un symbole de crypto (Bitcoin) au centre
inner_circle_radius = 100
inner_circle_color = (0, 0, 0, 200)  # Noir semi-transparent
draw.ellipse(
    (circle_center[0] - inner_circle_radius, circle_center[1] - inner_circle_radius,
     circle_center[0] + inner_circle_radius, circle_center[1] + inner_circle_radius),
    fill=inner_circle_color
)

# Dessiner le symbole '$' en blanc au centre
try:
    # Essayer de charger une police
    font = ImageFont.truetype("arial.ttf", 150)
except IOError:
    # Si la police n'est pas disponible, utiliser la police par défaut
    font = ImageFont.load_default()

# Dessiner le symbole '$' en blanc au centre
draw.text((width // 2 - 40, height // 2 - 75), "$", fill=(255, 255, 255, 255), font=font)

# Enregistrer l'image
image.save(LOGO_PATH)
print(f"Logo créé et enregistré à {LOGO_PATH}")
