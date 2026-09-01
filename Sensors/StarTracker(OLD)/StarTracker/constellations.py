# constellations.py

import numpy as np

def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=float)

# Big Dipper (Ursa Major) star positions
BIG_DIPPER_STARS = {
    "Dubhe":  (165.46, 61.75),
    "Merak":  (165.93, 56.38),
    "Phecda": (168.53, 53.69),
    "Megrez": (168.53, 57.03),
    "Alioth": (174.00, 55.96),
    "Mizar":  (176.46, 54.92),
    "Alkaid": (180.00, 49.31),
}

# Line segments between stars
BIG_DIPPER_LINES = [
    ("Dubhe", "Merak"),
    ("Merak", "Phecda"),
    ("Phecda", "Megrez"),
    ("Megrez", "Alioth"),
    ("Megrez", "Dubhe"),
    ("Alioth", "Mizar"),
    ("Mizar", "Alkaid"),
]

def get_constellation_vectors(star_dict):
    return {name: radec_to_vec(*coords) for name, coords in star_dict.items()}
