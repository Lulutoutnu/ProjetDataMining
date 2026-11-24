"""Variables par pixel pour l'approche classique.

Pour chaque pixel : couleur (RGB, HSV), réponses de filtres (gradient, flou),
position normalisée dans l'image, etc.
"""
import numpy as np


def pixel_features(image: np.ndarray) -> np.ndarray:
    """Retourne un tableau (n_pixels, n_variables) pour une image (H, W, 3).

    TODO (équipe) : choisir les variables et justifier chaque choix dans le rapport.
    """
    raise NotImplementedError
