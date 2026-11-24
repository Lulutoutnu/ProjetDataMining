"""Approche classique : classification pixel par pixel avec LightGBM.

Attention au volume : une image contient des dizaines de milliers de pixels.
Il faut sous-échantillonner les pixels pour l'entraînement.
"""


def build_model(params: dict):
    """TODO : retourner un LGBMClassifier configuré avec params."""
    raise NotImplementedError


def sample_pixels(features, labels, n_per_image: int, seed: int = 42):
    """TODO : sous-échantillonnage des pixels, en gardant la proportion des classes."""
    raise NotImplementedError
