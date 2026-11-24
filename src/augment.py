"""Augmentation de données pour la segmentation.

Règle essentielle : toute transformation géométrique appliquée à l'image
doit être appliquée au masque. Les transformations de couleur ne touchent
que l'image. L'augmentation s'applique uniquement au jeu d'entraînement.
"""


def get_train_transforms(image_size: int):
    """TODO : flips, rotations légères, variations de luminosité, redimensionnement."""
    raise NotImplementedError


def get_eval_transforms(image_size: int):
    """TODO : redimensionnement et normalisation seulement, sans aléa."""
    raise NotImplementedError
