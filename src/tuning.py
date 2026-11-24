"""Recherche d'hyperparamètres avec Optuna.

Les essais sont évalués sur la validation, jamais sur le test.
Pistes : taux d'apprentissage, taille d'image, alpha de la loss,
profondeur du U-Net, paramètres de LightGBM.
"""


def objective(trial):
    """TODO : définir l'espace de recherche, entraîner, retourner l'IoU moyen de validation."""
    raise NotImplementedError


def run_study(n_trials: int = 30):
    """TODO : créer l'étude Optuna, la lancer et sauvegarder les résultats."""
    raise NotImplementedError
