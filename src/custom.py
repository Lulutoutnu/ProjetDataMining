"""Composante personnalisée : loss et métrique écrites à la main.

La classe "contour" est rare et fine : une entropie croisée simple l'ignore.
D'où une loss combinée (entropie croisée pondérée + Dice).
"""
import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Loss = alpha * entropie croisée pondérée + (1 - alpha) * (1 - Dice).

    TODO : implémenter, choisir les poids de classes, justifier alpha.
    """

    def __init__(self, class_weights=None, alpha: float = 0.5):
        super().__init__()
        raise NotImplementedError

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, n_classes: int = 3):
    """Intersection sur union pour chaque classe.

    TODO : implémenter sans bibliothèque externe et gérer le cas union = 0.
    """
    raise NotImplementedError
