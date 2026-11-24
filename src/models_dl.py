"""Approche deep learning : U-Net (PyTorch).

Deux variantes à comparer :
1. U-Net entraîné de zéro ;
2. U-Net avec encodeur pré-entraîné (par exemple ResNet).
"""
import torch.nn as nn


class UNet(nn.Module):
    """U-Net écrit par l'équipe : encodeur, bottleneck, décodeur avec connexions de saut.

    TODO : implémenter __init__ et forward, puis documenter l'architecture
    (nombre de niveaux, canaux) dans le rapport.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_pretrained_unet(n_classes: int = 3) -> nn.Module:
    """TODO : U-Net avec encodeur pré-entraîné."""
    raise NotImplementedError
