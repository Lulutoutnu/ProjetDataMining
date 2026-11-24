"""Point d'entrée de l'entraînement.

Exemples :
    python -m src.train --model classic
    python -m src.train --model unet --epochs 30
"""
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement segmentation Oxford-IIIT Pet")
    parser.add_argument("--model", choices=["classic", "unet", "unet_pretrained"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO : charger les splits (src.data.make_splits), construire le modèle,
    # entraîner, évaluer sur la validation, sauvegarder les courbes et les métriques.
    raise NotImplementedError(f"Entraînement du modèle {args.model} à écrire")


if __name__ == "__main__":
    main()
