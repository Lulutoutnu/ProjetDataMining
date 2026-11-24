"""Chargement des données et découpage train / validation / test.

Le jeu Oxford-IIIT Pet fournit un découpage officiel (trainval.txt et test.txt).
Le jeu de test officiel n'est jamais utilisé avant l'évaluation finale.
La validation est tirée du trainval, de façon stratifiée par race.
"""
import argparse
import tarfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SEED = 42


def extract_archives(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    """Extrait images.tar.gz et annotations.tar.gz dans data/processed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images.tar.gz", "annotations.tar.gz"):
        archive = raw_dir / name
        if not archive.exists():
            raise FileNotFoundError(f"Archive manquante : {archive}")
        with tarfile.open(archive) as tar:
            tar.extractall(out_dir)  # archives officielles de confiance


def read_split_file(path: Path) -> pd.DataFrame:
    """Lit trainval.txt ou test.txt.

    Colonnes du fichier : nom de l'image, id de classe (1 à 37),
    espèce (1 = chat, 2 = chien), id de race dans l'espèce.
    """
    cols = ["image", "class_id", "species", "breed_id"]
    return pd.read_csv(path, sep=" ", comment="#", names=cols)


def make_splits(processed_dir: Path = PROCESSED_DIR, val_size: float = 0.2):
    """Retourne trois DataFrames : train, val, test.

    TODO (équipe) : justifier val_size dans le rapport et vérifier
    qu'aucune image n'apparaît dans deux jeux.
    """
    ann = processed_dir / "annotations"
    trainval = read_split_file(ann / "trainval.txt")
    test = read_split_file(ann / "test.txt")
    train, val = train_test_split(
        trainval,
        test_size=val_size,
        stratify=trainval["class_id"],
        random_state=SEED,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract_archives()
