# Projet Machine Learning : segmentation Oxford-IIIT Pet

Projet du cours de Machine Learning (Master 2 MBFA, Université de Montpellier).
Rendu : dimanche 28 février 2027, 23h59.

## Problème

Segmentation sémantique d'images : classer chaque pixel d'une photo de chat ou de chien
en trois classes (animal, contour, fond), à partir du jeu Oxford-IIIT Pet
(37 races, environ 200 images par race, masques "trimap").

## Approches comparées

| Approche | Idée | Fichier |
|---|---|---|
| Classique | Classification pixel par pixel (LightGBM) sur des variables de couleur et de texture | `src/models_classic.py` |
| Deep learning | U-Net maison et U-Net avec encodeur pré-entraîné | `src/models_dl.py` |
| Custom | Loss combinée (entropie croisée + Dice) et métrique IoU écrites à la main | `src/custom.py` |

## Structure

```
data/raw/        archives téléchargées (non versionnées)
data/processed/  données extraites (non versionnées)
notebooks/       exploration uniquement (EDA)
src/             pipeline ML (scripts .py)
reports/         rapport PDF et figures
```

## Installation

### 1. Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell) : .venv\Scripts\Activate.ps1
```

### 2. Installer PyTorch (avant le reste)

PyTorch doit être installé à part. Sinon `pip` installe une version
sans GPU, qui rend l'entraînement du U-Net très lent.

**Avec une carte NVIDIA** (vérifier avec `nvidia-smi` : la « CUDA Version »
affichée en haut à droite doit être ≥ 12.6) :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Sans carte NVIDIA** (Mac, PC sans GPU) :

```bash
pip install torch torchvision
```

### 3. Installer les autres dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Avec un GPU NVIDIA, la commande doit afficher `True`. Si elle affiche `False`,
reprendre l'étape 2.

## Données

Télécharger `images.tar.gz` et `annotations.tar.gz` (Oxford-IIIT Pet,
https://www.robots.ox.ac.uk/~vgg/data/pets/) et les placer dans `data/raw/`, puis :

```bash
python -m src.data --extract
```

## Entraînement

```bash
python -m src.train --model classic
python -m src.train --model unet --epochs 30
```

## Équipe

- Lucas
- Kenza

Chaque membre travaille sur sa branche (`feat/...`) et passe par une Pull Request relue par l'autre.
