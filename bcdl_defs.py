"""
Shared constants and helpers for bcdl-ids (used by data, training and inference).

Contents:
    paths and settings      PROJECT_DIR, DATA_DIR, ANNOTATIONS_CSV, IDENTITIES_CSV,
                            CHECKPOINT_DIR, IMAGE_SIZE, MEAN, STD
    get_device()            CUDA if available, else CPU
    load_image(path)        open an image file as an RGB PIL image
    load_identities()       {label: name} read from identities.csv
    evaluate(...)           average loss and accuracy of a model over a DataLoader

Target: Jetson Nano (JetPack 4.6.1, PyTorch 1.10, Python 3.6) and desktop/Colab.
"""

import csv
import os
from pathlib import Path

import torch
from PIL import Image

# --- paths -------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset folder: bcdl-faces/ inside the project, or set BCDL_DATA_DIR.
DATA_DIR = Path(os.environ.get("BCDL_DATA_DIR", str(PROJECT_DIR / "bcdl-faces")))
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"   # imagename,label
IDENTITIES_CSV = DATA_DIR / "identities.csv"     # label,name
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# --- image settings ----------------------------------------------------------
IMAGE_SIZE = 224                     # images are resized to IMAGE_SIZE x IMAGE_SIZE
MEAN = (0.485, 0.456, 0.406)         # ImageNet statistics, standard for RGB photos
STD = (0.229, 0.224, 0.225)


def get_device():
    """Return the GPU if PyTorch can use it, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path, size=IMAGE_SIZE):
    """Return the image at path as an RGB PIL image (file is closed).

    For JPEGs, draft() lets the decoder skip detail we would throw away anyway:
    a 3000x3000 photo is decoded at roughly 375x375 instead of full size, which
    saves a lot of CPU time on the Nano. The result is never smaller than size.
    """
    with Image.open(path) as image:
        image.draft("RGB", (size, size))
        return image.convert("RGB")  # convert() also reads the pixels into memory


def load_identities(csv_path=IDENTITIES_CSV):
    """Return {label: name}, e.g. {0: 'Al Pacino', 1: 'Robert De Niro'}."""
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        return {int(row["label"]): row["name"].strip() for row in csv.DictReader(f)}


def evaluate(model, loader, criterion, device):
    """Return (average loss, accuracy) of model over all batches in loader."""
    model.eval()                     # BatchNorm uses running stats, Dropout off
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():            # no gradients needed: faster, less memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
