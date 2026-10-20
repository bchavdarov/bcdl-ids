"""
Shared helper functions for bcdl-ids.

Small utilities used by more than one module in the project (data
loading, model, training), kept here so they are defined only once.
Designed for Jetson Nano (JetPack 4.6.1, PyTorch 1.10.0, Python 3.6)
and Google Colab.

Contents:
    read_class_names(csv_path)
        Read annotations.csv (header: imagename,person) and return the
        sorted list of unique person names. The sorted order gives each
        identity a stable class index, e.g.
        ['Al Pacino', 'Robert De Niro'] -> 0, 1.
"""

import csv
import os
from pathlib import Path

# Folder that contains this file, i.e. the project root.
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset folder. Defaults to bcdl-faces/ inside the project;
# set the environment variable BCDL_DATA_DIR to use another location.
DATA_DIR = Path(os.environ.get("BCDL_DATA_DIR", PROJECT_DIR / "bcdl-faces"))
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"

def read_class_names(csv_path):
    """Read the unique person names from annotations.csv.

    Returns:
        list[str]: names sorted alphabetically; the index is the class id,
        e.g. ['Al Pacino', 'Robert De Niro'].
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sorted({row["person"].strip() for row in csv.DictReader(f)})

