"""
Data loading for bcdl-ids.

Reads the face images and their annotations and serves them to the CNN
as batches of tensors. Designed for Jetson Nano (JetPack 4.6.1,
PyTorch 1.10.0, Python 3.6) and Google Colab.

Self-test:
    python3 bcdl_data.py
"""

import csv
import os
from torch.utils.data import Dataset
from bcdl_defs import ANNOTATIONS_CSV, DATA_DIR, load_image


# ---------------------------------------------------------------------------
# Step 1: read the annotations file
# ---------------------------------------------------------------------------


def read_annotations(csv_path=ANNOTATIONS_CSV, image_dir=DATA_DIR):
    """Return samples: [(path, label), ...] read from annotations.csv."""
    samples = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            path = os.path.join(image_dir, row['imagename'])
            samples.append((path, int(row['label'])))
    return samples


class BCDLDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = load_image(path)
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    records = read_annotations("bcdl-faces/annotations.csv")
    print(len(records))   # expect 60
    print(records[0])     # first De Niro image
    print(records[-1])    # last Pacino image
