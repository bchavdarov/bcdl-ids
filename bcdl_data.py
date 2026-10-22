"""
Data loading for bcdl-ids.

Reads the face images and their annotations and serves them to the CNN
as batches of tensors. Designed for Jetson Nano (JetPack 4.6.1,
PyTorch 1.10.0, Python 3.6) and Google Colab.

Self-test:
    python3 bcdl_data.py
"""

import csv
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Step 1: read the annotations file
# ---------------------------------------------------------------------------

def read_annotations(csv_path):
    """Read annotations.csv and return a list of (imagename, person) pairs.

    The CSV must start with a header row naming the columns: imagename,person
    """
    
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append((row["imagename"].strip(), row["person"].strip()))
    return records
    

class BcdlIdsDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]
        return sample, label

if __name__ == "__main__":
    records = read_annotations("bcdl-faces/annotations.csv")
    print(len(records))   # expect 60
    print(records[0])     # first De Niro image
    print(records[-1])    # last Pacino image
