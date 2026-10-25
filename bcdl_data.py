"""
Data loading for bcdl-ids: annotations.csv + images -> batches of tensors.

Pipeline:
    read_annotations()   annotations.csv     -> samples [(path, label), ...]
    split_samples()      samples             -> train_samples, val_samples
    build_transform()    PIL image           -> normalized tensor (with or without augmentation)
    BCDLDataset          one sample          -> (tensor, label), image opened lazily
    build_dataloaders()  all of the above    -> train_loader, val_loader

Self-test:
    python3 bcdl_data.py
"""

import csv
import os
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from bcdl_defs import ANNOTATIONS_CSV, DATA_DIR, IMAGE_SIZE, MEAN, STD, load_image


def read_annotations(csv_path=ANNOTATIONS_CSV, image_dir=DATA_DIR):
    """Return samples: [(image path, label), ...] read from annotations.csv."""
    samples = []
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(str(image_dir), row["imagename"].strip())
            samples.append((path, int(row["label"])))
    return samples


def split_samples(samples, val_split=0.2, seed=42):
    """Split samples into (train, val), taking val_split of EACH person for val.

    The fixed seed gives the same split on every run, so results are comparable.
    """
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_split))
        val += group[:n_val]
        train += group[n_val:]
    return train, val


def build_transform(train):
    """Return the image -> tensor transform.

    train=True adds random changes (crop, flip, light colour shift) so the
    network sees a slightly different picture of each face every epoch.
    That matters with only 60 images: it fights memorizing them.
    """
    to_tensor = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ] + to_tensor)
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))] + to_tensor)


class BCDLDataset(Dataset):
    """Serves (image tensor, label) pairs; each image is opened only when asked for."""

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


def build_dataloaders(batch_size=8, val_split=0.2, num_workers=2, seed=42):
    """Return (train_loader, val_loader) for the dataset in DATA_DIR."""
    train_samples, val_samples = split_samples(read_annotations(), val_split, seed)

    train_loader = DataLoader(
        BCDLDataset(train_samples, build_transform(train=True)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        BCDLDataset(val_samples, build_transform(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    samples = read_annotations()
    train, val = split_samples(samples)
    print("samples:", len(samples), "| train:", len(train), "| val:", len(val))

    train_loader, _ = build_dataloaders(num_workers=0)
    images, labels = next(iter(train_loader))
    print("one batch:", tuple(images.shape), "labels:", labels.tolist())
