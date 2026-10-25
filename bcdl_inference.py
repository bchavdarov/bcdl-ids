"""
Recognize the person in one or more face images with a trained checkpoint.

    python3 bcdl_inference.py bcdl-faces/045.jpeg
    python3 bcdl_inference.py --checkpoint checkpoints/best_model.pt img1.jpg img2.jpg

Prints:  bcdl-faces/045.jpeg -> Al Pacino (0.97)
"""

import argparse

import torch

from bcdl_data import build_transform
from bcdl_defs import CHECKPOINT_DIR, get_device, load_image
from bcdl_model import BCSimpleCNN


def load_model(checkpoint_path, device):
    """Rebuild BCSimpleCNN from a checkpoint; returns (model, class_names)."""
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model = BCSimpleCNN(num_classes=ckpt["num_classes"], conv1_out=ckpt["conv1_out"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"]


def predict(model, class_names, image_path, device, transform):
    """Return (name, confidence) for the face in image_path."""
    x = transform(load_image(image_path)).unsqueeze(0).to(device)   # batch of 1
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    confidence, label = probs.max(dim=0)
    return class_names[label.item()], confidence.item()


def main():
    p = argparse.ArgumentParser(description="Recognize faces with a trained BCSimpleCNN")
    p.add_argument("images", nargs="+", help="image file(s) to classify")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_model.pt"))
    args = p.parse_args()

    device = get_device()
    model, class_names = load_model(args.checkpoint, device)
    transform = build_transform(train=False)      # same preprocessing as validation

    for path in args.images:
        name, confidence = predict(model, class_names, path, device, transform)
        print("{} -> {} ({:.2f})".format(path, name, confidence))


if __name__ == "__main__":
    main()
