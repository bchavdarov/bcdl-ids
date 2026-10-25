"""
Train BCSimpleCNN on the bcdl-faces dataset.

    python3 bcdl_train.py                 # defaults below
    python3 bcdl_train.py --epochs 40 --lr 5e-4

Each epoch prints training loss, validation loss and validation accuracy.
The best epoch (by validation accuracy) is saved to checkpoints/best_model.pt
together with everything inference needs: class names, num_classes, conv1_out
and the image size.
"""

import argparse
import time

import torch
import torch.nn as nn

from bcdl_data import build_dataloaders
from bcdl_defs import CHECKPOINT_DIR, IMAGE_SIZE, evaluate, get_device, load_identities
from bcdl_model import BCSimpleCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training images; returns the average training loss."""
    model.train()                    # BatchNorm updates its stats, Dropout on
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)   # forward pass + loss
        loss.backward()                           # gradients
        optimizer.step()                          # update weights

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


def parse_args():
    p = argparse.ArgumentParser(description="Train BCSimpleCNN on bcdl-faces")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--conv1_out", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    class_names = load_identities()               # {0: 'Al Pacino', 1: 'Robert De Niro'}
    num_classes = len(class_names)
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.val_split, args.num_workers, args.seed)
    print("device: {} | classes: {} | train: {} | val: {}".format(
        device, class_names, len(train_loader.dataset), len(val_loader.dataset)))

    model = BCSimpleCNN(num_classes=num_classes, conv1_out=args.conv1_out).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / "best_model.pt"
    best = (-1.0, 0.0)                            # (val_acc, -val_loss): higher is better

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        mark = ""
        if (val_acc, -val_loss) > best:           # ties in accuracy -> lower loss wins
            best = (val_acc, -val_loss)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "conv1_out": args.conv1_out,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "val_acc": val_acc,
            }, str(best_path))
            mark = "  <- saved"

        print("epoch {:3d}/{} | train_loss {:.4f} | val_loss {:.4f} | val_acc {:.3f} | {:.1f}s{}".format(
            epoch, args.epochs, train_loss, val_loss, val_acc, time.time() - start, mark))

    print("done. best val_acc {:.3f} -> {}".format(best[0], best_path))


if __name__ == "__main__":
    main()
