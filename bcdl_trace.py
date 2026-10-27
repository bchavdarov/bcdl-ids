"""
Trace one image through BCSimpleCNN layer by layer: see every tensor on its way.

    python3 bcdl_trace.py bcdl-faces/045.jpeg              # trained model (if checkpoint exists)
    python3 bcdl_trace.py bcdl-faces/045.jpeg --maps       # + save feature-map PNGs
    python3 bcdl_trace.py --untrained                      # random weights, random input

Functions:
    short(layer)                  short readable layer name, e.g. "Conv2d 3->32, k3"
    trace(model, x, device)       run x through each layer separately, print the table,
                                  return [(index, layer, output tensor), ...]
    save_maps(outputs, out_dir)   save the first channels of each 4-D output as one PNG

Prints for each layer: output shape, parameter count, mean/std of the values
and the time the layer took (GPU-synchronized, so the times are real).
PNGs are written with PIL only: no display or matplotlib needed in the container.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from bcdl_data import build_transform
from bcdl_defs import CHECKPOINT_DIR, IMAGE_SIZE, PROJECT_DIR, get_device, load_image
from bcdl_inference import load_model
from bcdl_model import BCSimpleCNN


def short(layer):
    """Short readable name for each layer type."""
    if isinstance(layer, nn.Conv2d):
        return "Conv2d {}->{}, k{}".format(layer.in_channels, layer.out_channels, layer.kernel_size[0])
    if isinstance(layer, nn.BatchNorm2d):
        return "BatchNorm2d({})".format(layer.num_features)
    if isinstance(layer, nn.MaxPool2d):
        return "MaxPool2d({})".format(layer.kernel_size)
    if isinstance(layer, nn.AdaptiveAvgPool2d):
        return "AdaptiveAvgPool2d{}".format(tuple(layer.output_size))
    if isinstance(layer, nn.Linear):
        return "Linear {}->{}".format(layer.in_features, layer.out_features)
    if isinstance(layer, nn.Dropout):
        return "Dropout({})".format(layer.p)
    return layer.__class__.__name__


def sync(device):
    """Wait for the GPU to finish; CUDA calls are asynchronous, so timing needs this."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def trace(model, x, device):
    """Pass x through each layer separately; print one row per layer.

    Returns [(index, layer, output), ...] with the outputs moved to the CPU.
    """
    with torch.no_grad():
        model(x)                      # warm-up: first CUDA run includes cuDNN setup time
        sync(device)

    outputs = []
    header = "{:>3}  {:<24} {:<20} {:>9}  {:>7} {:>7} {:>8}"
    row = "{:>3}  {:<24} {:<20} {:>9,}  {:>7.3f} {:>7.3f} {:>8.2f}"
    print(header.format("#", "layer", "output shape", "params", "mean", "std", "ms"))
    print("{:>3}  {:<24} {:<20}".format("", "input", str(list(x.shape))))

    i, total_ms = 0, 0.0
    with torch.no_grad():
        for block_name, block in (("features", model.features), ("classifier", model.classifier)):
            print("---- {} ----".format(block_name))
            for layer in block:
                start = time.time()
                x = layer(x)
                sync(device)
                ms = (time.time() - start) * 1000
                total_ms += ms

                n = sum(p.numel() for p in layer.parameters())
                print(row.format(i, short(layer), str(list(x.shape)), n,
                                 x.mean().item(), x.std().item(), ms))
                outputs.append((i, layer, x.cpu()))
                i += 1

    total = sum(p.numel() for p in model.parameters())
    print("\nTotal parameters: {:,} | layer time sum: {:.2f} ms".format(total, total_ms))
    return outputs


def to_gray(m):
    """2-D tensor -> 8-bit grayscale PIL image, stretched to its own min..max."""
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return Image.fromarray((m * 255).byte().numpy(), mode="L")


def save_maps(outputs, out_dir, n=8, tile=112):
    """Save the first n channels after each ReLU / pooling layer as one PNG strip.

    Each channel becomes a tile x tile square (NEAREST keeps the 4x4 pixels sharp),
    so the shrinking 224 -> 112 -> 56 -> 28 -> 4 is visible as growing blockiness.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, layer, t in outputs:
        if t.dim() != 4 or not isinstance(layer, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            continue
        maps = t[0, :n]
        strip = Image.new("L", (tile * len(maps) + 2 * (len(maps) - 1), tile), 255)
        for k, m in enumerate(maps):
            strip.paste(to_gray(m).resize((tile, tile), Image.NEAREST), (k * (tile + 2), 0))
        h, w = t.shape[2], t.shape[3]
        path = out_dir / "{:02d}_{}_{}x{}.png".format(i, layer.__class__.__name__, h, w)
        strip.save(str(path))
        print("saved", path)


def parse_args():
    p = argparse.ArgumentParser(description="Trace tensors through BCSimpleCNN layer by layer")
    p.add_argument("image", nargs="?", help="face image; omitted -> random tensor")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_model.pt"))
    p.add_argument("--untrained", action="store_true", help="ignore checkpoint, use random weights")
    p.add_argument("--maps", action="store_true", help="save feature-map PNGs")
    p.add_argument("--out", default=str(PROJECT_DIR / "trace"), help="folder for the PNGs")
    p.add_argument("--channels", type=int, default=8, help="channels per PNG strip")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    class_names = None
    if args.untrained:
        model = BCSimpleCNN().to(device).eval()
    else:
        model, class_names = load_model(args.checkpoint, device)

    if args.image:
        x = build_transform(train=False)(load_image(args.image)).unsqueeze(0)  # [1, 3, H, W]
    else:
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = x.to(device)

    print("device: {} | weights: {} | input: {}\n".format(
        device, "random" if args.untrained else args.checkpoint, args.image or "random tensor"))
    outputs = trace(model, x, device)

    logits = outputs[-1][2][0]
    probs = torch.softmax(logits, dim=0)
    print("logits: {} | probabilities: {}".format(
        [round(v, 3) for v in logits.tolist()], [round(v, 3) for v in probs.tolist()]))
    if class_names:
        conf, label = probs.max(dim=0)
        print("prediction: {} ({:.2f})".format(class_names[label.item()], conf.item()))

    if args.maps:
        save_maps(outputs, Path(args.out), n=args.channels)


if __name__ == "__main__":
    main()
