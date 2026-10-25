# bcdl-ids

PyTorch CNN that recognizes a person's identity from face images. Targets the NVIDIA Jetson Nano (JetPack 4.6.1, PyTorch 1.10, Python 3.6, inside the `jetson-inference` container); also runs on a desktop or Colab.

## Files

| File | What it contains |
|------|------------------|
| `bcdl_model.py` | `BCSimpleCNN(num_classes, conv1_out)`: 5 conv layers with BatchNorm, then 2 fully connected layers |
| `bcdl_defs.py` | Shared paths and settings, `get_device()`, `load_image()`, `load_identities()`, `evaluate()` |
| `bcdl_data.py` | `read_annotations()`, `split_samples()`, `build_transform()`, `BCDLDataset`, `build_dataloaders()` |
| `bcdl_train.py` | `train_one_epoch()`, `main()`: trains and saves `checkpoints/best_model.pt` |
| `bcdl_inference.py` | `load_model()`, `predict()`, `main()`: name + confidence for given images |

## Dataset (`bcdl-faces/`, not committed)

- `001.jpeg` … `060.jpeg`: 001–030 Robert De Niro, 031–060 Al Pacino
- `annotations.csv`: `imagename,label`
- `identities.csv`: `label,name` (0 = Al Pacino, 1 = Robert De Niro). A new person gets the next free number.

Set `BCDL_DATA_DIR` to use a dataset somewhere else.

## Run

```bash
python3 bcdl_data.py                          # self-test: sample counts and one batch shape
python3 bcdl_train.py                         # 30 epochs, batch 8, lr 1e-3
python3 bcdl_inference.py bcdl-faces/045.jpeg
```

If the DataLoader crashes inside Docker (bus error / worker killed), add `--num_workers 0`.
