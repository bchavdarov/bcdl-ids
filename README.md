# bcdl-ids

Experimental PyTorch CNN that recognizes a person's identity from a small set of face images. It is being built step by step and targets the NVIDIA Jetson Nano.

**Status:** work in progress. The model, annotation reading and the PyTorch `Dataset` are done. Transforms, training and inference are not written yet.

## Target environment

- Jetson Nano, JetPack 4.6.1, PyTorch 1.10.0, Python 3.6 (inside the `jetson-inference` container)
- Google Colab for training and experiments

## Files

| File | Purpose |
|------|---------|
| `bcdl_model.py` | `BCSimpleCNN`: a small CNN (5 conv layers with BatchNorm, then 2 fully connected layers, ~665k parameters). Works with any input size, e.g. 32×32 or 224×224. |
| `bcdl_data.py` | Data loading: `read_annotations()` returns `(image_path, label)` pairs; `BCDLDataset` serves `(image, label)` samples. |
| `bcdl_defs.py` | Shared helpers: dataset paths (`DATA_DIR`, overridable with `BCDL_DATA_DIR`), `read_class_names()`, `load_image()`. |

## Dataset

The `bcdl-faces/` folder holds 60 face images (`001.jpeg` … `060.jpeg`): 30 of Robert De Niro and 30 of Al Pacino. It also holds `annotations.csv`, whose rows look like this:

```
imagename,person,label
001.jpeg,Robert De Niro,1
```

`label` is the index of `person` in the alphabetically sorted list of names (Al Pacino = 0, Robert De Niro = 1). `bcdl-faces/fgen.py` regenerates the file.

The dataset is not committed (it is listed in `.gitignore`). Put it next to the scripts before running them.

## Quick checks

```bash
python3 bcdl_data.py    # prints the sample count and the first and last samples
```
