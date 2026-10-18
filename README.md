# bcdl-ids

Experimental PyTorch CNN that recognizes a person's identity from a small set of face images. It is being built step by step and targets the NVIDIA Jetson Nano.

**Status:** work in progress. The model and annotation reading are done. Image loading, training and inference are not written yet.

## Target environment

- Jetson Nano, JetPack 4.6.1, PyTorch 1.10.0, Python 3.6 (inside the `jetson-inference` container)
- Google Colab for training and experiments

## Files

| File | Purpose |
|------|---------|
| `bcdl_model.py` | `BCSimpleCNN`: a small CNN (5 conv layers with BatchNorm, then 2 fully connected layers, ~665k parameters). Works with any input size, e.g. 32×32 or 224×224. |
| `bcdl_data.py` | Data loading. Currently reads `annotations.csv` into `(imagename, person)` pairs. |
| `bcdl_defs.py` | Shared helpers, e.g. `read_class_names()`, which returns the sorted list of identities. |

## Dataset

The `bcdl-faces/` folder holds 60 face images (`001.jpeg` … `060.jpeg`): 30 of Robert De Niro and 30 of Al Pacino. It also holds `annotations.csv`, whose rows look like this:

```
imagename,person
001.jpeg,Robert De Niro
```

The dataset is not committed (it is listed in `.gitignore`). Put it next to the scripts before running them.

## Quick checks

```bash
python3 bcdl_model.py   # prints the output shape and parameter count
python3 bcdl_data.py    # prints the annotation count and the first and last records
```
