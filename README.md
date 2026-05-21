# Emotion Recognition

A neural network for facial emotion recognition built from scratch using only Python standard library — no NumPy, no PyTorch, no external dependencies.

Recognises 5 emotions: **smiling, neutral, sad, angry, surprised**.

---

## Project structure

```text
emotion-recognition/
├── src/
│   ├── generate_faces.py   # generates synthetic PGM face images
│   ├── image_loader.py     # reads PGM files into pixel arrays
│   ├── model.py            # neural network (activations, loss, forward, backward, update)
│   ├── train.py            # loads data, trains the network, exports results
│   ├── infer.py            # runs inference on a single image
│   ├── validate.py         # confusion matrix and per-class accuracy report
│   └── metrics.py          # accuracy, precision, recall, F1 and file exports
├── results_v1_downsample_hidden8/    # results with downsampling + hidden size 8 (73% accuracy)
├── results_v2_fullres_hidden64/      # results with full resolution + hidden size 64 (96% accuracy)
├── results_v3_kfold_cv/             # results with k-fold cross-validation
├── results_bw_color_comparison/     # side-by-side BW vs color comparison
└── tests/                  # unit tests for every module
```

---

## Setup

Requires Python 3.8+. No packages to install — the standard library is all that is used.

Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

---

## Quickstart

### 1. Generate the dataset

```bash
# Grayscale PGM (default)
python src/generate_faces.py --count 100 --seed 42

# Color PPM
python src/generate_faces.py --count 100 --seed 42 --color
```

Creates 500 synthetic 16×16 images (100 per emotion) split into `train/`, `validation/`, and `test/` subdirectories. Output goes to `data/black_white/` by default, or `data/color/` with `--color`.

| Flag | Default | Description |
| --- | --- | --- |
| `--count` | 10 | Images per emotion class |
| `--out` | pgm_faces | Output directory |
| `--seed` | 42 | Random seed for reproducibility |

---

### 2. Train the network

```bash
# Grayscale (default)
python src/train.py --epochs 20 --hidden-size 64 --learning-rate 0.1

# Color
python src/train.py --data-dir data/color --color --epochs 20 --hidden-size 64 --learning-rate 0.1
```

Runs stratified 5-fold cross-validation, prints per-fold accuracy, then trains a final model on all CV data and evaluates it on the held-out test set. All results are saved to `results_v3_kfold_cv/`.

```text
input size:  256
hidden size: 64
cv pool / test: 400 / 100
folds: 5

fold 1/5  val acc=0.925  (train=320, val=80)
fold 2/5  val acc=0.912  (train=320, val=80)
fold 3/5  val acc=0.975  (train=320, val=80)
fold 4/5  val acc=0.938  (train=320, val=80)
fold 5/5  val acc=0.963  (train=320, val=80)

cross-validation accuracy: 0.943 ± 0.023

aggregated CV report
----------------------------------------
per-class accuracy:
  smiling     ...
  neutral     ...
  ...

training final model on all CV data...
test accuracy: 0.940

test set report
----------------------------------------
per-class accuracy:
  smiling     17/20  0.85  #################
  neutral     20/20  1.00  ####################
  sad         19/20  0.95  ###################
  angry       19/20  0.95  ###################
  surprised   19/20  0.95  ###################

results saved to results_v3_kfold_cv/
```

| Flag | Default | Description |
| --- | --- | --- |
| `--epochs` | 20 | Number of full passes over the training set |
| `--hidden-size` | 64 | Number of neurons in the hidden layer |
| `--learning-rate` | 0.1 | Step size for gradient descent |
| `--folds` | 5 | Number of folds for stratified k-fold cross-validation |
| `--data-dir` | data | Directory containing the train/validation/test splits |
| `--results-dir` | results | Directory where output files are written |
| `--seed` | 42 | Random seed for weight initialisation and shuffling |

---

### 3. Inspect the results

After training, four files are written to `results_v3_kfold_cv/`:

**`metrics.json`** — overall and per-class scores:

```json
{
  "accuracy": 0.96,
  "macro_precision": 0.9605,
  "macro_recall": 0.96,
  "macro_f1": 0.96,
  "per_class": {
    "smiling":   { "precision": 0.95,   "recall": 0.95, "f1": 0.95,   "support": 20 },
    "neutral":   { "precision": 0.9524, "recall": 1.00, "f1": 0.9756, "support": 20 },
    "sad":       { "precision": 0.95,   "recall": 0.95, "f1": 0.95,   "support": 20 },
    "angry":     { "precision": 0.95,   "recall": 0.95, "f1": 0.95,   "support": 20 },
    "surprised": { "precision": 1.00,   "recall": 0.95, "f1": 0.9744, "support": 20 }
  }
}
```

**`confusion_matrix.csv`** — rows are the true label, columns are the predicted label:

```text
,smiling,neutral,sad,angry,surprised
smiling,19,1,0,0,0
neutral,0,20,0,0,0
...
```

**`training_log.csv`** — averaged metrics per epoch across all folds:

```text
epoch,train_loss,train_acc,val_loss,val_acc
1,0.854800,0.6,0.877400,0.62
2,0.488000,0.81,0.644700,0.76
...
```

**`kfold_summary.json`** — cross-validation results:

```json
{
  "folds": 5,
  "fold_accuracies": [0.925, 0.9125, 0.975, 0.9375, 0.9625],
  "mean_accuracy": 0.943,
  "std_accuracy": 0.023
}
```

---

### 4. Try it out

Run the trained model on a single grayscale PGM image. The model is saved to `<results-dir>/model.json` after training — the default results directory is `results_v3_kfold_cv/`:

```bash
python src/infer.py \
  --image data/black_white/test/smiling/smiling_0000.pgm \
  --model results_v3_kfold_cv/model.json
```

If you used a custom `--results-dir` during training, point `--model` at that directory instead:

```bash
python src/infer.py --image data/black_white/test/sad/sad_0003.pgm --model my_results/model.json
```

Example output:

```text
prediction: smiling

confidences:
  smiling     0.941  ###################
  neutral     0.031
  surprised   0.018
  sad         0.007
  angry       0.003
```

| Flag | Default | Description |
| --- | --- | --- |
| `--image` | *(required)* | Path to a grayscale PGM image file |
| `--model` | `results/model.json` | Path to the saved model JSON |

> **Note:** `infer.py` supports grayscale PGM images only. Color PPM images are not supported for inference.

---

### 5. Run the tests

```bash
python -m unittest discover
```

All tests should pass. They cover every function in `model.py`, `train.py`, `validate.py`, `metrics.py`, `image_loader.py`, and `generate_faces.py`.

---

## How it works

The network has three layers:

```text
input (256)  →  hidden (64, ReLU)  →  output (5, softmax)
```

Each 16×16 image is flattened to a 256-value vector and pixel values are normalised to `[-0.5, 0.5]` before being fed in. Training uses stochastic gradient descent — one weight update per example — with the dataset shuffled each epoch.

| File | Responsibility |
| --- | --- |
| `model.py` | Maths: activations, loss, forward pass, backpropagation, weight update |
| `generate_faces.py` | Data: draws synthetic faces with randomised features and noise |
| `image_loader.py` | I/O: reads PGM files into pixel arrays |
| `train.py` | Loop: loads data, trains, evaluates, exports results |
| `infer.py` | Inference: loads a saved model and predicts on a single image |
| `validate.py` | Analysis: confusion matrix and per-class accuracy report |
| `metrics.py` | Scores: accuracy, precision, recall, F1 and file export functions |

---

## Results history

### Version comparison

| Version | Input | Hidden size | Test accuracy |
| --- | --- | --- | --- |
| V1 | 8×8 (downsampled) → 64 values | 8 | 0.73 |
| V2 | 16×16 (full resolution) → 256 values | 64 | 0.96 |
| V3 (k-fold CV) | 16×16 → 256 values | 64 | 0.94 |

Removing the downsampling step was the biggest single improvement — the 16×16 images are small enough that no size reduction is needed, and downsampling was discarding features the network needed to distinguish emotions, particularly **neutral**.

---

### Grayscale vs color (V3 settings, 5-fold CV)

Both models use hidden size 64, 20 epochs, and 5-fold cross-validation. The color model uses 8×8 downsampled PPM images (192 features, 3 RGB channels); the grayscale model uses 16×16 PGM images (256 features).

| Metric | Grayscale | Color |
| --- | --- | --- |
| Test accuracy | **0.94** | 0.87 |
| Macro F1 | **0.9413** | 0.8704 |
| Macro precision | **0.9489** | 0.8845 |
| CV mean accuracy | **0.943 ± 0.023** | 0.903 ± 0.017 |

**Per-class F1:**

| Class | Grayscale | Color |
| --- | --- | --- |
| smiling | 0.895 | 0.857 |
| neutral | 0.889 | 0.737 |
| sad | 0.974 | **0.976** |
| angry | 0.974 | 0.783 |
| surprised | 0.974 | **1.000** |

The grayscale model outperforms color across almost every metric. The classes most affected by the switch to color are **neutral** (F1 −0.15) and **angry** (F1 −0.19) — the 8×8 downsampling erases the fine eyebrow and mouth detail that the network relies on to distinguish these two emotions. Conversely, **sad** and **surprised** hold up or slightly improve with color, because chromatic cues (blue tears, dark mouth interior) survive the resolution reduction.
