# Emotion Recognition

A neural network for facial emotion recognition built from scratch using only Python standard library — no NumPy, no PyTorch, no external dependencies.

Recognises 5 emotions: **smiling, neutral, sad, angry, surprised**.

---

## Project structure

```
emotion-recognition/
├── src/
│   ├── generate_faces.py   # generates synthetic PGM face images
│   ├── image_loader.py     # reads PGM files into pixel arrays
│   ├── model.py            # neural network (activations, loss, forward, backward, update)
│   ├── train.py            # loads data, trains the network, exports results
│   ├── validate.py         # confusion matrix and per-class accuracy report
│   └── metrics.py          # accuracy, precision, recall, F1 and file exports
├── results_v1_downsample_hidden8/    # results with downsampling + hidden size 8 (73% accuracy)
├── results_v2_fullres_hidden64/      # results with full resolution + hidden size 64 (96% accuracy)
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
python src/generate_faces.py --count 100 --out data --seed 42
```

Creates 500 synthetic 16×16 grayscale PGM images (100 per emotion) split into `data/train/`, `data/validation/`, and `data/test/`.

| Flag | Default | Description |
|---|---|---|
| `--count` | 10 | Images per emotion class |
| `--out` | pgm_faces | Output directory |
| `--seed` | 42 | Random seed for reproducibility |

---

### 2. Train the network

```bash
python src/train.py --epochs 20 --hidden-size 64 --learning-rate 0.1
```

Prints loss and accuracy on both train and validation sets each epoch, then shows a full test report and saves all results to `results/`.

```
input size:  256
hidden size: 64
train / val / test: 300 / 100 / 100

epoch 001  train loss=0.8548 acc=0.60  |  val loss=0.8774 acc=0.62
epoch 002  train loss=0.4880 acc=0.81  |  val loss=0.6447 acc=0.76
...
epoch 020  train loss=0.0010 acc=1.00  |  val loss=0.0493 acc=0.97

test accuracy: 0.960

test set report
----------------------------------------
per-class accuracy:
  smiling     19/20  0.95  ###################
  neutral     20/20  1.00  ####################
  sad         19/20  0.95  ###################
  angry       19/20  0.95  ###################
  surprised   19/20  0.95  ###################

confusion matrix  (rows = true, cols = predicted)
              smi  neu  sad  ang  sur
  smiling      19    1    0    0    0
  neutral       0   20    0    0    0
  sad           0    0   19    1    0
  angry         0    0    1   19    0
  surprised     1    0    0    0   19

results saved to results/
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of full passes over the training set |
| `--hidden-size` | 64 | Number of neurons in the hidden layer |
| `--learning-rate` | 0.1 | Step size for gradient descent |
| `--data-dir` | data | Directory containing the train/validation/test splits |
| `--results-dir` | results | Directory where output files are written |
| `--seed` | 42 | Random seed for weight initialisation and shuffling |

---

### 3. Inspect the results

After training, three files are written to `results/`:

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
```
,smiling,neutral,sad,angry,surprised
smiling,19,1,0,0,0
neutral,0,20,0,0,0
...
```

**`training_log.csv`** — one row per epoch:
```
epoch,train_loss,train_acc,val_loss,val_acc
1,0.854800,0.6,0.877400,0.62
2,0.488000,0.81,0.644700,0.76
...
```

---

### 4. Run the tests

```bash
python -m unittest discover
```

All tests should pass. They cover every function in `model.py`, `train.py`, `validate.py`, `metrics.py`, `image_loader.py`, and `generate_faces.py`.

---

## How it works

The network has three layers:

```
input (256)  →  hidden (64, ReLU)  →  output (5, softmax)
```

Each 16×16 image is flattened to a 256-value vector and pixel values are normalised to `[-0.5, 0.5]` before being fed in. Training uses stochastic gradient descent — one weight update per example — with the dataset shuffled each epoch.

| File | Responsibility |
|---|---|
| `model.py` | Maths: activations, loss, forward pass, backpropagation, weight update |
| `generate_faces.py` | Data: draws synthetic faces with randomised features and noise |
| `image_loader.py` | I/O: reads PGM files into pixel arrays |
| `train.py` | Loop: loads data, trains, evaluates, exports results |
| `validate.py` | Analysis: confusion matrix and per-class accuracy report |
| `metrics.py` | Scores: accuracy, precision, recall, F1 and file export functions |

---

## Results history

| Version | Input | Hidden size | Test accuracy |
|---|---|---|---|
| V1 | 8×8 (downsampled) → 64 values | 8 | 0.73 |
| V2 | 16×16 (full resolution) → 256 values | 64 | 0.96 |

Removing the downsampling step was the biggest factor — the 16×16 images are small enough that no size reduction is needed, and downsampling was discarding features the network needed to distinguish emotions, particularly **neutral**.
