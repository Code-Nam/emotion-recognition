import csv
import json
import os

from model import LABELS


def accuracy(matrix):
    total = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[i])))
    if total == 0:
        return 0.0
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return correct / total


def precision(matrix):
    scores = []
    for j in range(len(LABELS)):
        predicted_positive = sum(matrix[i][j] for i in range(len(LABELS)))
        scores.append(matrix[j][j] / predicted_positive if predicted_positive > 0 else 0.0)
    return scores


def recall(matrix):
    scores = []
    for i in range(len(LABELS)):
        actual_positive = sum(matrix[i])
        scores.append(matrix[i][i] / actual_positive if actual_positive > 0 else 0.0)
    return scores


def f1_score(matrix):
    p = precision(matrix)
    r = recall(matrix)
    scores = []
    for pi, ri in zip(p, r):
        scores.append(2 * pi * ri / (pi + ri) if (pi + ri) > 0 else 0.0)
    return scores


def _macro(scores):
    return sum(scores) / len(scores) if scores else 0.0


def export_metrics(matrix, output_dir):
    p = precision(matrix)
    r = recall(matrix)
    f = f1_score(matrix)

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(p[i], 4),
            "recall": round(r[i], 4),
            "f1": round(f[i], 4),
            "support": sum(matrix[i]),
        }

    payload = {
        "accuracy": round(accuracy(matrix), 4),
        "macro_precision": round(_macro(p), 4),
        "macro_recall": round(_macro(r), 4),
        "macro_f1": round(_macro(f), 4),
        "per_class": per_class,
    }

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def export_confusion_matrix(matrix, output_dir):
    path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + LABELS)
        for i, label in enumerate(LABELS):
            writer.writerow([label] + matrix[i])
    return path


def export_training_log(log, output_dir):
    path = os.path.join(output_dir, "training_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log)
    return path
