import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (
    accuracy, precision, recall, f1_score,
    export_metrics, export_confusion_matrix, export_training_log,
)
from model import LABELS


def perfect_matrix(count=10):
    n = len(LABELS)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = count
    return matrix


def half_correct_matrix():
    n = len(LABELS)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 5
        matrix[i][(i + 1) % n] = 5
    return matrix


class TestAccuracy(unittest.TestCase):
    def test_perfect_matrix_gives_one(self):
        self.assertAlmostEqual(accuracy(perfect_matrix()), 1.0)

    def test_half_correct_gives_half(self):
        self.assertAlmostEqual(accuracy(half_correct_matrix()), 0.5)

    def test_empty_matrix_gives_zero(self):
        n = len(LABELS)
        self.assertAlmostEqual(accuracy([[0] * n for _ in range(n)]), 0.0)


class TestPrecision(unittest.TestCase):
    def test_perfect_matrix_gives_all_ones(self):
        for p in precision(perfect_matrix()):
            self.assertAlmostEqual(p, 1.0)

    def test_length_matches_number_of_labels(self):
        self.assertEqual(len(precision(perfect_matrix())), len(LABELS))

    def test_no_predicted_positives_gives_zero(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        for p in precision(matrix):
            self.assertEqual(p, 0.0)

    def test_all_predictions_wrong_gives_zero(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][(i + 1) % n] = 10
        for p in precision(matrix):
            self.assertAlmostEqual(p, 0.0)


class TestRecall(unittest.TestCase):
    def test_perfect_matrix_gives_all_ones(self):
        for r in recall(perfect_matrix()):
            self.assertAlmostEqual(r, 1.0)

    def test_length_matches_number_of_labels(self):
        self.assertEqual(len(recall(perfect_matrix())), len(LABELS))

    def test_no_actual_positives_gives_zero(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        for r in recall(matrix):
            self.assertEqual(r, 0.0)

    def test_known_values(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        matrix[0][0] = 8
        matrix[0][1] = 2
        r = recall(matrix)
        self.assertAlmostEqual(r[0], 0.8)


class TestF1Score(unittest.TestCase):
    def test_perfect_matrix_gives_all_ones(self):
        for f in f1_score(perfect_matrix()):
            self.assertAlmostEqual(f, 1.0)

    def test_length_matches_number_of_labels(self):
        self.assertEqual(len(f1_score(perfect_matrix())), len(LABELS))

    def test_zero_precision_and_recall_gives_zero(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        for f in f1_score(matrix):
            self.assertEqual(f, 0.0)

    def test_f1_is_harmonic_mean_of_precision_and_recall(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        matrix[0][0] = 8
        matrix[0][1] = 2
        matrix[1][0] = 4
        matrix[1][1] = 6
        p = precision(matrix)
        r = recall(matrix)
        f = f1_score(matrix)
        expected = 2 * p[0] * r[0] / (p[0] + r[0])
        self.assertAlmostEqual(f[0], expected)


class TestExports(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.matrix = perfect_matrix(count=10)

    def test_export_metrics_creates_json(self):
        path = export_metrics(self.matrix, self.tmpdir)
        self.assertTrue(os.path.exists(path))

    def test_metrics_json_has_required_keys(self):
        export_metrics(self.matrix, self.tmpdir)
        with open(os.path.join(self.tmpdir, "metrics.json")) as f:
            data = json.load(f)
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class"):
            self.assertIn(key, data)

    def test_metrics_json_per_class_has_all_labels(self):
        export_metrics(self.matrix, self.tmpdir)
        with open(os.path.join(self.tmpdir, "metrics.json")) as f:
            data = json.load(f)
        for label in LABELS:
            self.assertIn(label, data["per_class"])

    def test_metrics_json_perfect_accuracy_is_one(self):
        export_metrics(self.matrix, self.tmpdir)
        with open(os.path.join(self.tmpdir, "metrics.json")) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["accuracy"], 1.0)

    def test_export_confusion_matrix_creates_csv(self):
        path = export_confusion_matrix(self.matrix, self.tmpdir)
        self.assertTrue(os.path.exists(path))

    def test_confusion_matrix_csv_has_header_row(self):
        export_confusion_matrix(self.matrix, self.tmpdir)
        with open(os.path.join(self.tmpdir, "confusion_matrix.csv")) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header[1:], LABELS)

    def test_confusion_matrix_csv_diagonal_values(self):
        export_confusion_matrix(self.matrix, self.tmpdir)
        with open(os.path.join(self.tmpdir, "confusion_matrix.csv")) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                self.assertEqual(int(row[i + 1]), 10)

    def test_export_training_log_creates_csv(self):
        log = [{"epoch": 1, "train_loss": 1.5, "train_acc": 0.4,
                "val_loss": 1.6, "val_acc": 0.35}]
        path = export_training_log(log, self.tmpdir)
        self.assertTrue(os.path.exists(path))

    def test_training_log_csv_has_correct_columns(self):
        log = [{"epoch": 1, "train_loss": 1.5, "train_acc": 0.4,
                "val_loss": 1.6, "val_acc": 0.35}]
        export_training_log(log, self.tmpdir)
        with open(os.path.join(self.tmpdir, "training_log.csv")) as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
        self.assertEqual(columns, ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def test_training_log_csv_values_are_correct(self):
        log = [{"epoch": 3, "train_loss": 0.5, "train_acc": 0.9,
                "val_loss": 0.6, "val_acc": 0.85}]
        export_training_log(log, self.tmpdir)
        with open(os.path.join(self.tmpdir, "training_log.csv")) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        self.assertEqual(int(row["epoch"]), 3)
        self.assertAlmostEqual(float(row["train_acc"]), 0.9)


if __name__ == "__main__":
    unittest.main()
