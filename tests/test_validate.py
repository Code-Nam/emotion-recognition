import io
import sys
import unittest
from unittest.mock import patch

from src.model import LABELS, one_hot
from src.validate import confusion_matrix, per_class_accuracy, print_report


class MockNetwork:
    """Predicts a fixed sequence of class indices."""
    def __init__(self, predicted_indices):
        self._predictions = predicted_indices
        self._call = 0

    def predict(self, inputs):
        index = self._predictions[self._call % len(self._predictions)]
        self._call += 1
        return index, LABELS[index]


def perfect_matrix():
    n = len(LABELS)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 10
    return matrix


class TestConfusionMatrix(unittest.TestCase):
    def test_shape_is_n_by_n(self):
        net = MockNetwork([0])
        dataset = [([0.0], one_hot("smiling"))]
        matrix = confusion_matrix(net, dataset)
        self.assertEqual(len(matrix), len(LABELS))
        self.assertEqual(len(matrix[0]), len(LABELS))

    def test_perfect_predictions_fill_diagonal(self):
        predictions = list(range(len(LABELS)))
        dataset = [([0.0], one_hot(label)) for label in LABELS]
        net = MockNetwork(predictions)
        matrix = confusion_matrix(net, dataset)
        for i in range(len(LABELS)):
            self.assertEqual(matrix[i][i], 1)

    def test_wrong_predictions_leave_diagonal_zero(self):
        dataset = [([0.0], one_hot("smiling"))]
        net = MockNetwork([1])  # always predicts neutral
        matrix = confusion_matrix(net, dataset)
        self.assertEqual(matrix[0][0], 0)
        self.assertEqual(matrix[0][1], 1)

    def test_counts_accumulate_correctly(self):
        dataset = [([0.0], one_hot("smiling"))] * 5
        net = MockNetwork([0])  # always correct
        matrix = confusion_matrix(net, dataset)
        self.assertEqual(matrix[0][0], 5)

    def test_all_counts_sum_to_dataset_size(self):
        dataset = [([0.0], one_hot(label)) for label in LABELS for _ in range(4)]
        net = MockNetwork(list(range(len(LABELS))) * 4)
        matrix = confusion_matrix(net, dataset)
        total = sum(matrix[i][j] for i in range(len(LABELS)) for j in range(len(LABELS)))
        self.assertEqual(total, len(dataset))


class TestPerClassAccuracy(unittest.TestCase):
    def test_perfect_matrix_gives_all_ones(self):
        results = per_class_accuracy(perfect_matrix())
        for label, correct, total, accuracy in results:
            self.assertAlmostEqual(accuracy, 1.0)

    def test_returns_one_entry_per_label(self):
        results = per_class_accuracy(perfect_matrix())
        self.assertEqual(len(results), len(LABELS))

    def test_labels_match_order(self):
        results = per_class_accuracy(perfect_matrix())
        for (label, _, _, _), expected in zip(results, LABELS):
            self.assertEqual(label, expected)

    def test_correct_and_total_values(self):
        matrix = perfect_matrix()
        matrix[0][0] = 7
        matrix[0][1] = 3
        results = per_class_accuracy(matrix)
        label, correct, total, accuracy = results[0]
        self.assertEqual(correct, 7)
        self.assertEqual(total, 10)
        self.assertAlmostEqual(accuracy, 0.7)

    def test_zero_total_returns_zero_accuracy(self):
        n = len(LABELS)
        matrix = [[0] * n for _ in range(n)]
        results = per_class_accuracy(matrix)
        for _, _, _, accuracy in results:
            self.assertEqual(accuracy, 0.0)


class TestPrintReport(unittest.TestCase):
    def test_does_not_raise(self):
        print_report(perfect_matrix())

    def test_output_contains_all_labels(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_report(perfect_matrix(), title="test report")
            output = mock_out.getvalue()
        for label in LABELS:
            self.assertIn(label[:3], output)

    def test_output_contains_title(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_report(perfect_matrix(), title="my report")
            output = mock_out.getvalue()
        self.assertIn("my report", output)


if __name__ == "__main__":
    unittest.main()
