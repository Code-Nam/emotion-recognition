import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import evaluate, load_split
from model import NeuralNetwork, one_hot, LABELS


def write_pgm(path, width, height, pixels):
    """Write a P2 ASCII PGM file with the given flat pixel list."""
    with open(path, "w") as f:
        f.write(f"P2\n{width} {height}\n255\n")
        for i in range(height):
            row = pixels[i * width:(i + 1) * width]
            f.write(" ".join(str(p) for p in row) + "\n")


class MockNetwork:
    def __init__(self, predicted_index, fixed_loss=0.5):
        self._index = predicted_index
        self._loss = fixed_loss

    def predict(self, inputs):
        return self._index, LABELS[self._index]

    def loss(self, inputs, targets):
        return self._loss


class TestEvaluate(unittest.TestCase):
    def test_empty_dataset_returns_zeros(self):
        net = MockNetwork(0)
        loss, acc = evaluate(net, [])
        self.assertEqual(loss, 0.0)
        self.assertEqual(acc, 0.0)

    def test_perfect_predictions_give_accuracy_one(self):
        dataset = [([0.0], one_hot("smiling"))] * 5
        net = MockNetwork(0)  # smiling = index 0, always correct
        _, acc = evaluate(net, dataset)
        self.assertAlmostEqual(acc, 1.0)

    def test_all_wrong_predictions_give_accuracy_zero(self):
        dataset = [([0.0], one_hot("smiling"))] * 5
        net = MockNetwork(1)  # always predicts neutral, always wrong
        _, acc = evaluate(net, dataset)
        self.assertAlmostEqual(acc, 0.0)

    def test_half_correct_gives_half_accuracy(self):
        correct = [([0.0], one_hot("smiling"))] * 5
        wrong = [([0.0], one_hot("neutral"))] * 5
        net = MockNetwork(0)  # always predicts smiling
        _, acc = evaluate(net, correct + wrong)
        self.assertAlmostEqual(acc, 0.5)

    def test_loss_is_averaged_over_dataset(self):
        dataset = [([0.0], one_hot("smiling"))] * 4
        net = MockNetwork(0, fixed_loss=0.8)
        avg_loss, _ = evaluate(net, dataset)
        self.assertAlmostEqual(avg_loss, 0.8)


class TestLoadSplit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        label = "smiling"
        label_dir = os.path.join(self.tmpdir, "train", label)
        os.makedirs(label_dir)
        pixels = [200] * 256  # 16x16 uniform image
        write_pgm(os.path.join(label_dir, "smiling_0000.pgm"), 16, 16, pixels)

    def test_returns_one_example_per_file(self):
        dataset = load_split(self.tmpdir, "train")
        self.assertEqual(len(dataset), 1)

    def test_input_vector_length_is_256(self):
        dataset = load_split(self.tmpdir, "train")
        inputs, _ = dataset[0]
        self.assertEqual(len(inputs), 256)

    def test_inputs_are_centered_around_zero(self):
        dataset = load_split(self.tmpdir, "train")
        inputs, _ = dataset[0]
        mean = sum(inputs) / len(inputs)
        self.assertAlmostEqual(mean, (200 / 255.0) - 0.5, places=3)

    def test_target_is_valid_one_hot(self):
        dataset = load_split(self.tmpdir, "train")
        _, targets = dataset[0]
        self.assertEqual(len(targets), len(LABELS))
        self.assertEqual(sum(targets), 1.0)
        self.assertEqual(targets[LABELS.index("smiling")], 1.0)

    def test_missing_split_returns_empty(self):
        dataset = load_split(self.tmpdir, "validation")
        self.assertEqual(dataset, [])


if __name__ == "__main__":
    unittest.main()
