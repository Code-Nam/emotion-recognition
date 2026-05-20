import copy
import unittest

from src.model import (
    LABELS, LABEL_TO_INDEX, INDEX_TO_LABEL,
    one_hot, relu, softmax, cross_entropy,
    NeuralNetwork,
)


class TestLabels(unittest.TestCase):
    def test_labels_has_five_emotions(self):
        self.assertEqual(len(LABELS), 5)

    def test_label_to_index_covers_all_labels(self):
        for label in LABELS:
            self.assertIn(label, LABEL_TO_INDEX)

    def test_index_to_label_is_reverse_of_label_to_index(self):
        for label, index in LABEL_TO_INDEX.items():
            self.assertEqual(INDEX_TO_LABEL[index], label)


class TestOneHot(unittest.TestCase):
    def test_correct_position_is_one(self):
        vector = one_hot("sad")
        self.assertEqual(vector[LABEL_TO_INDEX["sad"]], 1.0)

    def test_all_other_positions_are_zero(self):
        vector = one_hot("sad")
        for i, value in enumerate(vector):
            if i != LABEL_TO_INDEX["sad"]:
                self.assertEqual(value, 0.0)

    def test_length_matches_number_of_labels(self):
        self.assertEqual(len(one_hot("smiling")), len(LABELS))

    def test_each_label_produces_unique_vector(self):
        vectors = [one_hot(label) for label in LABELS]
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                if i != j:
                    self.assertNotEqual(v1, v2)


class TestRelu(unittest.TestCase):
    def test_positive_value_passes_through(self):
        self.assertEqual(relu(3.5), 3.5)

    def test_zero_returns_zero(self):
        self.assertEqual(relu(0.0), 0.0)

    def test_negative_value_returns_zero(self):
        self.assertEqual(relu(-2.0), 0.0)


class TestSoftmax(unittest.TestCase):
    def test_output_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_largest_input_gets_largest_probability(self):
        probs = softmax([1.0, 5.0, 2.0])
        self.assertEqual(probs.index(max(probs)), 1)

    def test_empty_input_returns_empty(self):
        self.assertEqual(softmax([]), [])

    def test_all_probabilities_are_positive(self):
        for p in softmax([1.0, -1.0, 0.0]):
            self.assertGreater(p, 0.0)


class TestCrossEntropy(unittest.TestCase):
    def test_perfect_prediction_gives_near_zero_loss(self):
        target = [0.0, 1.0, 0.0]
        predicted = [0.0001, 0.9998, 0.0001]
        self.assertLess(cross_entropy(predicted, target), 0.01)

    def test_wrong_prediction_gives_high_loss(self):
        target = [0.0, 1.0, 0.0]
        predicted = [0.49, 0.02, 0.49]
        self.assertGreater(cross_entropy(predicted, target), 1.0)

    def test_loss_is_always_positive(self):
        target = [0.0, 0.0, 1.0]
        predicted = [0.3, 0.3, 0.4]
        self.assertGreater(cross_entropy(predicted, target), 0.0)


class TestNeuralNetworkInit(unittest.TestCase):
    def test_number_of_weight_matrices(self):
        net = NeuralNetwork([4, 6, 5])
        self.assertEqual(len(net.weights), 2)

    def test_weight_matrix_shapes(self):
        net = NeuralNetwork([4, 6, 5])
        self.assertEqual(len(net.weights[0]), 6)     # 6 hidden neurons
        self.assertEqual(len(net.weights[0][0]), 4)  # 4 inputs each
        self.assertEqual(len(net.weights[1]), 5)     # 5 output neurons
        self.assertEqual(len(net.weights[1][0]), 6)  # 6 inputs each

    def test_biases_are_zero_at_init(self):
        net = NeuralNetwork([4, 6, 5])
        for bias_vector in net.biases:
            for value in bias_vector:
                self.assertEqual(value, 0.0)

    def test_same_seed_gives_same_weights(self):
        net1 = NeuralNetwork([4, 6, 5], seed=7)
        net2 = NeuralNetwork([4, 6, 5], seed=7)
        self.assertEqual(net1.weights, net2.weights)

    def test_different_seeds_give_different_weights(self):
        net1 = NeuralNetwork([4, 6, 5], seed=1)
        net2 = NeuralNetwork([4, 6, 5], seed=2)
        self.assertNotEqual(net1.weights, net2.weights)


class TestForward(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork([4, 6, 5], seed=42)
        self.inputs = [0.2, 0.4, 0.6, 0.8]

    def test_activations_count(self):
        activations, _ = self.net.forward(self.inputs)
        # input + hidden + output = 3
        self.assertEqual(len(activations), 3)

    def test_pre_activations_count(self):
        _, pre_activations = self.net.forward(self.inputs)
        # one per layer (no pre-activation for raw input)
        self.assertEqual(len(pre_activations), 2)

    def test_output_length(self):
        activations, _ = self.net.forward(self.inputs)
        self.assertEqual(len(activations[-1]), 5)

    def test_output_sums_to_one(self):
        activations, _ = self.net.forward(self.inputs)
        self.assertAlmostEqual(sum(activations[-1]), 1.0)

    def test_output_probabilities_are_positive(self):
        activations, _ = self.net.forward(self.inputs)
        for p in activations[-1]:
            self.assertGreater(p, 0.0)

    def test_hidden_layer_has_no_negative_values(self):
        activations, _ = self.net.forward(self.inputs)
        for value in activations[1]:
            self.assertGreaterEqual(value, 0.0)


class TestBackward(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork([4, 6, 5], seed=42)
        self.inputs = [0.2, 0.4, 0.6, 0.8]
        self.targets = one_hot("sad")

    def test_weight_gradient_shapes(self):
        wg, _ = self.net.backward(self.inputs, self.targets)
        self.assertEqual(len(wg[0]), 6)
        self.assertEqual(len(wg[0][0]), 4)
        self.assertEqual(len(wg[1]), 5)
        self.assertEqual(len(wg[1][0]), 6)

    def test_bias_gradient_shapes(self):
        _, bg = self.net.backward(self.inputs, self.targets)
        self.assertEqual(len(bg[0]), 6)
        self.assertEqual(len(bg[1]), 5)

    def test_gradients_are_not_all_zero(self):
        wg, _ = self.net.backward(self.inputs, self.targets)
        all_zero = all(wg[i][j][k] == 0.0
                       for i in range(len(wg))
                       for j in range(len(wg[i]))
                       for k in range(len(wg[i][j])))
        self.assertFalse(all_zero)


class TestUpdate(unittest.TestCase):
    def test_weights_change_after_update(self):
        net = NeuralNetwork([4, 6, 5], seed=42)
        inputs = [0.2, 0.4, 0.6, 0.8]
        targets = one_hot("sad")
        before = copy.deepcopy(net.weights)
        wg, bg = net.backward(inputs, targets)
        net.update(wg, bg)
        self.assertNotEqual(before, net.weights)

    def test_biases_change_after_update(self):
        net = NeuralNetwork([4, 6, 5], seed=42)
        inputs = [0.2, 0.4, 0.6, 0.8]
        targets = one_hot("sad")
        wg, bg = net.backward(inputs, targets)
        net.update(wg, bg)
        all_zero = all(net.biases[i][j] == 0.0
                       for i in range(len(net.biases))
                       for j in range(len(net.biases[i])))
        self.assertFalse(all_zero)


class TestTrain(unittest.TestCase):
    def test_loss_decreases_over_training(self):
        net = NeuralNetwork([4, 6, 5], learning_rate=0.1, seed=42)
        dataset = [
            ([0.1, 0.9, 0.2, 0.4], one_hot("smiling")),
            ([0.8, 0.1, 0.7, 0.3], one_hot("sad")),
            ([0.5, 0.5, 0.1, 0.9], one_hot("angry")),
        ]
        activations_before, _ = net.forward(dataset[0][0])
        loss_before = cross_entropy(activations_before[-1], dataset[0][1])

        net.train(dataset, epochs=200)

        activations_after, _ = net.forward(dataset[0][0])
        loss_after = cross_entropy(activations_after[-1], dataset[0][1])

        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
