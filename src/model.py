import json
import math
import random


LABELS = ["smiling", "neutral", "sad", "angry", "surprised"]

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def one_hot(label):
    index = LABEL_TO_INDEX[label]
    return [1.0 if i == index else 0.0 for i in range(len(LABELS))]


def relu(value):
    if value > 0:
        return value
    return 0.0


def softmax(raw_scores):
    if not raw_scores:
        return []

    # Subtract the max before exponentiating to prevent overflow.
    # The result is mathematically identical because the constant cancels
    # in the numerator and denominator, but keeps values in a safe range.
    max_raw_score = max(raw_scores)
    exp_values = []
    for raw_score in raw_scores:
        exp_values.append(math.exp(raw_score - max_raw_score))

    total_exp_value = sum(exp_values)
    probabilities = []
    for exp_value in exp_values:
        probabilities.append(exp_value / total_exp_value)

    return probabilities


def cross_entropy(predicted_probabilities, target_probabilities, epsilon=1e-12):
    # Clamp predicted probability to epsilon before taking log to avoid log(0),
    # which would produce -inf and propagate NaN through training.
    total_loss = 0.0
    for predicted_probability, target_probability in zip(predicted_probabilities, target_probabilities):
        total_loss -= target_probability * math.log(max(predicted_probability, epsilon))
    return total_loss


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, seed=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        rng = random.Random(seed)

        self.weights = []
        self.biases = []

        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Scale weights by 1/√n (Xavier uniform). This keeps the variance of
            # pre-activations roughly constant across layers regardless of fan-in.
            scale = 1.0 / math.sqrt(input_size)
            matrix = [
                [rng.uniform(-scale, scale) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            self.weights.append(matrix)
            # Biases start at zero; the weights already break symmetry.
            self.biases.append([0.0] * output_size)

    def forward(self, inputs):
        activations = [inputs]
        pre_activations = []

        current = inputs
        for i, (weight_matrix, bias) in enumerate(zip(self.weights, self.biases)):
            z = [
                sum(w * x for w, x in zip(row, current)) + b
                for row, b in zip(weight_matrix, bias)
            ]
            pre_activations.append(z)

            is_last_layer = (i == len(self.weights) - 1)
            if is_last_layer:
                # Output layer uses softmax to produce a probability distribution.
                current = softmax(z)
            else:
                current = [relu(v) for v in z]

            activations.append(current)

        return activations, pre_activations

    def backward(self, inputs, targets):
        activations, pre_activations = self.forward(inputs)

        weight_gradients = [[[0.0] * len(row) for row in m] for m in self.weights]
        bias_gradients = [[0.0] * len(b) for b in self.biases]

        # For a softmax output with cross-entropy loss the gradient simplifies to
        # (predicted − target). The chain rule terms for softmax and log cancel,
        # so no separate softmax derivative is needed here.
        delta = [p - t for p, t in zip(activations[-1], targets)]

        for i in reversed(range(len(self.weights))):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weight_gradients[i][j][k] = delta[j] * activations[i][k]
            bias_gradients[i] = list(delta)

            if i > 0:
                # Propagate the error back through the ReLU by zeroing out the
                # gradient for any neuron that was inactive on the forward pass
                # (pre-activation ≤ 0 → derivative = 0, the "step gate").
                new_delta = []
                for k in range(len(self.weights[i][0])):
                    error = sum(self.weights[i][j][k] * delta[j] for j in range(len(delta)))
                    relu_deriv = 1.0 if pre_activations[i - 1][k] > 0 else 0.0
                    new_delta.append(error * relu_deriv)
                delta = new_delta

        return weight_gradients, bias_gradients

    def update(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.learning_rate * weight_gradients[i][j][k]
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.learning_rate * bias_gradients[i][j]

    def predict(self, inputs):
        activations, _ = self.forward(inputs)
        probabilities = activations[-1]
        best_index = probabilities.index(max(probabilities))
        return best_index, INDEX_TO_LABEL[best_index]

    def loss(self, inputs, targets):
        activations, _ = self.forward(inputs)
        return cross_entropy(activations[-1], targets)

    def save(self, path):
        payload = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            payload = json.load(f)
        # __new__ skips __init__ so we can set attributes directly from the
        # saved payload without triggering weight re-initialisation.
        network = cls.__new__(cls)
        network.layer_sizes = payload["layer_sizes"]
        network.learning_rate = payload["learning_rate"]
        network.weights = payload["weights"]
        network.biases = payload["biases"]
        return network

    def train(self, dataset, epochs=100):
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for inputs, targets in dataset:
                weight_gradients, bias_gradients = self.backward(inputs, targets)
                self.update(weight_gradients, bias_gradients)
                predictions = self.forward(inputs)[0][-1]
                total_loss += cross_entropy(predictions, targets)

            avg_loss = total_loss / len(dataset)
            print(f"epoch {epoch:03d}  loss={avg_loss:.4f}")
