"""Train the simple neural network on the generated face dataset."""

import argparse
import os
import random
from pathlib import Path

from image_loader import load_pgm
from metrics import export_confusion_matrix, export_metrics, export_training_log
from model import LABELS, NeuralNetwork, one_hot
from validate import confusion_matrix, print_report


def load_split(data_root, split_name):
    split_path = Path(data_root) / split_name
    dataset = []

    for label_name in LABELS:
        label_path = split_path / label_name
        if not label_path.exists():
            continue
        for image_path in sorted(label_path.glob("*.pgm")):
            image = load_pgm(str(image_path))
            input_vector = [(pixel / 255.0) - 0.5 for row in image for pixel in row]
            target_vector = one_hot(label_name)
            dataset.append((input_vector, target_vector))

    return dataset


def evaluate(network, dataset):
    if not dataset:
        return 0.0, 0.0

    total_loss = 0.0
    correct = 0
    for inputs, targets in dataset:
        total_loss += network.loss(inputs, targets)
        predicted_index, _ = network.predict(inputs)
        if targets[predicted_index] == 1.0:
            correct += 1

    return total_loss / len(dataset), correct / len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Train the emotion recognition network.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    train_data = load_split(args.data_dir, "train")
    val_data = load_split(args.data_dir, "validation")
    test_data = load_split(args.data_dir, "test")

    if not train_data:
        raise SystemExit(
            "No training data found. Generate it first with:\n"
            "  python src/generate_faces.py --count 100 --out data --seed 42"
        )

    input_size = len(train_data[0][0])
    network = NeuralNetwork([input_size, args.hidden_size, len(LABELS)],
                            learning_rate=args.learning_rate,
                            seed=args.seed)

    print(f"input size:  {input_size}")
    print(f"hidden size: {args.hidden_size}")
    print(f"train / val / test: {len(train_data)} / {len(val_data)} / {len(test_data)}")
    print()

    rng = random.Random(args.seed)
    training_log = []

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_data)
        for inputs, targets in train_data:
            wg, bg = network.backward(inputs, targets)
            network.update(wg, bg)

        train_loss, train_acc = evaluate(network, train_data)
        val_loss, val_acc = evaluate(network, val_data)
        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
        })
        print(f"epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.2f}"
              f"  |  val loss={val_loss:.4f} acc={val_acc:.2f}")

    _, test_acc = evaluate(network, test_data)
    print(f"\ntest accuracy: {test_acc:.3f}")

    matrix = confusion_matrix(network, test_data)
    print_report(matrix, title="test set report")

    os.makedirs(args.results_dir, exist_ok=True)
    export_training_log(training_log, args.results_dir)
    export_confusion_matrix(matrix, args.results_dir)
    export_metrics(matrix, args.results_dir)
    print(f"\nresults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
