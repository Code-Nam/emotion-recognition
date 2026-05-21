"""Train the simple neural network on the generated face dataset."""

import argparse
import os
import random
from pathlib import Path

from image_loader import load_pgm
from metrics import export_confusion_matrix, export_kfold_summary, export_metrics, export_training_log
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


def stratified_kfold(dataset, k, seed):
    rng = random.Random(seed)
    by_class = {}
    for sample in dataset:
        idx = sample[1].index(1.0)
        by_class.setdefault(idx, []).append(sample)
    for samples in by_class.values():
        rng.shuffle(samples)
    folds = [[] for _ in range(k)]
    for samples in by_class.values():
        for i, sample in enumerate(samples):
            folds[i % k].append(sample)
    return folds


def train_network(train_data, input_size, hidden_size, learning_rate, epochs, seed, val_data=None):
    rng = random.Random(seed)
    network = NeuralNetwork([input_size, hidden_size, len(LABELS)],
                            learning_rate=learning_rate,
                            seed=seed)
    epoch_log = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_data)
        for inputs, targets in train_data:
            wg, bg = network.backward(inputs, targets)
            network.update(wg, bg)
        train_loss, train_acc = evaluate(network, train_data)
        entry = {"epoch": epoch, "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 4)}
        if val_data is not None:
            val_loss, val_acc = evaluate(network, val_data)
            entry["val_loss"] = round(val_loss, 6)
            entry["val_acc"] = round(val_acc, 4)
        epoch_log.append(entry)
    return network, epoch_log


def main():
    parser = argparse.ArgumentParser(description="Train the emotion recognition network.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cv_data = load_split(args.data_dir, "train") + load_split(args.data_dir, "validation")
    test_data = load_split(args.data_dir, "test")

    if not cv_data:
        raise SystemExit(
            "No training data found. Generate it first with:\n"
            "  python src/generate_faces.py --count 100 --out data --seed 42"
        )

    input_size = len(cv_data[0][0])
    print(f"input size:  {input_size}")
    print(f"hidden size: {args.hidden_size}")
    print(f"cv pool / test: {len(cv_data)} / {len(test_data)}")
    print(f"folds: {args.folds}")
    print()

    folds = stratified_kfold(cv_data, args.folds, args.seed)
    fold_results = []
    combined_matrix = [[0] * len(LABELS) for _ in range(len(LABELS))]
    all_fold_logs = []

    for i in range(args.folds):
        val_data = folds[i]
        train_data = [s for j, fold in enumerate(folds) if j != i for s in fold]

        network, epoch_log = train_network(train_data, input_size, args.hidden_size,
                                           args.learning_rate, args.epochs, args.seed + i,
                                           val_data=val_data)

        _, val_acc = evaluate(network, val_data)
        matrix = confusion_matrix(network, val_data)

        for r in range(len(LABELS)):
            for c in range(len(LABELS)):
                combined_matrix[r][c] += matrix[r][c]

        fold_results.append(round(val_acc, 4))
        all_fold_logs.append(epoch_log)
        print(f"fold {i + 1}/{args.folds}  val acc={val_acc:.3f}"
              f"  (train={len(train_data)}, val={len(val_data)})")

    mean_acc = sum(fold_results) / len(fold_results)
    std_acc = (sum((x - mean_acc) ** 2 for x in fold_results) / len(fold_results)) ** 0.5

    averaged_log = []
    for epoch_idx in range(args.epochs):
        averaged_log.append({
            "epoch": epoch_idx + 1,
            "train_loss": round(sum(f[epoch_idx]["train_loss"] for f in all_fold_logs) / args.folds, 6),
            "train_acc": round(sum(f[epoch_idx]["train_acc"] for f in all_fold_logs) / args.folds, 4),
            "val_loss": round(sum(f[epoch_idx]["val_loss"] for f in all_fold_logs) / args.folds, 6),
            "val_acc": round(sum(f[epoch_idx]["val_acc"] for f in all_fold_logs) / args.folds, 4),
        })

    print(f"\ncross-validation accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print_report(combined_matrix, title="aggregated CV report")

    print("\ntraining final model on all CV data...")
    final_network, _ = train_network(cv_data, input_size, args.hidden_size,
                                     args.learning_rate, args.epochs, args.seed)
    _, test_acc = evaluate(final_network, test_data)
    print(f"test accuracy: {test_acc:.3f}")

    test_matrix = confusion_matrix(final_network, test_data)
    print_report(test_matrix, title="test set report")

    os.makedirs(args.results_dir, exist_ok=True)
    export_confusion_matrix(test_matrix, args.results_dir)
    export_metrics(test_matrix, args.results_dir)
    export_kfold_summary(fold_results, mean_acc, std_acc, args.results_dir)
    export_training_log(averaged_log, args.results_dir)
    final_network.save(os.path.join(args.results_dir, "model.json"))
    print(f"\nresults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
