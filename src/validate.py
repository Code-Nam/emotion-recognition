from model import LABELS


def confusion_matrix(network, dataset):
    n = len(LABELS)
    matrix = [[0] * n for _ in range(n)]
    for inputs, targets in dataset:
        true_index = targets.index(1.0)
        predicted_index, _ = network.predict(inputs)
        matrix[true_index][predicted_index] += 1
    return matrix


def per_class_accuracy(matrix):
    results = []
    for i, label in enumerate(LABELS):
        total = sum(matrix[i])
        correct = matrix[i][i]
        accuracy = correct / total if total > 0 else 0.0
        results.append((label, correct, total, accuracy))
    return results


def print_report(matrix, title="validation report"):
    print(f"\n{title}")
    print("-" * 40)

    print("per-class accuracy:")
    for label, correct, total, accuracy in per_class_accuracy(matrix):
        bar = "#" * int(accuracy * 20)
        print(f"  {label:10s}  {correct:2d}/{total}  {accuracy:.2f}  {bar}")

    print()
    print("confusion matrix  (rows = true, cols = predicted)")
    col_headers = "  ".join(f"{label[:3]:>3s}" for label in LABELS)
    print(f"  {'':10s}  {col_headers}")
    for i, label in enumerate(LABELS):
        row_values = "  ".join(f"{matrix[i][j]:3d}" for j in range(len(LABELS)))
        print(f"  {label:10s}  {row_values}")
