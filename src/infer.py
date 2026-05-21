"""Run inference on a single PGM image using a trained model."""

import argparse

from image_loader import load_pgm
from model import NeuralNetwork, INDEX_TO_LABEL


def main():
    parser = argparse.ArgumentParser(description="Predict the emotion in a PGM face image.")
    parser.add_argument("--image", required=True, help="Path to a PGM image file")
    parser.add_argument("--model", default="results/model.json", help="Path to saved model JSON")
    args = parser.parse_args()

    network = NeuralNetwork.load(args.model)

    image = load_pgm(args.image)
    input_vector = [(pixel / 255.0) - 0.5 for row in image for pixel in row]

    activations, _ = network.forward(input_vector)
    probabilities = activations[-1]

    ranked = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

    predicted_label = INDEX_TO_LABEL[ranked[0][0]]
    print(f"prediction: {predicted_label}")
    print()
    print("confidences:")
    for idx, prob in ranked:
        bar = "#" * int(prob * 20)
        print(f"  {INDEX_TO_LABEL[idx]:10s}  {prob:.3f}  {bar}")


if __name__ == "__main__":
    main()
