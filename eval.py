import argparse

from services.cifar_data import CifarData
from services.cifar_trainer import CifarTrainer
from services.neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description="Train cifar neural network")

parser.add_argument(
    "-u",
    "--bucket",
    required=True,
    type=str,
    help="Bucket to save the model",
)

parser.add_argument(
    "-p",
    "--path",
    required=True,
    type=str,
    help="Path to save the model",
)

args = parser.parse_args()


def main():
    neural_network = NeuralNetwork()
    model = CifarTrainer(neural_network=neural_network)
    model.load(bucket=args.bucket, path=args.path)
    dataloader = CifarData(split="test").dataloader()
    model.eval(dataloader=dataloader)


if __name__ == "__main__":
    main()
