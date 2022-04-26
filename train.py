import argparse

from services.cifar_data import CifarData
from services.cifar_trainer import CifarTrainer
from services.neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description="Train cifar neural network")

parser.add_argument(
    "-n",
    "--num_epochs",
    required=True,
    type=int,
    help="Number of training epochs",
)

parser.add_argument(
    "-b",
    "--batch_size",
    required=True,
    type=int,
    help="Size of training batch",
)

parser.add_argument(
    "-l",
    "--learning_rate",
    required=True,
    type=float,
    help="Learning rate",
)

parser.add_argument(
    "-m",
    "--momentum",
    required=True,
    type=float,
    help="Momentum",
)

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
    dataloader = CifarData().dataloader(batch_size=args.batch_size)
    neural_network = NeuralNetwork()
    model = CifarTrainer(neural_network=neural_network)
    model.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    model.save(bucket=args.bucket, path=args.path)


if __name__ == "__main__":
    main()
