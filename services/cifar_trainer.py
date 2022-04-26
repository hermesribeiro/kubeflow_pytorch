import os

from pathlib import Path

import boto3
import torch
import torch.nn as nn
import torch.optim as optim


class CifarTrainer:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.s3 = boto3.client('s3')

    def train(
        self,
        dataloader,
        num_epochs,
        learning_rate,
        momentum,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.neural_network.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )

        for epoch in range(num_epochs):

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):

                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.neural_network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    mean_loss = running_loss / 2000
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {mean_loss:.3f}")
                    running_loss = 0.0

    def save(self, bucket, path):
        print('Saving model.')
        tmp_path = 'cifar_net.pth'
        torch.save(self.neural_network.state_dict(), tmp_path)
        self.s3.upload_file(tmp_path, bucket, path)

    def load(self, bucket, path):
        print("Loading saved model")
        tmp_path = 'cifar_net.pth'
        self.s3.download_file(bucket, path, tmp_path)
        self.neural_network.load_state_dict(torch.load(tmp_path))

    def eval(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                
                outputs = self.neural_network(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
