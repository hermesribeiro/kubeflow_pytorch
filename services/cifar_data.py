import torch
import torchvision
import torchvision.transforms as transforms


class CifarData:
    def __init__(self, root="./data", split="train", download=True):
        self.root = root
        self.train = True if split == "train" else False
        self.download = download

    @staticmethod
    def _transform():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return transform

    def _get_dataset(self):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=self._transform(),
        )

        return dataset

    def dataloader(self, batch_size=1, num_workers=2, shuffle=True):
        dataloader = torch.utils.data.DataLoader(
            self._get_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return dataloader
