import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets import DTD as DTDdataSet


class DTD:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
    ):
        # Data loading code
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")
        try:
            self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        except FileNotFoundError:
            self.train_dataset = DTDdataSet(
                root=location, split="train", transform=preprocess, download=True
            )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        try:
            self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        except FileNotFoundError:
            self.test_dataset = DTDdataSet(
                root=location, split="test", transform=preprocess, download=True
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
