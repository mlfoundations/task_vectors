import os
import shutil
from typing import Tuple

import torch
import torchvision.datasets as datasets
import re
from torchvision.datasets import EuroSAT as EuroSATPyTorch
from os import listdir
import numpy as np


def pretify_classname(classname):
    l = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    l = [i.lower() for i in l]
    out = " ".join(l)
    if out.endswith("al"):
        return out + " area"
    return out


class EuroSATBase:
    def __init__(
        self,
        preprocess,
        test_split,
        location="~/data",
        batch_size=32,
        num_workers=16,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),  # train, val, test split
        random_seed: int = 1350,
    ):
        self.ratios = ratios
        self.random_seed = random_seed

        # Data loading code
        self.train_path = os.path.join(location, "EuroSAT_splits", "train")
        self.val_path = os.path.join(location, "EuroSAT_splits", "val")
        self.test_path = os.path.join(location, "EuroSAT_splits", "test")
        if not os.path.exists(self.train_path):
            # download the data to eurosat/2750
            _ = EuroSATPyTorch(location, download=True)
            data_source_dir = os.path.join(location, "eurosat", "2750")
            target_classes = listdir(data_source_dir)
            for target in target_classes:
                os.makedirs(os.path.join(self.train_path, target))
                os.makedirs(os.path.join(self.val_path, target))
                os.makedirs(os.path.join(self.test_path, target))
                self.copy_data_from_source_to_splits(
                    source_dir=os.path.join(data_source_dir, target), class_name=target
                )
            # clean up
            os.remove(os.path.join(location, "EuroSAT.zip"))
            shutil.rmtree(os.path.join(location, "eurosat"))

        self.train_dataset = datasets.ImageFolder(self.train_path, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        testdir = self.val_path if test_split == "val" else self.test_path
        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            "annual crop": "annual crop land",
            "forest": "forest",
            "herbaceous vegetation": "brushland or shrubland",
            "highway": "highway or road",
            "industrial area": "industrial buildings or commercial buildings",
            "pasture": "pasture land",
            "permanent crop": "permanent crop land",
            "residential area": "residential buildings or homes or apartments",
            "river": "river",
            "sea lake": "lake or sea",
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]

    def copy_data_from_source_to_splits(self, source_dir: str, class_name: str) -> None:
        val_ratio = self.ratios[1]
        test_ratio = self.ratios[2]
        sample_collection = os.listdir(source_dir)
        np.random.seed(self.random_seed)

        np.random.shuffle(
            sample_collection,
        )
        train_files, val_files, test_files = np.split(
            np.array(sample_collection),
            [
                int(len(sample_collection) * (1 - (val_ratio + test_ratio))),
                int(len(sample_collection) * (1 - val_ratio)),
            ],
        )
        train_files = [os.path.join(source_dir, name) for name in train_files.tolist()]
        val_files = [os.path.join(source_dir, name) for name in val_files.tolist()]
        test_files = [os.path.join(source_dir, name) for name in test_files.tolist()]

        for name in train_files:
            shutil.copy(name, os.path.join(self.train_path, class_name))

        for name in val_files:
            shutil.copy(name, os.path.join(self.val_path, class_name))

        for name in test_files:
            shutil.copy(name, os.path.join(self.test_path, class_name))


class EuroSAT(EuroSATBase):
    def __init__(self, preprocess, location="~/datasets", batch_size=32, num_workers=16):
        super().__init__(preprocess, "test", location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(self, preprocess, location="~/datasets", batch_size=32, num_workers=16):
        super().__init__(preprocess, "val", location, batch_size, num_workers)
