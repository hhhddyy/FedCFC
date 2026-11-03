from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .data import dirichlet_partition, iid_partition


CLASS_NAMES = [str(digit) for digit in range(10)]


def mnist_preprocessing(sequence_mode: str) -> dict:
    if sequence_mode not in ("rows", "pixels"):
        raise ValueError("sequence_mode must be 'rows' or 'pixels'")
    return {
        "dataset": "mnist", "sequence_mode": sequence_mode,
        "sequence_length": 28 if sequence_mode == "rows" else 784,
        "input_size": 28 if sequence_mode == "rows" else 1,
        "pixel_scale": 255.0, "timespan": 1.0, "scan_order": "row-major",
    }


class SequentialMNIST(Dataset):
    def __init__(self, images: Tensor, labels: Tensor, *, sequence_mode: str = "rows") -> None:
        self.preprocessing = mnist_preprocessing(sequence_mode)
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != torch.uint8:
            raise ValueError("images must be uint8 tensors with shape [samples, 28, 28]")
        if labels.ndim != 1 or len(labels) != len(images) or labels.dtype != torch.long:
            raise ValueError("labels must be int64 tensors with one label per image")
        if len(labels) == 0 or not bool(((labels >= 0) & (labels < 10)).all()):
            raise ValueError("MNIST requires nonempty labels in [0, 9]")
        self.images = images
        self.labels = labels
        self.timespans = torch.ones(self.preprocessing["sequence_length"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        inputs = self.images[index].to(torch.float32).div(255.0).reshape(
            self.preprocessing["sequence_length"], self.preprocessing["input_size"]
        )
        return inputs, self.labels[index], self.timespans


def load_mnist(
    root: str | Path, *, train: bool, download: bool = False, sequence_mode: str = "rows",
) -> SequentialMNIST:
    mnist_preprocessing(sequence_mode)
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise ImportError("MNIST requires torchvision; install a version matching your PyTorch build") from error
    dataset = MNIST(root=str(root), train=train, download=download)
    return SequentialMNIST(dataset.data, dataset.targets, sequence_mode=sequence_mode)


def make_mnist_loaders(
    train: SequentialMNIST, test: SequentialMNIST, *, num_clients: int,
    alpha: float, batch_size: int, seed: int, partition: str,
) -> tuple[list[DataLoader], DataLoader, list[Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if train.preprocessing != test.preprocessing:
        raise ValueError("train and test sequence settings must match")
    if partition == "iid":
        indices = iid_partition(train.labels, num_clients, seed=seed, min_samples=batch_size)
    elif partition == "dirichlet":
        indices = dirichlet_partition(train.labels, num_clients, alpha, seed=seed, min_samples=batch_size)
    else:
        raise ValueError("partition must be 'iid' or 'dirichlet'")
    generator = torch.Generator().manual_seed(seed)
    clients = [DataLoader(Subset(train, index.tolist()), batch_size=batch_size,
                          shuffle=True, generator=generator) for index in indices]
    return clients, DataLoader(test, batch_size=batch_size * 2), [train.labels[index] for index in indices]
