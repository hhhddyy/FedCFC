from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
ACTIVITY_NAMES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)


def _dataset_complete(dataset_dir: Path) -> bool:
    required_files = [
        dataset_dir / split / name
        for split in ("train", "test")
        for name in (
            f"y_{split}.txt",
            *(f"Inertial Signals/{signal}_{split}.txt" for signal in SIGNALS),
        )
    ]
    return all(path.is_file() for path in required_files)


def _extract_archive(archive: Path, root: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (root / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"unsafe path in UCI archive: {member.filename}")
        zipped.extractall(root)


def download_uci_har(root: str | Path) -> Path:
    """Download and safely extract either official UCI HAR archive layout."""
    root = Path(root).expanduser().resolve()
    dataset_dir = root / "UCI HAR Dataset"
    if _dataset_complete(dataset_dir):
        return dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "uci_har.zip"
    if not zipfile.is_zipfile(archive):
        # Cache only complete downloads.
        partial = archive.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(UCI_HAR_URL) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if not zipfile.is_zipfile(partial):
                raise zipfile.BadZipFile("the UCI download is not a ZIP archive")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    _extract_archive(archive, root)
    # Extract the nested archive when present.
    nested_archive = root / "UCI HAR Dataset.zip"
    if not _dataset_complete(dataset_dir) and nested_archive.is_file():
        _extract_archive(nested_archive, root)
    if not _dataset_complete(dataset_dir):
        raise FileNotFoundError("the UCI archive did not contain a complete 'UCI HAR Dataset'")
    return dataset_dir


def _load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    signal_dir = dataset_dir / split / "Inertial Signals"
    arrays = [
        np.loadtxt(signal_dir / f"{signal}_{split}.txt", dtype=np.float32)
        for signal in SIGNALS
    ]
    features = np.stack(arrays, axis=-1)
    labels = np.loadtxt(
        dataset_dir / split / f"y_{split}.txt", dtype=np.int64
    ) - 1
    return features, labels


def load_uci_har(
    root: str | Path,
    *,
    download: bool = False,
    return_normalization: bool = False,
) -> tuple[Tensor, ...]:
    """Load normalized windows; optionally append the fitted mean and std."""
    root = Path(root).expanduser().resolve()
    dataset_dir = root / "UCI HAR Dataset"
    if download:
        dataset_dir = download_uci_har(root)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"UCI HAR not found at {dataset_dir}; pass --download to fetch it"
        )
    train_x, train_y = _load_split(dataset_dir, "train")
    test_x, test_y = _load_split(dataset_dir, "test")
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    tensors = (
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(test_x),
        torch.from_numpy(test_y),
    )
    if return_normalization:
        return (*tensors, torch.from_numpy(mean.copy()), torch.from_numpy(std.copy()))
    return tensors


def load_uci_har_test(
    root: str | Path, mean: Tensor, std: Tensor, *, download: bool = False
) -> tuple[Tensor, Tensor]:
    """Load only the official test split using saved training statistics."""
    root = Path(root).expanduser().resolve()
    dataset_dir = download_uci_har(root) if download else root / "UCI HAR Dataset"
    features, labels = _load_split(dataset_dir, "test")
    expected_shape = (1, 1, features.shape[-1])
    if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
        raise ValueError("checkpoint normalization shape does not match UCI HAR channels")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization must be finite with strictly positive std")
    features = (features - mean.cpu().numpy()) / std.cpu().numpy()
    return torch.from_numpy(features).float(), torch.from_numpy(labels)


def _validate_partition(labels: Tensor, num_clients: int, min_samples: int) -> None:
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if labels.numel() < num_clients * min_samples:
        raise ValueError("not enough samples to satisfy min_samples for every client")


def iid_partition(
    labels: Tensor,
    num_clients: int,
    *,
    seed: int = 0,
    min_samples: int = 1,
) -> list[Tensor]:
    """Balance client sizes and per-class counts within one sample."""
    _validate_partition(labels, num_clients, min_samples)
    labels_array = labels.detach().cpu().numpy()
    random = np.random.default_rng(seed)
    class_indices = []
    for label in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == label)
        random.shuffle(indices)
        class_indices.append(indices)
    ordered = np.concatenate(class_indices)
    # Distribute class remainders across clients.
    partitions = []
    for client in range(num_clients):
        indices = ordered[client::num_clients].copy()
        random.shuffle(indices)
        partitions.append(torch.from_numpy(indices).long())
    return partitions


def dirichlet_partition(
    labels: Tensor,
    num_clients: int,
    alpha: float,
    *,
    seed: int = 0,
    min_samples: int = 1,
) -> list[Tensor]:
    """Partition indices with class-skewed Dirichlet proportions."""
    _validate_partition(labels, num_clients, min_samples)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    labels_array = labels.detach().cpu().numpy()
    random = np.random.default_rng(seed)
    classes = np.unique(labels_array)
    for _ in range(200):
        partitions: list[list[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            indices = np.flatnonzero(labels_array == label)
            random.shuffle(indices)
            proportions = random.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, chunk in enumerate(np.split(indices, split_points)):
                partitions[client].extend(chunk.tolist())
        if min(map(len, partitions)) >= min_samples:
            for partition in partitions:
                random.shuffle(partition)
            return [torch.tensor(partition, dtype=torch.long) for partition in partitions]
    raise RuntimeError(
        "could not produce non-empty client partitions; increase alpha or reduce clients"
    )


def make_uci_har_loaders(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    *,
    num_clients: int,
    alpha: float,
    batch_size: int,
    seed: int,
    partition: str = "dirichlet",
) -> tuple[list[DataLoader], DataLoader]:
    """Create selected client partitions and the official global test loader."""
    if partition == "iid":
        partitions = iid_partition(
            train_y, num_clients, seed=seed, min_samples=batch_size
        )
    elif partition == "dirichlet":
        partitions = dirichlet_partition(
            train_y, num_clients, alpha, seed=seed, min_samples=batch_size
        )
    else:
        raise ValueError("partition must be 'iid' or 'dirichlet'")
    # Sampling interval: 1/50 seconds.
    train_dt = torch.full(train_x.shape[:2], 1.0 / 50.0)
    test_dt = torch.full(test_x.shape[:2], 1.0 / 50.0)
    generator = torch.Generator().manual_seed(seed)
    client_loaders = [
        DataLoader(
            TensorDataset(train_x[index], train_y[index], train_dt[index]),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        for index in partitions
    ]
    test_loader = DataLoader(
        TensorDataset(test_x, test_y, test_dt),
        batch_size=batch_size * 2,
        shuffle=False,
    )
    return client_loaders, test_loader

