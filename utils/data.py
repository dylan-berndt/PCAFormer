import os
from typing import Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from pathlib import Path
import kagglehub


def download_dataset():
    # 1) Download the dataset
    path = kagglehub.dataset_download("aryankaushik005/custom-dataset")
    print("KaggleHub dataset path:", path)

    # 2) Find the directory that contains both real_images and fake_images
    root = Path(path)
    # KaggleHub may place files under a nested 'content' or similar folder; search for the class dirs
    candidates = []
    for p in [root, *root.rglob("*")]:
        if p.is_dir():
            real = p / "real_images"
            fake = p / "fake_images"
            if real.is_dir() and fake.is_dir():
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate a directory containing both 'real_images' and 'fake_images'. Inspect the downloaded path and adjust.")

    # Choose the deepest match to avoid picking a higher-level folder accidentally
    candidates = sorted(candidates, key=lambda x: len(str(x)))
    data_dir = str(candidates[-1])
    return data_dir


def _build_transforms(image_size: int, augment: bool) -> Tuple[T.Compose, T.Compose]:
    train_tfms = [
        T.Resize((image_size, image_size)),
    ]
    if augment:
        train_tfms.extend([
            T.RandomHorizontalFlip(p=0.5),
        ])
    train_tfms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return T.Compose(train_tfms), val_tfms


def _has_dir(path: str) -> bool:
    return isinstance(path, str) and len(path) > 0 and os.path.isdir(path)


def get_datasets(cfg, device="cpu") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[int, str]]:
    image_size = int(cfg.model.imageSize)
    augment = bool(getattr(cfg.dataset, "augment", True))
    train_tfms, val_tfms = _build_transforms(image_size, augment)

    train_dir = getattr(cfg.dataset, "trainDir", "")
    val_dir = getattr(cfg.dataset, "valDir", "")
    data_dir = getattr(cfg.dataset, "dataDir", "")

    if _has_dir(train_dir) and _has_dir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
        classes = {i: c for i, c in enumerate(train_ds.classes)}
        return train_ds, val_ds, classes

    if not _has_dir(data_dir):
        raise FileNotFoundError("dataset.dataDir is not a valid directory and train/val dirs not provided")

    base_ds = datasets.ImageFolder(data_dir, transform=train_tfms)

    split = float(getattr(cfg.dataset, "trainValSplit", 0.8))
    seed = int(getattr(cfg.dataset, "seed", 42))

    n = len(base_ds)
    n_train = int(n * split)
    n_val = n - n_train

    g = torch.Generator(device=device).manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(base_ds, [n_train, n_val], generator=g)

    classes = {i: c for i, c in enumerate(base_ds.classes)}
    return train_ds, val_ds, classes


def get_dataloaders(cfg, device="cpu"):
    train_ds, val_ds, classes = get_datasets(cfg, device)

    batch_size = int(cfg.batchSize)
    num_workers = int(getattr(cfg.dataset, "numWorkers", 4))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device=device)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device=device)
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "classes": classes,
    }

