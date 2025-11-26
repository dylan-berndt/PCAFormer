import os
from typing import Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from pathlib import Path
import kagglehub

import re


checkDirectories = {"aryankaushik005/custom-dataset": ["real_images", "fake_images"],
                    "dimensi0n/imagenet-256": ["abacus", "admiral", "agama"]}


def download_dataset(name=None):
    name = name if name is not None else "aryankaushik005/custom-dataset"

    # 1) Download the dataset
    path = kagglehub.dataset_download(name)
    print("KaggleHub dataset path:", path)

    # 2) Find the directory that contains both real_images and fake_images
    root = Path(path)
    # KaggleHub may place files under a nested 'content' or similar folder; search for the class dirs
    candidates = []
    for p in [root, *root.rglob("*")]:
        if p.is_dir():
            broke = False
            for directory in checkDirectories[name]:
                if not (p / directory).is_dir():
                    broke = True
                    break

            if not broke:
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


class ImageNetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, target):
        return self.mapping[target]


def _has_dir(path: str) -> bool:
    return isinstance(path, str) and len(path) > 0 and os.path.isdir(path)


def get_datasets(cfg, device="cpu", transform=None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[int, str]]:
    image_size = int(cfg.model.imageSize)
    augment = bool(getattr(cfg.dataset, "augment", True))
    if transform is None:
        train_tfms, val_tfms = _build_transforms(image_size, augment)
    else:
        train_tfms, val_tfms = transform, transform

    train_dir = getattr(cfg.dataset, "trainDir", "")
    val_dir = getattr(cfg.dataset, "valDir", "")
    data_dir = getattr(cfg.dataset, "dataDir", "")

    # So glad that NLP had to butt its way into this
    folder_names = [os.path.basename(f.path) for f in os.scandir(data_dir) if f.is_dir()]
    folder_names = sorted(folder_names)
    synsets = open("LOC_synset_mapping.txt", "r").read().split("\n")
    synsets = [s[10:].split(", ") for s in synsets]
    synsets = [[re.sub(r'[^\w\s]', ' ', n).lower().replace("  ", " ") for n in s] for s in synsets]
    mapping = {}
    for i, name in enumerate(folder_names):
        found = False
        for s, synset in enumerate(synsets):
            if name.lower().replace("_", " ") in synset:
                found = True
                mapping[i] = s
                break
        if not found:
            print(name, i)
    target_transform = ImageNetTransform(mapping)

    if _has_dir(train_dir) and _has_dir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms, target_transform=target_transform)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tfms, target_transform=target_transform)
        classes = {i: c for i, c in enumerate(train_ds.classes)}
        return train_ds, val_ds, classes

    if not _has_dir(data_dir):
        raise FileNotFoundError("dataset.dataDir is not a valid directory and train/val dirs not provided")

    base_ds = datasets.ImageFolder(data_dir, transform=train_tfms, target_transform=target_transform)

    split = float(getattr(cfg.dataset, "trainValSplit", 0.8))
    seed = int(getattr(cfg.dataset, "seed", 42))

    n = len(base_ds)
    n_train = int(n * split)
    n_val = n - n_train

    g = torch.Generator(device=device).manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(base_ds, [n_train, n_val], generator=g)

    classes = {i: c for i, c in enumerate(base_ds.classes)}
    return train_ds, val_ds, classes


def get_dataloaders(cfg, device="cpu", transform=None):
    train_ds, val_ds, classes = get_datasets(cfg, device, transform)

    batch_size = int(cfg.batchSize)
    num_workers = int(getattr(cfg.dataset, "numWorkers", 4))
    # pin_memory = torch.cuda.is_available()
    pin_memory = False

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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device=device)
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "classes": classes,
    }

