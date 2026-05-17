"""Native CIFAR-10 loader.

Downloads the dataset from a stable mirror, extracts it, and unpacks the
pickled batches into NumPy arrays. This avoids depending on
``tf.keras.datasets.cifar10`` which can fail behind restrictive networks.
"""

from __future__ import annotations

import os
import pickle
import shutil
import tarfile
import urllib.request
from typing import Tuple

import numpy as np

from . import config


def _ensure_clean_paths(tar_path: str, extract_path: str) -> None:
    """Remove any half-downloaded or partially extracted artifacts."""
    if os.path.exists(tar_path):
        os.remove(tar_path)
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading CIFAR-10 from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def _extract(tar_path: str, extract_path: str) -> str:
    print("Extracting files ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    return os.path.join(extract_path, config.BATCHES_DIR_NAME)


def _load_batch(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Unpickle a single CIFAR-10 batch into (images, labels)."""
    with open(fpath, "rb") as f:
        raw = pickle.load(f, encoding="bytes")
    decoded = {k.decode("utf8"): v for k, v in raw.items()}

    # CIFAR batches store images as (N, 3072) — reshape to (N, 32, 32, 3).
    data = (
        decoded["data"]
        .reshape(10000, 3, 32, 32)
        .transpose(0, 2, 3, 1)
    )
    labels = np.array(decoded["labels"])
    return data, labels


def load_cifar10(
    force_redownload: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` as float32 in [0, 1]."""

    tar_path = config.TAR_PATH
    extract_path = config.EXTRACT_PATH
    data_dir = os.path.join(extract_path, config.BATCHES_DIR_NAME)

    if force_redownload or not os.path.isdir(data_dir):
        _ensure_clean_paths(tar_path, extract_path)
        _download(config.CIFAR10_URL, tar_path)
        data_dir = _extract(tar_path, extract_path)

    print("Loading data into memory ...")
    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = _load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        x_train.append(data)
        y_train.append(labels)

    x_train = np.vstack(x_train)
    y_train = np.concatenate(y_train).reshape(-1, 1)

    x_test, y_test = _load_batch(os.path.join(data_dir, "test_batch"))
    y_test = y_test.reshape(-1, 1)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"Dataset loaded. Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test
