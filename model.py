"""
Autoencoders and GANs on Fashion-MNIST in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_pixels
import gzip
import os
import tempfile
import urllib.request
import numpy as np
import torch


def load_fashion_pixels(n_train: int = 5000, n_test: int = 1000) -> dict[str, torch.Tensor]:
    """Download and parse Fashion-MNIST as flattened float32 tensors scaled to [0, 1].

    Files are cached in the system temp directory with a 'fashion_' prefix.
    """
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz",
    }

    temp_dir = tempfile.gettempdir()
    local_paths = {}

    # Download each file once if not already cached
    for key, filename in files.items():
        dest_path = os.path.join(temp_dir, f"fashion_{filename}")
        if not os.path.exists(dest_path):
            url = base_url + filename
            urllib.request.urlretrieve(url, dest_path)
        local_paths[key] = dest_path

    # Parse raw bytes using np.frombuffer with IDX offsets
    with gzip.open(local_paths["train_img"], "rb") as f:
        X_train_raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

    with gzip.open(local_paths["train_lbl"], "rb") as f:
        y_train_raw = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(local_paths["test_img"], "rb") as f:
        X_test_raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

    with gzip.open(local_paths["test_lbl"], "rb") as f:
        y_test_raw = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # Slice requested counts, scale pixel values to [0.0, 1.0], and convert to torch tensors
    X_train = torch.from_numpy(X_train_raw[:n_train].copy()).to(torch.float32) / 255.0
    y_train = torch.from_numpy(y_train_raw[:n_train].copy()).to(torch.int64)

    X_test = torch.from_numpy(X_test_raw[:n_test].copy()).to(torch.float32) / 255.0
    y_test = torch.from_numpy(y_test_raw[:n_test].copy()).to(torch.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 2 - pca_reconstruction_error
def pca_reconstruction_error(X, n_components):
    # TODO: SVD on centered X; reconstruct from the top components; per-entry MSE
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_components].T 
    recon = Xc @ V @ V.T 
    return float(((Xc - recon) ** 2).mean())

