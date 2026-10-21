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

# Step 3 - LinearAutoencoder
class LinearAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, n_codings=30):
        super().__init__()
        # TODO
        self.encoder = nn.Linear(n_inputs, n_codings)
        self.decoder = nn.Linear(n_codings, n_inputs)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Step 4 - train_autoencoder
import torch
import torch.nn.functional as F


def train_autoencoder(
    model: torch.nn.Module,
    X: torch.Tensor,
    epochs: int = 5,
    lr: float = 0.005,
    batch_size: int = 64,
    seed: int = 42,
    noise_std: float = 0.0,
) -> list[float]:
    """Train an autoencoder to reconstruct X with optional Gaussian input noise.

    Returns the list of mean MSE reconstruction losses for each epoch.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = X.size(0)
    device = next(model.parameters()).device
    epoch_losses = []

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        batch_losses = []

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            target = X[batch_indices].to(device)

            if noise_std > 0.0:
                noise = torch.randn_like(target) * noise_std
                inp = target + noise
            else:
                inp = target

            optimizer.zero_grad()
            recon = model(inp)
            loss = F.mse_loss(recon, target)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses


def reconstruction_error(model: torch.nn.Module, X: torch.Tensor) -> float:
    """Compute the MSE between model(X) and X in eval mode without gradients."""
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    X_dev = X.to(device)

    with torch.no_grad():
        recon = model(X_dev)
        error = F.mse_loss(recon, X_dev).item()

    model.train(was_training)
    return float(error)

# Step 5 - StackedAutoencoder
class StackedAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, hidden=100, n_codings=30):
        super().__init__()
        # TODO: encoder Sequential, decoder Sequential ending in Sigmoid
        self.encoder = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, n_codings))
        self.decoder = nn.Sequential(nn.Linear(n_codings, hidden), nn.ReLU(), nn.Linear(hidden, n_inputs), nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Step 11 - VAE
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, n_inputs: int = 784, hidden: int = 100, n_codings: int = 10):
        super().__init__()
        self.n_codings = n_codings
        self.hidden = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, n_codings)
        self.logvar = nn.Linear(hidden, n_codings)
        self.decoder = nn.Sequential(
            nn.Linear(n_codings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Step 12 - vae_loss
import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    return 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1).mean()


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the total VAE loss: reconstruction loss + KL divergence.

    Reconstruction loss is the Binary Cross Entropy summed over feature dimensions
    per sample, then averaged over the batch.
    """
    batch_size = x.size(0)
    # Sum across all pixels per sample, then average across the batch
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / batch_size
    kl = kl_divergence(mu, logvar)
    return bce + kl

