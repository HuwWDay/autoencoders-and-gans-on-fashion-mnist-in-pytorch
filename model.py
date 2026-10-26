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

# Step 6 - TiedAutoencoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedAutoencoder(nn.Module):
    def __init__(self, n_inputs: int = 784, hidden: int = 100, n_codings: int = 30):
        super().__init__()
        # Encoder weights & biases
        self.W1 = nn.Parameter(torch.empty(hidden, n_inputs))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.empty(n_codings, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_codings))

        # Decoder tied biases (reusing W2.T and W1.T)
        self.b3 = nn.Parameter(torch.zeros(hidden))
        self.b4 = nn.Parameter(torch.zeros(n_inputs))

        # Initialize weights matching PyTorch's nn.Linear default
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(x, self.W1, self.b1))
        codings = F.linear(h, self.W2, self.b2)
        return codings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codings = self.encode(x)
        # Transposed weights: (codings @ W2 + b3) -> F.linear(codings, self.W2.t(), self.b3)
        h_recon = F.relu(codings @ self.W2 + self.b3)
        recon = torch.sigmoid(h_recon @ self.W1 + self.b4)
        return recon

# Step 7 - denoising_gain
def denoising_gain(model, X, noise_std=0.3, seed=0):
    # TODO: mse(noisy, X) / mse(model(noisy), X)
    torch.manual_seed(seed)
    noisy = (X + noise_std*torch.randn_like(X)).clamp(0, 1)
    return float(F.mse_loss(noisy, X) / F.mse_loss(model(noisy), X))

# Step 8 - kl_sparsity_loss
import torch
import torch.nn as nn


def kl_sparsity_loss(activations: torch.Tensor, target: float = 0.1) -> torch.Tensor:
    """Compute the KL divergence between target sparsity and average hidden unit activations.

    Formula:
        KL(p_target || p) = sum_j [ target * log(target / p_j) + (1 - target) * log((1 - target) / (1 - p_j)) ]
    where p_j is the batch mean activation of unit j clamped to [1e-6, 1 - 1e-6].
    """
    # Mean activation per hidden unit across the batch dimension
    p = activations.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)

    t = torch.as_tensor(target, dtype=p.dtype, device=p.device)

    loss_per_unit = t * torch.log(t / p) + (1.0 - t) * torch.log((1.0 - t) / (1.0 - p))
    return loss_per_unit.sum()


class SparseAutoencoder(nn.Module):
    def __init__(self, n_inputs: int = 784, hidden: int = 300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, n_inputs),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

# Step 9 - train_sparse
import torch
import torch.nn.functional as F


def train_sparse(
    model: torch.nn.Module,
    X: torch.Tensor,
    weight: float = 0.1,
    target: float = 0.1,
    epochs: int = 5,
    lr: float = 0.005,
    batch_size: int = 64,
    seed: int = 42,
) -> list[float]:
    """Train a SparseAutoencoder using MSE reconstruction loss plus a KL sparsity penalty.

    Returns the list of per-epoch mean total losses.
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
            batch = X[batch_indices].to(device)

            optimizer.zero_grad()

            codings = model.encode(batch)
            recon = model.decoder(codings)

            recon_loss = F.mse_loss(recon, batch)
            sparsity_loss = kl_sparsity_loss(codings, target=target)
            total_loss = recon_loss + weight * sparsity_loss

            total_loss.backward()
            optimizer.step()

            batch_losses.append(total_loss.item())

        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses


def mean_activation(model: torch.nn.Module, X: torch.Tensor) -> float:
    """Compute the mean activation of model.encode(X) in eval mode without gradients."""
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    X_dev = X.to(device)

    with torch.no_grad():
        codings = model.encode(X_dev)
        mean_val = codings.mean().item()

    model.train(was_training)
    return float(mean_val)

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

# Step 15 - Generator
class Generator(nn.Module):
    def __init__(self, n_codings=30, hidden=100, n_outputs=784):
        super().__init__()
        self.n_codings = n_codings 
        self.net = nn.Sequential(nn.Linear(n_codings, hidden), nn.ReLU(), nn.Linear(hidden, n_outputs), nn.Sigmoid())
         
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, n_inputs=784, hidden=100):
        super().__init__()
        # TODO
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        # TODO: logits (batch,)
        return self.net(x).squeeze(-1)

