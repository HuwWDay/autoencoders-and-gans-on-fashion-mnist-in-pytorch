# Autoencoders and GANs on Fashion-MNIST in PyTorch

Chapter 18 of Hands-On Machine Learning on real images. Compress Fashion-MNIST with a linear autoencoder and check it against PCA, then a stacked autoencoder that beats it; tie weights, denoise and add a sparsity penalty. Then generate: a variational autoencoder with the ELBO loss, and a GAN trained with the two-player step.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_fashion_pixels
- [x] **2.** pca_reconstruction_error
- [x] **3.** LinearAutoencoder
- [x] **4.** train_autoencoder
- [x] **5.** StackedAutoencoder
- [x] **6.** TiedAutoencoder
- [x] **7.** denoising_gain
- [x] **8.** kl_sparsity_loss
- [x] **9.** train_sparse
- [x] **10.** codings_classifier
- [x] **11.** VAE
- [x] **12.** vae_loss
- [x] **13.** train_vae
- [x] **14.** generate_images
- [x] **15.** Generator

---

Built on Deep-ML.
