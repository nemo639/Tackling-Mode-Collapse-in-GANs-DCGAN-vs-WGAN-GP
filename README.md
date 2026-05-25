# Tackling Mode Collapse in GANs — DCGAN vs WGAN-GP

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![DCGAN](https://img.shields.io/badge/DCGAN-Baseline-blue?style=flat-square)
![WGAN-GP](https://img.shields.io/badge/WGAN--GP-Improved-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20x2-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**DCGAN baseline vs WGAN-GP — demonstrating how Wasserstein loss + gradient penalty eliminates mode collapse**

[📝 Medium Blog](#) · [💼 LinkedIn Post](#) · [🤗 Live Demo](#)

</div>

---

## 📌 Overview

This project implements two GAN architectures to study and solve **mode collapse** — the failure where a generator produces only a small variety of outputs regardless of input noise.

```
Noise z  →  Generator  →  Fake Image
                               ↓
Real Image  →  Discriminator/Critic  →  Real / Fake
```

**DCGAN** (baseline) uses Binary Cross-Entropy and often suffers mode collapse. **WGAN-GP** replaces the discriminator with a Critic and uses the Wasserstein distance + gradient penalty for dramatically more stable training and diverse outputs.

---

## ✨ Features

- ✅ DCGAN — Generator (transposed convolutions + BatchNorm) + Discriminator (strided convolutions + LeakyReLU)
- ✅ WGAN-GP — Generator + Critic with Wasserstein loss + gradient penalty
- ✅ Side-by-side output comparison (DCGAN vs WGAN-GP diversity)
- ✅ Generator loss and Discriminator/Critic loss curves
- ✅ FID and Inception Score (optional)
- ✅ Mixed precision training for Kaggle T4×2
- ✅ Streamlit / Gradio demo app

---

## 🗂️ Repository Structure

```
📦 dcgan-wgangp-mode-collapse/
├── 📁 models/
│   ├── dcgan.py                   # DCGAN Generator + Discriminator
│   └── wgan_gp.py                 # WGAN-GP Generator + Critic
├── 📁 training/
│   ├── train_dcgan.py             # DCGAN training loop (BCE loss)
│   └── train_wgangp.py            # WGAN-GP training loop (Wasserstein + GP)
├── 📁 evaluation/
│   └── metrics.py                 # FID (optional), IS (optional), diversity comparison
├── 📁 visualization/
│   └── visualize.py               # Side-by-side generated samples
├── 📁 app/
│   └── app.py                     # Streamlit / Gradio demo
├── 📓 dcgan_wgangp.ipynb          # Full pipeline notebook
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧠 Model Architecture

### DCGAN — Baseline

**Generator**
- Input: 100-dim noise vector z
- Transposed convolutions with BatchNorm and ReLU
- Output: 64×64 RGB image (tanh activation)

**Discriminator**
- Strided convolutions with LeakyReLU
- Binary real/fake classification (sigmoid output)
- Loss: Binary Cross-Entropy

### WGAN-GP — Improved

**Generator** — same architecture as DCGAN

**Critic** (replaces Discriminator)
- No sigmoid output — outputs a real-valued score
- Loss: **Wasserstein distance** — measures how different real and fake distributions are
- **Gradient Penalty** — enforces the 1-Lipschitz constraint (replaces weight clipping)

### Why WGAN-GP solves mode collapse
Vanilla GAN discriminators saturate when they get too good — gradients vanish and the generator stops learning, collapsing to a few modes. The Wasserstein metric provides meaningful, non-saturating gradients even when distributions don't overlap, keeping the generator learning across the full output space.

---

## 📊 Datasets

- [Pokemon Sprites — Kaggle](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)
- [Anime Faces 64×64 — Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=0.0002, betas=(0.5, 0.999) |
| Batch Size | 64 (adjust for GPU memory) |
| Strategy | Train DCGAN first → then WGAN-GP → compare |
| Precision | Mixed (torch.cuda.amp) |
| Checkpointing | Every 5–10 epochs |

---

## 📈 Evaluation

| Metric | Description |
|---|---|
| Generator Loss | Should decrease and stabilize |
| Discriminator/Critic Loss | Should maintain balance with generator |
| Output Diversity | Visual comparison of 5–10 generated samples per model |
| FID (optional) | Fréchet Inception Distance — lower = more realistic |
| IS (optional) | Inception Score — higher = better quality + diversity |

---

## 🖥️ Demo App

Upload or generate a noise vector → compare DCGAN and WGAN-GP outputs side by side in real time.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/dcgan-wgangp-mode-collapse.git
cd dcgan-wgangp-mode-collapse
pip install -r requirements.txt
python training/train_dcgan.py
python training/train_wgangp.py
streamlit run app/app.py
```

---

## ✅ Tasks Completed

- [x] DCGAN Generator + Discriminator (BCE loss)
- [x] WGAN-GP Generator + Critic (Wasserstein loss + gradient penalty)
- [x] Generator and Critic/Discriminator loss curves
- [x] Side-by-side output diversity comparison
- [x] FID and IS (optional)
- [x] Streamlit / Gradio app

---

## 🔗 Links

| Resource | Link |
|---|---|
| Medium Blog | [Read on Medium](#) |
| LinkedIn Post | [View on LinkedIn](#) |
| Live Demo | [Streamlit / Gradio](#) |
| Pokemon Dataset | [Kaggle](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites) |
| Anime Faces Dataset | [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) |

---

*Muhammad Naeem |FAST-NUCES | Generative AI *
