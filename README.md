---
title: DiffLocks Studio
emoji: 💇‍♀️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: other
---

# 💇‍♀️ DiffLocks Studio

**High-Fidelity 3D Hair Generation from Single Images**

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sVM0G5sI4xqaZvjmBjsFYDwZwFRQBnmC#scrollTo=8DfIC_lPUu4a)
[![Original Repo](https://img.shields.io/badge/Original-Meshcapade-blue)](https://github.com/Meshcapade/difflocks)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/arqdariogomez/DiffLocks-Studio)

## 🌟 About this Project

This repository is a modification of **DiffLocks**, the original project by **Meshcapade GmbH** and the **Max Planck Institute for Intelligent Systems**.

### 🎯 Our Mission: Democratization
The primary goal of **DiffLocks Studio** is to democratize access to this cutting-edge technology. We have optimized the original code to run on consumer-grade hardware (such as an 8GB GPU) and accessible platforms like **Pinokio**, removing the entry barrier of requiring high-performance workstations.

### 💡 Personal Motivation
This project was born from a desire to learn, gain experience in deploying complex diffusion models, and contribute to the community by providing powerful tools for creators worldwide.

---

## ⚖️ Attribution and License

> [!IMPORTANT]
> **Original Credit**: All credit for the model architecture, data, and research belongs to **Meshcapade GmbH** and its authors (Rosu et al., CVPR 2025).

- **License**: This project directly inherits the license from [Meshcapade/difflocks](https://github.com/Meshcapade/difflocks). It is for **Non-Commercial Scientific Research Use Only**.
- **Citation**: If you use this work in your research, please cite the original paper:
  ```bibtex
  @inproceedings{difflocks2025,
    title = {DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models},
    author = {Rosu, Radu Alexandru and Wu, Keyu and Feng, Yao and Zheng, Youyi and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025}
  }
  ```

---

## 🚀 Getting Started (Choose your path)

We recommend using the platforms in this order of simplicity:

1. **Hugging Face Space (ZeroGPU)**: [Click here](https://huggingface.co/spaces/arqdariogomez/DiffLocks-Studio). The easiest way. Just upload a photo and wait. (Requires ZeroGPU grant or waiting in queue).
2. **Google Colab / Kaggle**: Best for free GPU access. Use the badges at the top.
3. **Pinokio (Local)**: One-click installer for Windows/Mac/Linux. Download [Pinokio](https://pinokio.computer) and search for "DiffLocks Studio".
4. **Docker / Manual**: For advanced users and developers.

---

## 🧠 Model Setup (Checkpoints)

Due to licensing, we do not include the model weights in the repository. You have two ways to set them up:

### Option A: Automatic (Recommended)
Set your `HF_TOKEN` environment variable or enter it in the app settings. The app will automatically download the required assets from our [Hugging Face Dataset](https://huggingface.co/datasets/arqdariogomez/difflocks-assets-hybrid).

### Option B: Manual Download
1. Download the checkpoints from the [original Meshcapade repo](https://github.com/Meshcapade/difflocks).
2. Place them in the following structure:
   ```
   checkpoints/
   ├── scalp_diffusion.pth
   └── strand_vae/
       └── strand_codec.pt
   ```

---

## ⚙️ Technical Improvements

- **Platform-Aware Precision**: Automatic VRAM detection (toggles `float16` if < 12GB).
- **Natten Fallback**: Native PyTorch implementation for Neighborhood Attention.
- **Self-Update**: Integrated "Check for Updates" button in the UI.
- **Blender Integration**: Native `.blend` export with modern hair curves.

---

## 🐳 Docker Deployment

To run with Docker and NVIDIA GPU support:

1. Install [Docker](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Clone the repo.
3. Run:
   ```bash
   docker-compose up --build
   ```
4. Open `http://localhost:7860`.

---

## 🎨 Exporting to Blender

1. **Generate**: After inference, download the results (ZIP).
2. **Import**: 
   - Use the `.blend` file directly.
   - Or use the `npz_blender_importer.py` script to load `.npz` data.
   - Or install the `blender_addon` folder.
