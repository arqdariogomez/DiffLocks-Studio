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

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/rubndarogmezhurtado/difflocks-github-launcher-minimal)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sVM0G5sI4xqaZvjmBjsFYDwZwFRQBnmC#scrollTo=8DfIC_lPUu4a)
[![Original Repo](https://img.shields.io/badge/Original-Meshcapade-blue)](https://github.com/Meshcapade/difflocks)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/arqdariogomez/DiffLocks-Studio)

## 🌟 About this Project

This repository is a modification of **DiffLocks**, the original project by **Meshcapade GmbH** and the **Max Planck Institute for Intelligent Systems**.

### 🎯 The Mission: Democratization
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

## 🚀 Getting Started (Recommended Order)

1. **Kaggle (Best Experience)**: [Kaggle Notebook](https://www.kaggle.com/code/rubndarogmezhurtado/difflocks-github-launcher-minimal). This is the most tested platform, providing generous GPU hours and the most stable environment.
2. **Pinokio (Local)**: One-click installer for Windows/Mac/Linux.
   - Ensure you have the latest version of [Pinokio](https://pinokio.computer).
   - Go to the **Discover** section.
   - Click on **Add from URL**.
   - Paste this repository URL and hit **Enter**.
   - **Tip**: When the console says the app is ready at `0.0.0.0`, use [http://localhost:7860/](http://localhost:7860/) in your browser for the best compatibility.
3. **Google Colab**: [Colab Notebook](https://colab.research.google.com/drive/1sVM0G5sI4xqaZvjmBjsFYDwZwFRQBnmC#scrollTo=8DfIC_lPUu4a). (Currently being refined to ensure 100% stability).
4. **Hugging Face Space**: [HF Space](https://huggingface.co/spaces/arqdariogomez/DiffLocks-Studio). A great "try-it-now" option. It becomes the #1 choice if a ZeroGPU grant is active; otherwise, you may need to wait in a queue.

---

## 🧠 Model Setup (Checkpoints)

Due to licensing restrictions, model weights are not included. Here is how to set them up:

### 🔐 Using "Secrets" (Kaggle / Colab)
For cloud platforms, the best and most private way is to use the **Secrets** add-on. Store your `HF_TOKEN` there. This ensures your credentials remain private while allowing the app to securely download the necessary assets for the first run.

### 📥 Manual Download (Local / Docker)
If you are running locally or want full control:
1. Download the checkpoints from the [original Meshcapade repository](https://github.com/Meshcapade/difflocks).
2. Create a `checkpoints` folder in the project root.
3. Place the files in this exact structure:
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
1. Install Docker and NVIDIA Container Toolkit.
2. Run: `docker-compose up --build`
3. Open `http://localhost:7860`.
