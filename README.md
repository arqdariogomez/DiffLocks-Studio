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

## ⚙️ Technical Improvements and Changes

We have introduced several key optimizations compared to the original repository:
- **Platform-Aware Precision**: Automatic VRAM detection to toggle between `float16` and `float32`.
- **Pinokio Support**: One-click installation and execution scripts for Windows/Linux/Mac.
- **Natten Fallback**: Native PyTorch implementation for Neighborhood Attention, allowing usage without compiling external libraries.
- **Robust Model Search**: Recursive checkpoint detection system to simplify initial setup.

---

## 🚀 Usage Guide

### 🖥️ Web Interface
Simply run the project in Pinokio or via `python app.py`. Upload a face image and press "Generate".

### 🔌 Developer API

#### Python
```python
from inference.img2hair import DiffLocksInference

# Initialize (automatically detects GPU and precision)
infer = DiffLocksInference(
    path_ckpt_strandcodec="checkpoints/strand_vae/strand_codec.pt",
    path_config_difflocks="configs/config_scalp_texture_conditional.json",
    path_ckpt_difflocks="checkpoints/scalp_diffusion.pth"
)

# Generate from image
for step, msg in infer.run_from_image("input.jpg"):
    print(f"[{step}] {msg}")
```

#### Curl (Gradio API)
```bash
curl -X POST https://your-gradio-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["data:image/jpeg;base64,..."]}'
```

---

## 🎨 Exporting to Blender

DiffLocks Studio includes tools to bring your 3D hair into Blender:
1. **Generate**: After inference, download the `.npz` file.
2. **Import**: Use the `npz_blender_importer.py` script inside Blender to load the strands as native curves.
3. **Addon**: You can also install the `blender_addon` folder as a standard Blender addon.

---

## 🛠️ Local Installation

1. Clone the repo.
2. Create venv: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Download models: `python download_checkpoints.py`.
5. Start: `python app.py`.
