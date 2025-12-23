# DiffLocks Studio (Universal V2)

AI-powered 3D hair generation from a single image. This project provides a unified interface for Pinokio, Docker, Kaggle, Colab, and local environments.

---

## 🚀 Getting Started (Choose your platform)

| Platform | Best For | Setup Speed | GPU Required | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Pinokio** | One-click local install | ⚡ Fast | Yes (NVIDIA) | [Install via Pinokio](https://pinokio.computer) |
| **Docker** | Local power users | ⚡⚡ Very Fast | Yes (NVIDIA) | [See Instructions Below](#-docker-deployment-easiest-for-local) |
| **Kaggle** | Free GPU (30h/week) | 🐢 Slow | No (Cloud) | [Open in Kaggle]([https://www.kaggle.com/code/arqdariogomez/difflocks-v6d](https://www.kaggle.com/code/rubndarogmezhurtado/difflocks-github-launcher-minimal)) |
| **Colab** | Free/Paid GPU | 🐢 Slow | No (Cloud) | [Open in Colab](https://colab.research.google.com/github/arqdariogomez/DiffLocks-Studio/blob/main/Difflocks_v6d.ipynb) |

---

## 🧠 Model Setup (Checkpoints)

Due to the **Non-Commercial Scientific Research Use Only** license, checkpoints must be downloaded manually from the official source.

### 📥 Step 1: Manual Download
1. Register/Login at [difflocks.is.tue.mpg.de](https://difflocks.is.tue.mpg.de/).
2. Download [**difflocks_checkpoints.zip**](https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=difflocks_checkpoints.zip).
3. **Pinokio**: Place the `.zip` in `C:\pinokio\api\DiffLocks-Studio`.
4. **Docker**: Place the `.zip` in your project folder.
5. **Kaggle/Colab**: Upload to your drive or use the provided download cells.

### 🔐 Step 2: Using "Secrets" (Kaggle / Colab)
Even with manual setup, using the **Secrets** add-on is the best practice for privacy and license compliance. It allows the app to securely download required assets (like Face Landmarkers) without exposing your keys.

#### 1. Generate your Hugging Face Token
- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Click **New token**.
- Name it (e.g., "DiffLocks") and set the type to **Read**.
- Copy the generated token (starts with `hf_...`).

#### 2. Add to your Platform
- **Kaggle**: Go to **Add-ons** -> **Secrets**. Add a new secret with Label `HF_TOKEN` and paste your token in the Value field. Ensure the "Attached" checkbox is checked.
- **Colab**: Click the **Key icon** (Secrets) in the left sidebar. Add a new secret named `HF_TOKEN` and paste your token. Enable "Notebook access".

---

## ⚙️ Technical Improvements

- **Platform-Aware Precision**: Automatic VRAM detection (toggles `float16` if < 12GB).
- **Natten Fallback**: Native PyTorch implementation for Neighborhood Attention.
- **Self-Update**: Integrated "Check for Updates" button in the UI.
- **Blender Integration**: Native `.blend` export with modern hair curves.

---

## 🐳 Docker Deployment (Easiest for Local)

This is the recommended method for running locally with full control.

### 1. Get the Code
- Click the green **Code** button at the top of this page.
- Select **Download ZIP** and extract it to a folder of your choice.
- *Alternative*: `git clone https://github.com/arqdariogomez/DiffLocks-Studio.git`

### 2. Get the Checkpoints (Mandatory)
Due to licensing, you must download the weights from the official source:
1. Register/Login at [difflocks.is.tue.mpg.de](https://difflocks.is.tue.mpg.de/).
2. Go to the **Download** section.
3. Download [**difflocks_checkpoints.zip**](https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=difflocks_checkpoints.zip).
4. **Important**: Place the `difflocks_checkpoints.zip` file directly into the project folder (where `Run_Docker.bat` is).

### 3. Launch
1. Ensure [Docker Desktop](https://www.docker.com/) is running.
2. Double-click **`Run_Docker.bat`**.
3. The script will automatically unzip the checkpoints and start the server.
4. Open [**http://localhost:7860**](http://localhost:7860) in your browser.

### 4. GPU Support (NVIDIA)
To enable GPU acceleration on Windows:
1. Ensure you have the latest [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx).
2. Run `wsl --update` in PowerShell to ensure WSL2 is up to date.
3. In Docker Desktop Settings, ensure **"Use the WSL 2 based engine"** is enabled.
4. The `Run_Docker.bat` script will automatically verify GPU access on startup.
