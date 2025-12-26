
"""
DiffLocks Platform Configuration Module
Autodetects: Kaggle, Colab, HuggingFace Spaces, Pinokio, Local.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

PlatformType = Literal['kaggle', 'colab', 'huggingface', 'pinokio', 'docker', 'local']

@dataclass
class Config:
    platform: PlatformType
    work_dir: Path
    repo_dir: Path
    output_dir: Path
    checkpoints_dir: Path
    configs_dir: Path
    blender_exe: Path
    has_gpu: bool
    vram_gb: float
    needs_share: bool

    @staticmethod
    def detect() -> 'Config':
        if Path("/kaggle").exists():
            platform = 'kaggle'
            work_dir = Path("/kaggle/working")
            repo_dir = work_dir / "DiffLocks-Studio"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
        elif 'COLAB_GPU' in os.environ or Path("/content").exists():
            platform = 'colab'
            work_dir = Path("/content")
            repo_dir = work_dir / "DiffLocks-Studio"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
        elif 'SPACE_ID' in os.environ: 
            platform = 'huggingface'
            # In HF Spaces, we want to ensure we are in /app if possible
            if Path("/app").exists():
                work_dir = Path("/app")
            else:
                work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("/tmp/blender/blender") 
            needs_share = False
            
            # Persistent storage check for HF Spaces
            # Check /data (Persistent Storage) or /home/user/app/data
            potential_data_dirs = [Path("/data"), Path("./data"), Path("/home/user/app/data")]
            for d in potential_data_dirs:
                if d.exists() and os.access(str(d), os.W_OK):
                    # We found a writable data dir
                    break
        elif Path("/app").exists() and os.environ.get("DOCKER_CONTAINER", "false") == "true":
            platform = 'docker'
            work_dir = Path("/app")
            repo_dir = Path("/app")
            blender_exe = Path("/app/blender/blender") 
            needs_share = False
        elif 'PINOKIO_HOME' in os.environ:
            platform = 'pinokio'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = work_dir / "blender" / ("blender.exe" if sys.platform == 'win32' else "blender")
            needs_share = False
        else:
            platform = 'local'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("blender/blender") 
            needs_share = False

        # Set standard directories relative to repo_dir
        checkpoints_dir = repo_dir / "checkpoints"
        configs_dir = repo_dir / "configs"

        # HF Spaces persistent storage override
        if platform == 'huggingface' and Path("/data").exists() and os.access("/data", os.W_OK):
            checkpoints_dir = Path("/data/checkpoints")

        # Platform-specific checkpoint overrides
        if platform == 'kaggle':
            # Check for connected datasets first
            # Kaggle datasets are usually in /kaggle/input/<dataset-name>
            potential_ckpt_dirs = [
                Path("/kaggle/input/difflocks-checkpoints"),
                Path("/kaggle/input/difflocks/checkpoints"),
                repo_dir / "checkpoints"
            ]
            for d in potential_ckpt_dirs:
                if d.exists() and (d / "difflocks_diffusion").exists():
                    checkpoints_dir = d
                    break
        elif platform == 'colab':
            # Check for Google Drive mount
            gdrive_ckpt = Path("/content/drive/MyDrive/DiffLocks/checkpoints")
            if gdrive_ckpt.exists():
                checkpoints_dir = gdrive_ckpt

        # Special case for Docker if they are mapped differently
        if platform == 'docker':
            if not checkpoints_dir.exists() and Path("/app/checkpoints").exists():
                checkpoints_dir = Path("/app/checkpoints")
            if not configs_dir.exists() and Path("/app/configs").exists():
                configs_dir = Path("/app/configs")

        has_gpu = False
        vram_gb = 0.0
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except: pass

        if platform == 'docker' or platform == 'pinokio' or platform == 'local':
            output_dir = repo_dir / "studio_outputs"
        else:
            output_dir = work_dir / "outputs"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        return Config(platform, work_dir, repo_dir, output_dir, checkpoints_dir, configs_dir, blender_exe, has_gpu, vram_gb, needs_share)

cfg = Config.detect()
