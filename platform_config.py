
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
        elif any(k in os.environ for k in ['SPACE_ID', 'SPACE_REPO_NAME', 'HF_SPACE_ID', 'HF_SPACE_ID']): 
            platform = 'huggingface'
            # In HF Spaces, we want to ensure we are in /app if possible
            if Path("/app").exists():
                work_dir = Path("/app")
            elif Path("/home/user/app").exists():
                work_dir = Path("/home/user/app")
            else:
                work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("/tmp/blender/blender") 
            needs_share = False
            
            # Persistent storage check for HF Spaces
            # Check /data (Persistent Storage) or /home/user/app/data
            data_dir = None
            potential_data_dirs = [Path("/data"), Path("./data"), Path("/home/user/app/data")]
            for d in potential_data_dirs:
                if d.exists() and os.access(str(d), os.W_OK):
                    data_dir = d
                    break
            
            # Set standard directories relative to repo_dir initially
            checkpoints_dir = repo_dir / "checkpoints"
            configs_dir = repo_dir / "configs"

            # HF Spaces persistent storage override
            if data_dir:
                data_checkpoints = data_dir / "checkpoints"
                # Solo creamos si es posible
                try:
                    if not data_checkpoints.exists():
                        data_checkpoints.mkdir(parents=True, exist_ok=True)
                    checkpoints_dir = data_checkpoints
                    print(f"📍 HF Persistent Storage detected: {checkpoints_dir}")
                except:
                    print("⚠️ HF Persistent Storage (/data) is not writable. Using local storage.")
            else:
                print("ℹ️ HF Free Tier detected (No persistent storage). Using local session storage.")
            
            import torch
            return Config(
                platform=platform,
                work_dir=work_dir,
                repo_dir=repo_dir,
                output_dir=work_dir / "outputs",
                checkpoints_dir=checkpoints_dir,
                configs_dir=configs_dir,
                blender_exe=blender_exe,
                has_gpu=torch.cuda.is_available(),
                vram_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
                needs_share=needs_share
            )
        elif Path("/app").exists() and (os.environ.get("DOCKER_CONTAINER", "false") == "true" or Path("/.dockerenv").exists()):
            platform = 'docker'
            work_dir = Path("/app")
            repo_dir = Path("/app")
            blender_exe = Path("/app/blender/blender") 
            needs_share = False
        elif 'PINOKIO_HOME' in os.environ:
            platform = 'pinokio'
            work_dir = Path.cwd()
            repo_dir = work_dir
            
            # 1. Check local folder (Pinokio style)
            blender_exe = work_dir / "blender" / ("blender.exe" if sys.platform == 'win32' else "blender")
            
            # 2. Check system PATH if local not found
            if not blender_exe.exists():
                import shutil
                system_blender = shutil.which("blender")
                if system_blender:
                    blender_exe = Path(system_blender)
                elif sys.platform == 'win32':
                    # 3. Check common Windows paths for Blender 4.2+
                    common_paths = [
                        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
                        Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
                        Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"),
                        Path(r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"),
                    ]
                    for p in common_paths:
                        if p.exists():
                            blender_exe = p
                            break
            
            needs_share = False
        else:
            platform = 'local'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("blender/blender") 
            needs_share = False

        # HF Spaces persistent storage override (moved to detection logic for HF)
        if platform != 'huggingface':
            checkpoints_dir = repo_dir / "checkpoints"
            configs_dir = repo_dir / "configs"
        
        # Determine output_dir
        output_dir = work_dir / "outputs"

        import torch
        return Config(
            platform=platform,
            work_dir=work_dir,
            repo_dir=repo_dir,
            output_dir=output_dir,
            checkpoints_dir=checkpoints_dir,
            configs_dir=configs_dir,
            blender_exe=blender_exe,
            has_gpu=torch.cuda.is_available(),
            vram_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
            needs_share=needs_share
        )

cfg = Config.detect()
