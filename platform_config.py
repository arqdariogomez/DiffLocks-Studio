
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
    def ensure_persistence(repo_dir: Path, data_dir: Path) -> Path:
        """Helper to link repo/checkpoints to a persistent data directory."""
        data_checkpoints = data_dir / "checkpoints"
        local_checkpoints = repo_dir / "checkpoints"
        
        try:
            if not data_checkpoints.exists():
                data_checkpoints.mkdir(parents=True, exist_ok=True)
            
            # If local exists and is NOT a link, move contents to data_dir then delete local
            if local_checkpoints.exists() and not local_checkpoints.is_link():
                import shutil
                print(f"🔄 Moving existing checkpoints to persistent storage: {data_checkpoints}")
                for item in local_checkpoints.iterdir():
                    dest = data_checkpoints / item.name
                    if not dest.exists():
                        if item.is_dir():
                            shutil.copytree(str(item), str(dest))
                            shutil.rmtree(str(item))
                        else:
                            shutil.move(str(item), str(dest))
                shutil.rmtree(local_checkpoints)
            
            # Create symlink if it doesn't exist
            if not local_checkpoints.exists():
                # On Windows, symlinks might need admin or specific dev mode
                # Fallback to just using the data_dir if symlink fails
                try:
                    os.symlink(str(data_checkpoints), str(local_checkpoints))
                    print(f"🔗 Symlink created: {local_checkpoints} -> {data_checkpoints}")
                    return local_checkpoints
                except Exception as e:
                    print(f"⚠️ Symlink failed ({e}). Using direct path instead.")
                    return data_checkpoints
            
            return local_checkpoints
        except Exception as e:
            print(f"⚠️ Persistence setup error: {e}")
            return local_checkpoints

    @staticmethod
    def detect() -> 'Config':
        platform: PlatformType = 'local'
        data_dir = None
        
        if 'COLAB_GPU' in os.environ or Path("/content").exists():
            platform = 'colab'
            work_dir = Path("/content")
            repo_dir = work_dir / "DiffLocks-Studio"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
            # Colab often uses Drive for persistence
            if Path("/content/drive/MyDrive").exists():
                data_dir = Path("/content/drive/MyDrive/DiffLocks_Data")
        elif Path("/kaggle").exists():
            platform = 'kaggle'
            work_dir = Path("/kaggle/working")
            repo_dir = work_dir / "DiffLocks-Studio"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
        elif any(k in os.environ for k in ['SPACE_ID', 'SPACE_REPO_NAME', 'HF_SPACE_ID']): 
            platform = 'huggingface'
            work_dir = Path("/app") if Path("/app").exists() else (Path("/home/user/app") if Path("/home/user/app").exists() else Path.cwd())
            repo_dir = work_dir
            blender_exe = Path("/tmp/blender/blender") 
            needs_share = False
            # HF Spaces persistent storage
            potential_data_dirs = [Path("/data"), Path("./data"), Path("/home/user/app/data")]
            for d in potential_data_dirs:
                if d.exists() and os.access(str(d), os.W_OK):
                    data_dir = d
                    break
        elif 'PINOKIO_HOME' in os.environ:
            platform = 'pinokio'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = work_dir / "blender" / ("blender.exe" if sys.platform == 'win32' else "blender")
            needs_share = False
            # Pinokio persistent data can be in a 'data' folder next to the script
            if (work_dir.parent / "data").exists():
                data_dir = work_dir.parent / "data"
        elif Path("/app").exists() and (os.environ.get("DOCKER_CONTAINER", "false") == "true" or Path("/.dockerenv").exists()):
            platform = 'docker'
            work_dir = Path("/app")
            repo_dir = Path("/app")
            blender_exe = Path("/app/blender/blender") 
            needs_share = False
            if Path("/data").exists(): data_dir = Path("/data")
        else:
            platform = 'local'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("blender/blender") 
            needs_share = False

        # Apply persistence if data_dir found
        checkpoints_dir = repo_dir / "checkpoints"
        if data_dir:
            checkpoints_dir = Config.ensure_persistence(repo_dir, data_dir)
            
        configs_dir = repo_dir / "configs"
        output_dir = work_dir / "outputs"
        
        # Robust Blender check for local/pinokio
        if not blender_exe.exists() and platform in ['local', 'pinokio']:
            import shutil
            system_blender = shutil.which("blender")
            if system_blender:
                blender_exe = Path(system_blender)
            elif sys.platform == 'win32':
                for v in ["4.2", "4.1", "4.0", "3.6"]:
                    p = Path(fr"C:\Program Files\Blender Foundation\Blender {v}\blender.exe")
                    if p.exists():
                        blender_exe = p
                        break

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
