#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

def main():
    print("🚀 DiffLocks Asset Downloader")
    
    # Try to get token from argument, environment or Pinokio input
    token = os.environ.get("HF_TOKEN")
    if token == "null" or token == "": token = None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    REPO_ID = "arqdariogomez/difflocks-assets-hybrid"
    base_dir = Path(".")
    
    print(f"📁 Downloading to: {base_dir.absolute()}")
    
    try:
        # 1. Assets (Face Landmarker, etc.)
        print("🔹 Downloading necessary resources (Face Landmarker, etc.)...")
        assets_dir = base_dir / "inference" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], local_dir=str(base_dir / "inference"), token=token)
        
        # Cleanup nested paths if they occur
        src_assets = base_dir / "inference" / "assets" / "assets"
        if src_assets.exists():
            for f in src_assets.glob("*"):
                shutil.move(str(f), str(assets_dir / f.name))
            shutil.rmtree(src_assets)
            
        # 2. Checkpoints (Models)
        print("🔹 Downloading models (Checkpoints)...")
        checkpoints_dir = base_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Download checkpoints from the same repo (assuming they are there or in another one)
        # Patterns for the checkpoints we need
        patterns = [
            "difflocks_diffusion/scalp_*.pth",
            "strand_vae/strand_codec.pt",
            "rgb2material/rgb2material.pt"
        ]
        
        snapshot_download(
            repo_id=REPO_ID, 
            repo_type="dataset", 
            allow_patterns=patterns, 
            local_dir=str(checkpoints_dir), 
            token=token
        )
            
        print("✅ Download completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
