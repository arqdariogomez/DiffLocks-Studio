#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

def main():
    print("🚀 DiffLocks Asset Downloader")
    
    # Intentar obtener token de argumento, entorno o input de pinokio
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
        # Checkpoints
        print("🔹 Downloading checkpoints...")
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["checkpoints/*", "*.pth", "*.pt"], local_dir=str(base_dir), token=token)
        
        # Assets
        print("🔹 Downloading assets...")
        assets_dir = base_dir / "inference" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], local_dir=str(base_dir / "inference"), token=token)
        
        # Cleanup rutas anidadas si ocurren
        src_assets = base_dir / "inference" / "assets" / "assets"
        if src_assets.exists():
            for f in src_assets.glob("*"):
                shutil.move(str(f), str(assets_dir / f.name))
            shutil.rmtree(src_assets)
            
        print("✅ Download Complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
