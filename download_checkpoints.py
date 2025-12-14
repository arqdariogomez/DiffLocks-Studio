#!/usr/bin/env python3
"""
DiffLocks Checkpoint Downloader
Compatible with: HF Space, Kaggle, Colab, Pinokio, Local
"""
import os
import sys
import shutil
from pathlib import Path

def main():
    print("🚀 DiffLocks Asset Downloader")
    print("=" * 50)
    
    # Detectar token
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("⚠️  HF_TOKEN not found in environment.")
        print("   For Pinokio: Set it in Pinokio Settings -> Environment Variables")
        print("   For local: export HF_TOKEN=your_token")
        print("")
        print("   Attempting download without token (may fail for private repos)...")
        token = None
    
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download, hf_hub_download
    
    REPO_ID = "arqdariogomez/difflocks-assets-hybrid"
    
    # Detectar directorio base
    if Path("app.py").exists():
        base_dir = Path(".")
    elif Path("app/app.py").exists():
        base_dir = Path("app")
    else:
        base_dir = Path(".")
    
    print(f"📁 Base directory: {base_dir.absolute()}")
    
    try:
        # 1. Descargar checkpoints
        print("\n🔹 Downloading checkpoints...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["checkpoints/*", "*.pth", "*.pt"],
            local_dir=str(base_dir),
            token=token
        )
        
        # 2. Descargar assets para inference
        print("\n🔹 Downloading inference assets...")
        assets_dir = base_dir / "inference" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["assets/*"],
            local_dir=str(base_dir / "inference"),
            token=token
        )
        
        # Mover assets si quedaron en subcarpeta
        src_assets = base_dir / "inference" / "assets" / "assets"
        if src_assets.exists():
            for f in src_assets.glob("*"):
                shutil.move(str(f), str(assets_dir / f.name))
            shutil.rmtree(src_assets)
        
        # Verificar descarga
        pth_files = list(base_dir.rglob("*.pth"))
        pt_files = list(base_dir.rglob("strand_codec.pt"))
        
        print("\n" + "=" * 50)
        print("📊 Download Summary:")
        print(f"   • .pth files: {len(pth_files)}")
        print(f"   • strand_codec.pt: {'✅ Found' if pt_files else '❌ Missing'}")
        
        if pth_files:
            print("\n✅ Download Complete!")
            return True
        else:
            print("\n⚠️  No checkpoint files found. Check your HF_TOKEN permissions.")
            return False
            
    except Exception as e:
        print(f"\n❌ Download Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify HF_TOKEN is set correctly")
        print("  2. Ensure you have access to the dataset repo")
        print("  3. Check your internet connection")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
