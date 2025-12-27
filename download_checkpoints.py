import os
import sys
import shutil
import argparse
import requests
from pathlib import Path

def download_from_meshcapade(user, password, base_dir):
    """
    Automated download from Meshcapade.
    Note: Requires MESH_USER and MESH_PASS secrets.
    """
    if not user or not password:
        print("❌ Meshcapade credentials missing (MESH_USER/MESH_PASS)")
        return False

    print(f"🔐 Attempting official download from Meshcapade for user: {user}")
    
    # Checkpoints we need
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # REGLA: No redistribuir checkpoints.
    # El usuario debe descargarlos.
    print("ℹ️  Please ensure you have accepted the terms at https://meshcapade.com/models")
    
    # If the user has provided a custom download URL in environment, use it
    # This is a safe way to allow "automated" download without redistributing
    custom_url = os.environ.get("MESH_DOWNLOAD_URL")
    if custom_url:
        print(f"🚀 Downloading from custom Meshcapade link...")
        try:
            r = requests.get(custom_url, stream=True)
            zip_path = checkpoints_dir / "meshcapade_models.zip"
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(checkpoints_dir)
            os.remove(zip_path)
            print("✅ Models extracted successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Error with custom download: {e}")

    print("\n💡 Automated scraping of Meshcapade is restricted.")
    print("Please manually place the 'difflocks_diffusion' and 'strand_vae' folders in:")
    print(f"📍 {checkpoints_dir.absolute()}")
    
    return False

def download_public_assets(base_dir, token=None):
    """Downloads non-restricted assets (Face Landmarker, etc.) from HF."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import os
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    REPO_ID = "arqdariogomez/difflocks-assets-hybrid"
    print("🔹 Downloading public resources (Face Landmarker, etc.)...")
    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], 
                         local_dir=str(base_dir / "inference"), token=token)
        return True
    except Exception as e:
        print(f"⚠️ Error downloading public assets: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="DiffLocks Asset Downloader")
    parser.add_argument("--meshcapade", action="store_true", help="Use Meshcapade official login")
    parser.add_argument("--hf", action="store_true", help="Use Hugging Face (only for public assets)")
    args = parser.parse_args()

    print("🚀 DiffLocks Asset Downloader")
    base_dir = Path(".")
    checkpoints_dir = base_dir / "checkpoints"
    
    # 1. Download Public Assets (Always needed)
    token = os.environ.get("HF_TOKEN")
    if token == "null" or token == "": token = None
    download_public_assets(base_dir, token=token)

    # 2. Check for Checkpoints
    # If they exist, we are done
    from platform_config import cfg
    
    # Simple check for checkpoints
    search_dirs = [
        Path("checkpoints"),
        cfg.repo_dir / "checkpoints",
        Path("/data/checkpoints") if Path("/data").exists() else None,
        Path("/app/checkpoints") if Path("/app").exists() else None
    ]
    search_dirs = [d for d in search_dirs if d and d.exists()]
    
    found_ckpt = False
    for d in search_dirs:
        if list(d.rglob("scalp_*.pth")) and (list(d.rglob("strand_codec.pt")) or list(d.rglob("*.pt"))):
            found_ckpt = True
            break
            
    if found_ckpt:
        print("✅ Checkpoints already present.")
        return True

    # 3. Try Meshcapade login if credentials provided
    user = os.environ.get("MESH_USER")
    password = os.environ.get("MESH_PASS")
    if user and password:
        if download_from_meshcapade(user, password, base_dir):
            print("✅ Download from Meshcapade successful!")
            return True

    print("\n❌ [REQUIRED] Checkpoints missing!")
    print("💡 Meshcapade prohibits redistribution. Please:")
    print("1. Login to https://meshcapade.com/models")
    print("2. Download DiffLocks checkpoints.")
    print(f"3. Place them in: {checkpoints_dir.absolute()}")
    print("\nOr set MESH_USER and MESH_PASS in your Secrets/Environment for automated attempt.")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
