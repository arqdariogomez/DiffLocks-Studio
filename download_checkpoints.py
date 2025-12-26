import os
import sys
import shutil
import argparse
import requests
from pathlib import Path

def download_from_meshcapade(user, password, base_dir):
    """
    Placeholder for Meshcapade official download.
    Note: Exact implementation depends on Meshcapade's portal structure.
    """
    print(f"🔐 Attempting official download from Meshcapade for user: {user}")
    # In a real scenario, we would use requests.Session() to login and download.
    # Since the exact URLs for DiffLocks are not public yet, we'll guide the user.
    
    # Checkpoints we need
    required = [
        "difflocks_diffusion/scalp_texture_conditional.pth",
        "strand_vae/strand_codec.pt",
        "rgb2material/rgb2material.pt"
    ]
    
    print("⚠️  Official Meshcapade automated download is currently in 'manual-assistance' mode.")
    print("Please ensure you have downloaded the files from https://meshcapade.com/models")
    print(f"and placed them in: {base_dir / 'checkpoints'}")
    
    # If we had the direct links, we would do:
    # session = requests.Session()
    # session.post("https://meshcapade.com/login", data={"username": user, "password": password})
    # session.get("https://meshcapade.com/download/difflocks/models.zip") ...
    
    return False

def main():
    parser = argparse.ArgumentParser(description="DiffLocks Asset Downloader")
    parser.add_argument("--meshcapade", action="store_true", help="Use Meshcapade official login")
    parser.add_argument("--hf", action="store_true", help="Use Hugging Face (requires token for private repos)")
    args = parser.parse_args()

    print("🚀 DiffLocks Asset Downloader")
    base_dir = Path(".")
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check for Meshcapade login
    if args.meshcapade or os.environ.get("MESH_USER"):
        user = os.environ.get("MESH_USER")
        password = os.environ.get("MESH_PASS")
        if user and password:
            if download_from_meshcapade(user, password, base_dir):
                print("✅ Download from Meshcapade successful!")
                return True
        else:
            print("❌ Meshcapade credentials missing in environment (MESH_USER/MESH_PASS)")

    # 2. Try Hugging Face (Existing logic)
    token = os.environ.get("HF_TOKEN")
    if token == "null" or token == "": token = None
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    REPO_ID = "arqdariogomez/difflocks-assets-hybrid"
    
    try:
        # Assets
        print("🔹 Downloading resources (Face Landmarker, etc.)...")
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], 
                         local_dir=str(base_dir / "inference"), token=token)
        
        # Checkpoints
        print("🔹 Downloading models (Checkpoints)...")
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
            token=token,
            local_dir_use_symlinks=False
        )
            
        print("✅ Download completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Download error: {e}")
        print("\n💡 TIP: If this is a private repo, ensure HF_TOKEN is set.")
        print("Alternatively, download manually and place in the 'checkpoints' folder.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
