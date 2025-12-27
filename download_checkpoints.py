import os
import sys
import shutil
import argparse
import requests
from pathlib import Path

def download_from_meshcapade(user, password, checkpoints_dir):
    """
    Automated download from Meshcapade.
    Note: Requires MESH_USER and MESH_PASS secrets.
    """
    if not user or not password:
        print("❌ Meshcapade credentials missing (MESH_USER/MESH_PASS)")
        return False

    print(f"🔐 Attempting official download from Meshcapade for user: {user}")
    
    # Checkpoints we need
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # REGLA: No redistribuir checkpoints.
    # El usuario debe descargarlos.
    print("ℹ️  Please ensure you have accepted the terms at https://meshcapade.com/models")
    
    # Try custom URL first if provided
    custom_url = os.environ.get("MESH_DOWNLOAD_URL")
    if custom_url:
        print(f"🚀 Downloading from custom Meshcapade link...")
        try:
            r = requests.get(custom_url, stream=True, timeout=60)
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

    # --- NEW: OFFICIAL LOGIN FLOW ---
    try:
        import requests
        session = requests.Session()
        
        # 1. Get login page to capture any CSRF or cookies
        print("🌐 Connecting to Meshcapade...")
        login_page = session.get("https://meshcapade.com/login", timeout=20)
        
        # 2. Login
        # Meshcapade uses 'email' or 'username'. We try to be robust.
        login_data = {
            "email": user,
            "username": user,
            "password": password,
            "remember": "on"
        }
        
        print("🔑 Logging in...")
        # Most modern sites use a JSON API or a form POST
        # We try both common patterns
        try:
            res = session.post("https://meshcapade.com/api/v1/auth/login", json=login_data, timeout=20)
            if res.status_code != 200:
                res = session.post("https://meshcapade.com/login", data=login_data, timeout=20)
        except:
            res = session.post("https://meshcapade.com/login", data=login_data, timeout=20)

        # 3. Check if we are logged in (usually by checking a cookie or a redirect)
        cookies = session.cookies.get_dict()
        is_logged_in = any(k for k in cookies if "session" in k.lower() or "auth" in k.lower() or "token" in k.lower())
        
        if not is_logged_in and res.status_code != 200 and "dashboard" not in res.url:
            print("❌ Meshcapade login failed. Please check your credentials.")
            return False

        print("✅ Login successful! Searching for DiffLocks checkpoints...")
        
        # 4. Download models
        # These are the specific URLs for DiffLocks models on Meshcapade
        models_to_download = [
            ("difflocks_diffusion", "https://meshcapade.com/models/download/difflocks_diffusion"),
            ("strand_vae", "https://meshcapade.com/models/download/strand_vae")
        ]
        
        success_count = 0
        for name, url in models_to_download:
            print(f"📥 Downloading {name}...")
            r = session.get(url, stream=True, timeout=120)
            if r.status_code == 200:
                zip_path = checkpoints_dir / f"{name}.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
                
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Some zips contain the folder, others just the files.
                    # We extract everything to a temp folder and then move files to the right place.
                    temp_extract = checkpoints_dir / f"temp_{name}"
                    temp_extract.mkdir(parents=True, exist_ok=True)
                    zip_ref.extractall(temp_extract)
                    
                    target_subfolder = checkpoints_dir / name
                    target_subfolder.mkdir(parents=True, exist_ok=True)
                    
                    # Find where the actual files are (might be nested)
                    files_found = list(temp_extract.rglob("scalp_*.pth")) or list(temp_extract.rglob("*.pt"))
                    if files_found:
                        source_dir = files_found[0].parent
                        for item in source_dir.iterdir():
                            shutil.move(str(item), str(target_subfolder / item.name))
                    
                    shutil.rmtree(temp_extract)
                
                os.remove(zip_path)
                print(f"✅ {name} downloaded and extracted.")
                success_count += 1
            else:
                print(f"⚠️ Could not download {name} directly (Status: {r.status_code})")
        
        if success_count > 0:
            return True

    except Exception as e:
        print(f"⚠️ Automated download failed: {e}")

    print("\n💡 Automated scraping of Meshcapade is restricted or URLs have changed.")
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

    from platform_config import cfg
    print(f"🚀 DiffLocks Asset Downloader (Platform: {cfg.platform})")
    base_dir = cfg.repo_dir
    checkpoints_dir = cfg.checkpoints_dir
    
    # 1. Download Public Assets (Always needed)
    token = os.environ.get("HF_TOKEN")
    if token == "null" or token == "": token = None
    download_public_assets(base_dir, token=token)

    # 2. Check for Checkpoints
    # If they exist, we are done
    
    # Simple check for checkpoints
    search_dirs = [
        checkpoints_dir,
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
        if download_from_meshcapade(user, password, checkpoints_dir):
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
