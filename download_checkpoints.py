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
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://meshcapade.com/login",
            "Origin": "https://meshcapade.com"
        })
        
        # 1. Get login page to capture any CSRF or cookies
        print("🌐 Connecting to Meshcapade...")
        login_page = session.get("https://meshcapade.com/login", timeout=30)
        
        # 2. Login
        # Meshcapade uses 'email' or 'username'. We try to be robust.
        login_data = {
            "email": user,
            "username": user,
            "password": password,
            "remember": "on"
        }
        
        print(f"🔑 Logging in as {user}...")
        # Most modern sites use a JSON API or a form POST
        # We try both common patterns
        try:
            res = session.post("https://meshcapade.com/api/v1/auth/login", json=login_data, timeout=30)
            if res.status_code not in [200, 201]:
                res = session.post("https://meshcapade.com/login", data=login_data, timeout=30)
        except:
            res = session.post("https://meshcapade.com/login", data=login_data, timeout=30)

        # 3. Check if we are logged in (usually by checking a cookie or a redirect)
        cookies = session.cookies.get_dict()
        is_logged_in = any(k for k in cookies if any(x in k.lower() for x in ["session", "auth", "token", "member"]))
        
        if not is_logged_in and "dashboard" not in res.url and res.status_code not in [200, 201]:
            print(f"❌ Meshcapade login failed (Status: {res.status_code}).")
            if "Incorrect" in res.text or "invalid" in res.text.lower():
                print("❌ Possible invalid credentials.")
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
            r = session.get(url, stream=True, timeout=180)
            
            # If we get a redirect back to login, the download failed
            if "login" in r.url and r.status_code == 200:
                 print(f"⚠️ Could not download {name}: Redirected to login. Session might have expired.")
                 continue

            if r.status_code == 200:
                content_type = r.headers.get('Content-Type', '')
                if 'html' in content_type.lower():
                    print(f"⚠️ Could not download {name}: Received HTML instead of a file. (Check terms acceptance at meshcapade.com)")
                    continue

                zip_path = checkpoints_dir / f"{name}.zip"
                total_size = int(r.headers.get('content-length', 0))
                
                print(f"📦 File size: {total_size / (1024*1024):.1f} MB")
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024*1024*10) < 65536: # Log every 10MB
                            print(f"   ... {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                
                print(f"📂 Extracting {name}...")
                import zipfile
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Some zips contain the folder, others just the files.
                        # We extract everything to a temp folder and then move files to the right place.
                        temp_extract = checkpoints_dir / f"temp_{name}"
                        if temp_extract.exists(): shutil.rmtree(temp_extract)
                        temp_extract.mkdir(parents=True, exist_ok=True)
                        zip_ref.extractall(temp_extract)
                        
                        target_subfolder = checkpoints_dir / name
                        target_subfolder.mkdir(parents=True, exist_ok=True)
                        
                        # Find where the actual files are (might be nested)
                        files_found = list(temp_extract.rglob("scalp_*.pth")) or list(temp_extract.rglob("*.pt"))
                        if files_found:
                            source_dir = files_found[0].parent
                            for item in source_dir.iterdir():
                                dest = target_subfolder / item.name
                                if dest.exists(): 
                                    if dest.is_dir(): shutil.rmtree(dest)
                                    else: dest.unlink()
                                shutil.move(str(item), str(dest))
                            print(f"✅ {name} files moved to {target_subfolder}")
                        else:
                            print(f"⚠️ No relevant files found in zip for {name}")
                        
                        shutil.rmtree(temp_extract)
                    
                    os.remove(zip_path)
                    print(f"✅ {name} completed.")
                    success_count += 1
                except zipfile.BadZipFile:
                    print(f"❌ Error: {name}.zip is not a valid zip file. Download might have been interrupted.")
                    if zip_path.exists(): os.remove(zip_path)
            else:
                print(f"⚠️ Could not download {name} directly (Status: {r.status_code})")
        
        if success_count > 0:
            return True

    except Exception as e:
        print(f"⚠️ Automated download failed: {str(e)}")
        import traceback
        traceback.print_exc()

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
    print(f"🔹 Downloading public resources from {REPO_ID}...")
    try:
        # Try downloading. If it's a 401, it might be private or gated.
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], 
                         local_dir=str(base_dir / "inference"), token=token)
        return True
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "Unauthorized" in error_str or "Repository Not Found" in error_str:
            print(f"⚠️ Access denied to {REPO_ID}. This repository might be private or gated.")
            print("💡 To fix this:")
            print("   1. Create a Hugging Face account if you don't have one.")
            print("   2. Generate a READ token at https://huggingface.co/settings/tokens")
            print("   3. Set the HF_TOKEN secret in your environment (Colab/HF Spaces).")
            
            # Fallback check: if we already have some assets, don't fail hard
            asset_path = base_dir / "inference" / "assets"
            if asset_path.exists() and any(asset_path.iterdir()):
                print("✅ Found existing assets in inference/assets. Continuing...")
                return True
        else:
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
