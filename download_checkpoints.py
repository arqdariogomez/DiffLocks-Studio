import os
import sys
import shutil
import argparse
import requests
import zipfile
from pathlib import Path

def unzip_checkpoints(zip_path, target_dir):
    """Unzips the checkpoints and organizes them into the target directory."""
    print(f"📦 Extracting {zip_path.name} to {target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Create a temporary extraction directory
            temp_extract = target_dir / "temp_unzip"
            if temp_extract.exists(): shutil.rmtree(temp_extract)
            temp_extract.mkdir(parents=True, exist_ok=True)
            
            zip_ref.extractall(temp_extract)
            
            # Look for the relevant folders (difflocks_diffusion, strand_vae)
            # They might be at the root of the zip or nested
            found_folders = False
            for folder_name in ["difflocks_diffusion", "strand_vae"]:
                # Search for this folder in the extracted content
                matches = list(temp_extract.rglob(folder_name))
                if matches:
                    source_folder = matches[0]
                    dest_folder = target_dir / folder_name
                    if dest_folder.exists(): shutil.rmtree(dest_folder)
                    shutil.move(str(source_folder), str(dest_folder))
                    print(f"✅ Moved {folder_name} to {dest_folder}")
                    found_folders = True
            
            shutil.rmtree(temp_extract)
            return found_folders
    except Exception as e:
        print(f"❌ Error unzipping {zip_path.name}: {e}")
        return False

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
    
    # We use the public env repo which also contains necessary assets or is accessible
    REPO_ID = "arqdariogomez/difflocks-env"
    print(f"🔹 Downloading public resources from {REPO_ID}...")
    try:
        # Try downloading from the public repo. No token needed.
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["assets/*"], 
                         local_dir=str(base_dir / "inference"), token=token)
        return True
    except Exception as e:
        # Fallback to the hybrid repo if env doesn't have them yet, or if env download fails
        ALT_REPO = "arqdariogomez/difflocks-assets-hybrid"
        if REPO_ID != ALT_REPO:
            try:
                print(f"🔹 Attempting fallback to {ALT_REPO}...")
                snapshot_download(repo_id=ALT_REPO, repo_type="dataset", allow_patterns=["assets/*"], 
                                 local_dir=str(base_dir / "inference"), token=token)
                return True
            except:
                pass

        error_str = str(e)
        if "401" in error_str or "Unauthorized" in error_str or "Repository Not Found" in error_str:
            print(f"⚠️ Access denied to assets. If the repo is private, please set HF_TOKEN.")
            
            # Fallback check: if we already have some assets, don't fail hard
            asset_path = base_dir / "inference" / "assets"
            if asset_path.exists() and any(asset_path.iterdir()):
                print("✅ Found existing assets in inference/assets. Continuing...")
                return True
        else:
            print(f"⚠️ Error downloading public assets: {e}")
        return False

def backup_to_hf(checkpoints_dir, token):
    """Backs up checkpoints to a private HF dataset for persistence."""
    if not token:
        return
    
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/difflocks-checkpoints"
        
        print(f"☁️  Attempting to backup checkpoints to HF: {repo_id}")
        
        # 1. Create repo if it doesn't exist
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"⚠️ Could not create backup repo: {e}")
                return

        # 2. Upload the folders
        # We only upload if they aren't already there to save time
        print(f"📤 Uploading checkpoints to your private HF repo (this happens once)...")
        api.upload_folder(
            folder_path=str(checkpoints_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Backup complete! Your checkpoints are now safe at {repo_id}")
        return repo_id
    except Exception as e:
        print(f"⚠️ HF Backup skipped: {e}")
        return None

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
    
    # In Colab, we can try to get the token from userdata if present, but it's optional
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token:
                os.environ["HF_TOKEN"] = token # Export for other tools
        except:
            pass
            
    if token == "null" or token == "": token = None
    download_public_assets(base_dir, token=token)

    # 2. Check for Checkpoints
    # If they exist, we are done
    
    # Simple check for checkpoints
    search_dirs = [
        checkpoints_dir,
        cfg.repo_dir / "checkpoints",
        Path("/data/checkpoints") if Path("/data").exists() else None,
        Path("/app/checkpoints") if Path("/app").exists() else None,
        Path("/kaggle/input/difflocks-checkpoints") if Path("/kaggle").exists() else None
    ]
    search_dirs = [d for d in search_dirs if d and d.exists()]
    
    found_ckpt = False
    for d in search_dirs:
        # Check if we have the diffusion weights and the VAE/Codec
        if (list(d.rglob("scalp_*.pth")) or list(d.rglob("*.ckpt"))) and \
           (list(d.rglob("strand_codec.pt")) or list(d.rglob("*.pt"))):
            found_ckpt = True
            print(f"✅ Checkpoints found in: {d}")
            break
            
    if found_ckpt:
        return True

    # 2.5 Check for local zip file (difflocks_checkpoints.zip)
    # This is common in Pinokio or manual setups
    potential_zips = [
        cfg.repo_dir / "difflocks_checkpoints.zip",
        cfg.repo_dir / "checkpoints.zip",
        Path("/kaggle/input/difflocks-checkpoints/difflocks_checkpoints.zip") if Path("/kaggle").exists() else None
    ]
    
    for zip_p in potential_zips:
        if zip_p and zip_p.exists():
            print(f"📦 Found local zip: {zip_p}")
            if unzip_checkpoints(zip_p, checkpoints_dir):
                print("✅ Checkpoints extracted from local zip!")
                return True

    # 3. Try private HF repo if specified (Alternative to Google Drive for HF/Colab)
    # Note: On Kaggle, if you have a dataset connected, it's found in step 2 (Faster).
    # This step is the universal fallback for Colab, HF Spaces, or local without datasets.
    hf_ckpt_repo = os.environ.get("HF_CHECKPOINTS_REPO")
    
    # Auto-detect private repo if not specified but token is present
    if not hf_ckpt_repo and token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            username = api.whoami()['name']
            potential_repo = f"{username}/difflocks-checkpoints"
            # Quick check if it exists
            api.repo_info(repo_id=potential_repo, repo_type="dataset")
            hf_ckpt_repo = potential_repo
            print(f"✨ Auto-detected your private backup repo: {hf_ckpt_repo}")
        except:
            pass

    if hf_ckpt_repo:
        print(f"🔹 Attempting to download checkpoints from private HF repo: {hf_ckpt_repo}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_ckpt_repo,
                repo_type="dataset",
                local_dir=str(checkpoints_dir),
                token=token
            )
            # Re-verify after download
            if (list(checkpoints_dir.rglob("scalp_*.pth")) or list(checkpoints_dir.rglob("*.ckpt"))) and \
               (list(checkpoints_dir.rglob("strand_codec.pt")) or list(checkpoints_dir.rglob("*.pt"))):
                print("✅ Checkpoints downloaded from private HF repo!")
                return True
        except Exception as e:
            print(f"⚠️ Error downloading from HF repo {hf_ckpt_repo}: {e}")

    # 4. Try Meshcapade login if credentials provided
    user = os.environ.get("MESH_USER")
    password = os.environ.get("MESH_PASS")
    
    if user and password:
        if download_from_meshcapade(user, password, checkpoints_dir):
            print("✅ Download from Meshcapade successful!")
            backup_to_hf(checkpoints_dir, token)
            return True
    
    # 5. Fallback: Try official MPG link or direct download URL if provided
    mpg_url = "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=difflocks_checkpoints.zip"
    direct_url = os.environ.get("CHECKPOINTS_URL") or os.environ.get("MESH_DOWNLOAD_URL") or mpg_url
    
    if direct_url:
        print(f"🚀 Attempting direct download from: {direct_url}")
        try:
            zip_path = checkpoints_dir / "downloaded_checkpoints.zip"
            # Use stream=True for large files
            r = requests.get(direct_url, stream=True, timeout=300)
            
            # Check if it's the MPG link and handle potential redirects/headers
            if r.status_code == 200:
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (1024*1024*50) < 8192: # Log every 50MB
                                print(f"   ... {downloaded / (1024*1024):.1f} MB downloaded")
                
                if unzip_checkpoints(zip_path, checkpoints_dir):
                    if zip_path.exists(): os.remove(zip_path)
                    backup_to_hf(checkpoints_dir, token)
                    return True
            else:
                print(f"❌ Direct download failed (Status: {r.status_code})")
        except Exception as e:
            print(f"⚠️ Error during direct download: {e}")

    print("\n❌ [REQUIRED] Checkpoints missing!")
    print("💡 Due to licensing (Non-Commercial), you must provide the model weights.")
    print("Methods to fix this:")
    print("1. [Manual] Download 'difflocks_checkpoints.zip' from https://difflocks.is.tue.mpg.de/")
    print(f"   and place it in: {cfg.repo_dir.absolute()}")
    print("2. [Secrets] Set MESH_USER and MESH_PASS for automated download from Meshcapade.")
    print("3. [Direct] Set CHECKPOINTS_URL to a direct download link of the zip file.")
    
    if cfg.platform == 'hf':
        print("\n🤗 On Hugging Face Spaces, the best way is to set CHECKPOINTS_URL in your Space Secrets.")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
