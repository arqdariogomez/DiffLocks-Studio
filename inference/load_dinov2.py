import os
import torch
import torchvision.transforms as T
from pathlib import Path
import urllib.request
import time
from tqdm import tqdm
import zipfile
import shutil

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path, max_retries=3, retry_delay=5):
    """
    Download a file from a URL with retry logic and progress bar.
    """
    for attempt in range(max_retries):
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(url)) as t:
                urllib.request.urlretrieve(
                    url,
                    filename=output_path,
                    reporthook=t.update_to
                )
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    return False

def setup_dinov2_cache():
    """Set up the cache directory for DINOv2."""
    # Define cache directories - Use /app/.cache/torch as defined in docker-compose
    # If not in Docker, it will still use this path which is fine or can be overridden
    torch_home = os.environ.get('TORCH_HOME', '/app/.cache/torch')
    torch_home = Path(torch_home)
    hub_dir = torch_home / "hub"
    dinov2_repo_dir = hub_dir / "facebookresearch_dinov2_main"
    
    # Create hub directory
    hub_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['TORCH_HOME'] = str(torch_home)
    
    return str(hub_dir), str(dinov2_repo_dir)

def download_dinov2_repo(repo_dir):
    """Download the DINOv2 repository source code if missing."""
    repo_dir = Path(repo_dir)
    if (repo_dir / "hubconf.py").exists():
        return True
    
    print(f"DINOv2 repository source not found at {repo_dir}. Downloading...")
    zip_path = repo_dir.parent / "dinov2_repo.zip"
    repo_url = "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip"
    
    try:
        # Ensure parent dir exists
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        
        download_url(repo_url, str(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # The zip contains a folder 'dinov2-main'
            zip_ref.extractall(repo_dir.parent)
        
        # Move the extracted folder to what torch.hub expects
        extracted_folder = repo_dir.parent / "dinov2-main"
        if extracted_folder.exists():
            print(f"Moving {extracted_folder} to {repo_dir}...")
            if repo_dir.exists():
                try:
                    if repo_dir.is_dir():
                        shutil.rmtree(repo_dir)
                    else:
                        repo_dir.unlink()
                except Exception as cleanup_err:
                    print(f"Warning: Could not cleanup existing repo dir: {cleanup_err}")
            
            try:
                # Use shutil.move instead of rename for better cross-device support
                shutil.move(str(extracted_folder), str(repo_dir))
            except Exception as move_err:
                print(f"Move failed, trying copy: {move_err}")
                if repo_dir.exists(): shutil.rmtree(repo_dir, ignore_errors=True)
                shutil.copytree(str(extracted_folder), str(repo_dir))
                shutil.rmtree(extracted_folder, ignore_errors=True)
        
        if zip_path.exists():
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"Failed to download DINOv2 repository: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

def load_dinov2(device=None):
    """
    Load DINOv2 model with robust downloading and caching.
    Returns the model and the transform function.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up cache
    hub_dir, dinov2_repo_dir = setup_dinov2_cache()
    
    # Model configuration
    model_name = "dinov2_vitl14_reg"
    model_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
    model_weights_path = Path(dinov2_repo_dir) / "dinov2_vitl14_reg4_pretrain.pth"
    
    # First, try loading normally via torch.hub from GitHub
    try:
        print("Attempting to load DINOv2 model from GitHub...")
        model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True,
            source='github',
            force_reload=False,
            verbose=True
        )
        model = model.to(device).float()
        model.eval()
        print("Successfully loaded DINOv2 from GitHub/Cache.")
        return model, T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    except Exception as e:
        print(f"GitHub/Default load failed: {e}. Switching to robust local fallback...")

    # Fallback: Ensure repo source exists
    if not download_dinov2_repo(dinov2_repo_dir):
        raise RuntimeError("Failed to obtain DINOv2 repository source code.")

    # Ensure weights exist
    if not model_weights_path.exists():
        print(f"Downloading DINOv2 weights to {model_weights_path}...")
        try:
            download_url(model_url, str(model_weights_path))
        except Exception as download_error:
            print(f"Failed to download DINOv2 weights: {download_error}")
            raise RuntimeError("Could not download DINOv2 weights.")

    # Load using local source
    try:
        print(f"Loading DINOv2 from local source: {dinov2_repo_dir}")
        model = torch.hub.load(
            str(dinov2_repo_dir),
            model_name,
            pretrained=False,
            source='local',
            verbose=False
        )
        # Load the state dict
        print(f"Loading weights from {model_weights_path}...")
        state_dict = torch.load(model_weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).float()
        model.eval()
        print("Successfully loaded DINOv2 from local fallback.")
    except Exception as load_error:
        print(f"Final error loading DINOv2 model: {load_error}")
        raise RuntimeError(f"Failed to load DINOv2 model: {load_error}")
    
    # Create the transform
    transform = T.Compose([
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return model, transform
