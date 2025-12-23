import os
import torch
import torchvision.transforms as T
from pathlib import Path
import shutil

def setup_dinov2_cache():
    """Set up the cache directory structure for DINOv2."""
    # Define cache directories
    cache_dir = Path("/app/.cache")
    torch_home = cache_dir / "torch"
    hub_dir = torch_home / "hub"
    dinov2_dir = hub_dir / "facebookresearch_dinov2_main"
    
    # Create directories
    dinov2_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy required files from torch hub cache if they exist
    try:
        # Try to get the default hub directory
        import torch.hub
        default_hub_dir = torch.hub.get_dir()
        
        # If the default hub directory is different from our target
        if os.path.abspath(default_hub_dir) != str(hub_dir):
            # Copy the DINOv2 files if they exist in the default location
            src_dir = Path(default_hub_dir) / "facebookresearch_dinov2_main"
            if src_dir.exists():
                for item in src_dir.glob("*"):
                    dest = dinov2_dir / item.name
                    if not dest.exists():
                        if item.is_dir():
                            shutil.copytree(item, dest)
                        else:
                            shutil.copy2(item, dest)
    except Exception as e:
        print(f"Warning: Could not copy DINOv2 files: {e}")
    
    return str(hub_dir)

def load_dinov2(device=None):
    """
    Load DINOv2 model with caching to a persistent directory.
    Returns the model and the transform function.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up cache directory
    hub_dir = setup_dinov2_cache()
    
    # Set environment variable for torch.hub
    os.environ['TORCH_HUB'] = hub_dir
    
    try:
        # Try to load the model with the standard hub loader first
        print("Loading DINOv2 model...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=True)
        model = model.to(device).float()
        model.eval()
        
    except Exception as e:
        print(f"Error loading DINOv2: {e}")
        print("Falling back to direct download...")
        
        # Fallback: Download the model directly
        import urllib.request
        import zipfile
        
        model_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
        model_path = Path(hub_dir) / "dinov2_vitl14_reg4_pretrain.pth"
        
        if not model_path.exists():
            print("Downloading DINOv2 model...")
            urllib.request.urlretrieve(model_url, model_path)
        
        # Load the model architecture
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=False, verbose=False)
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).float()
        model.eval()
    
    # Create the transform
    transform = T.Compose([
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return model, transform
