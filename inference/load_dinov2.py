import os
import torch
import torchvision.transforms as T
from pathlib import Path

def load_dinov2(device=None):
    """
    Load DINOv2 model with caching to a persistent directory.
    Returns the model and the transform function.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the model directory if it doesn't exist
    model_dir = Path("/app/.cache/torch/hub/facebookresearch_dinov2_main")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the model path
    model_name = "dinov2_vitl14_reg"
    model_path = model_dir / f"{model_name}.pth"
    
    # Check if model is already downloaded
    if not model_path.exists():
        print("Downloading DINOv2 model...")
        # Download the model using torch.hub
        model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=True, skip_validation=True)
        # Save the model state dict
        torch.save(model.state_dict(), model_path)
    else:
        print("Loading DINOv2 model from cache...")
        # Load the model architecture
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False, verbose=False)
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Move to device and set to float32
    model = model.to(device).float()
    model.eval()
    
    # Create the transform
    transform = T.Compose([
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return model, transform
