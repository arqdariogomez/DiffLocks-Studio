
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

# Config
REPO_ID = "arqdariogomez/difflocks-assets-hybrid"

def main():
    print(f"🚀 Downloading assets from {REPO_ID}...")
    
    # Verificar Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ FATAL: 'HF_TOKEN' not found in Environment Secrets!")
        print("   Go to Space Settings -> Variables and secrets -> New Secret")
        print("   Name: HF_TOKEN, Value: Your HuggingFace Write Token")
        sys.exit(1) # Forzar error para que se vea en el log
    
    try:
        # 1. Checkpoints
        print("🔹 Downloading Checkpoints...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="checkpoints/*",
            local_dir=".", 
            token=token
        )
        
        # 2. Blender Assets (El archivo que movimos antes)
        print("🔹 Downloading Assets...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="assets/*",
            local_dir="inference",
            token=token
        )
        
        print("✅ Download Complete!")
        
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
