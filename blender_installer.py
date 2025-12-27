import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

def download_blender(dest_folder: Path):
    """
    Downloads and extracts Blender 4.2.0 for Windows/Linux into the specified folder.
    """
    if sys.platform == 'win32':
        url = "https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.2/blender-4.2.0-windows-x64.zip"
        filename = "blender_win.zip"
    else:
        # Assuming Linux if not Windows for now (Pinokio is mostly Win/Mac/Linux)
        url = "https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz"
        filename = "blender_linux.tar.xz"
        # tar.xz needs different handling, but let's focus on Windows for Pinokio as requested
    
    if not dest_folder.exists():
        dest_folder.mkdir(parents=True, exist_ok=True)
    
    archive_path = dest_folder / filename
    
    print(f"📥 Descargando Blender desde: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(archive_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)
        
        print(f"📦 Extrayendo Blender...")
        if filename.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
        else:
            # For .tar.xz (Linux)
            import tarfile
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(dest_folder)
        
        # Cleanup
        archive_path.unlink()
        print(f"✅ Blender instalado correctamente en {dest_folder}")
        return True
    except Exception as e:
        print(f"❌ Error descargando Blender: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False

if __name__ == "__main__":
    # Test
    base_dir = Path(__file__).parent
    blender_dir = base_dir / "blender"
    download_blender(blender_dir)
