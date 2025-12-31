import subprocess
from pathlib import Path
import sys
import os

# Configuration
REPO_DIR = Path(r"c:\pinokio\api\DiffLocks-Studio")
BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe")
# If that doesn't exist, try common fallback
if not BLENDER_EXE.exists():
    BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe")
if not BLENDER_EXE.exists():
    BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 3.1\blender.exe")
if not BLENDER_EXE.exists():
    # Try common Pinokio path or search
    import shutil
    system_blender = shutil.which("blender")
    if system_blender:
        BLENDER_EXE = Path(system_blender)
    else:
        # Check other versions
        for v in ["4.3", "4.2", "4.1", "4.0", "3.6"]:
            p = Path(fr"C:\Program Files\Blender Foundation\Blender {v}\blender.exe")
            if p.exists():
                BLENDER_EXE = p
                break

NPZ_PATH = Path(r"C:\pinokio\api\DiffLocks-Studio\studio_outputs\job_1766797132\difflocks_output_strands.npz")
if not NPZ_PATH.exists():
    # Try relative to repo
    NPZ_PATH = REPO_DIR / "studio_outputs/job_1766797132/difflocks_output_strands.npz"

OUTPUT_BASE = REPO_DIR / "blender_test_real_output"
SCRIPT_PATH = REPO_DIR / "inference/converter_blender.py"

print(f"--- Blender Real NPZ Test ---")
print(f"Blender: {BLENDER_EXE}")
print(f"NPZ: {NPZ_PATH}")
print(f"Script: {SCRIPT_PATH}")

if not BLENDER_EXE.exists():
    print(f"❌ Blender not found at {BLENDER_EXE}")
    sys.exit(1)

if not NPZ_PATH.exists():
    print(f"❌ NPZ not found at {NPZ_PATH}")
    sys.exit(1)

cmd = [
    str(BLENDER_EXE),
    "-b",
    "-P", str(SCRIPT_PATH),
    "--",
    str(NPZ_PATH),
    str(OUTPUT_BASE),
    "blend", "abc", "usd"
]

print(f"Running command: {' '.join(cmd)}")
try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in process.stdout:
        print(f"[Blender] {line.strip()}")
    process.wait()
    
    print("\n--- Results ---")
    for ext in [".blend", ".abc", ".usd"]:
        p = Path(f"{OUTPUT_BASE}{ext}")
        if p.exists():
            print(f"✅ Created: {p.name} ({p.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"❌ Missing: {p.name}")

except Exception as e:
    print(f"❌ Execution failed: {e}")
