@echo off
setlocal enabledelayedexpansion

echo 💇‍♀️ DiffLocks Studio - Docker Launcher
echo ========================================

:: Check for Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/
    pause
    exit /b 1
)

:: Check for checkpoints
set "CHECKPOINT_FILE=checkpoints\scalp_diffusion.pth"
set "ZIP_FILE=difflocks_checkpoints.zip"

if not exist "%CHECKPOINT_FILE%" (
    if exist "%ZIP_FILE%" (
        echo 📦 Found %ZIP_FILE%. Unzipping...
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '.' -Force"
        
        :: Handle case where zip contains a 'checkpoints' folder or just files
        if not exist "checkpoints" mkdir checkpoints
        if exist "scalp_diffusion.pth" move "scalp_diffusion.pth" "checkpoints\"
        if exist "strand_vae" move "strand_vae" "checkpoints\"
        
        echo ✅ Unzipped successfully.
    ) else (
        echo ❌ Error: Checkpoints not found!
        echo.
        echo 📥 How to fix:
        echo 1. Register/Login at https://difflocks.is.tue.mpg.de/
        echo 2. Download 'difflocks_checkpoints.zip'
        echo 3. Place the .zip file in THIS folder: %cd%
        echo 4. Run this .bat again.
        echo.
        pause
        exit /b 1
    )
)

:: Check for NVIDIA Container Toolkit (Optional but recommended for GPU)
docker info | findstr /C:"Runtimes: nvidia" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: NVIDIA Container Toolkit not detected. 
    echo If you have an NVIDIA GPU, please install it for 100x faster generation.
    echo Running in CPU mode by default...
    echo.
)

echo 🚀 Starting Docker containers...
echo 🌐 Once ready, open http://localhost:7860 in your browser.
echo.

docker-compose up --build

if %errorlevel% neq 0 (
    echo.
    echo ❌ Docker Compose failed to start.
    pause
    exit /b 1
)

pause
