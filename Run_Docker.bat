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

:: Check for checkpoints (Flexible search)
set "FOUND_CKPT="
if exist "checkpoints" (
    for /f "delims=" %%i in ('dir /s /b "checkpoints\scalp_*.pth" 2^>nul') do (
        set "FOUND_CKPT=%%i"
    )
)

set "ZIP_FILE=difflocks_checkpoints.zip"

if not defined FOUND_CKPT (
    if exist "%ZIP_FILE%" (
        echo 📦 Found %ZIP_FILE%. Unzipping...
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '.' -Force"
        
        :: Re-check after unzip
        for /f "delims=" %%i in ('dir /s /b "checkpoints\scalp_*.pth" 2^>nul') do (
            set "FOUND_CKPT=%%i"
        )
        
        if not defined FOUND_CKPT (
            :: Handle case where zip doesn't have a 'checkpoints' folder
            if exist "scalp_*.pth" (
                if not exist "checkpoints" mkdir checkpoints
                move "scalp_*.pth" "checkpoints\"
                if exist "strand_vae" move "strand_vae" "checkpoints\"
                set "FOUND_CKPT=checkpoints\found"
            )
        )
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

:: Check for GPU support (More robust check for Windows)
echo 🔍 Checking for GPU support...
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Warning: GPU support not detected in Docker.
    echo.
    echo 💡 To enable GPU (NVIDIA):
    echo 1. Run 'wsl --update' in PowerShell.
    echo 2. Ensure 'Use the WSL 2 based engine' is enabled in Docker Desktop Settings.
    echo 3. Restart Docker Desktop.
    echo.
    echo Running in CPU mode...
) else (
    echo ✅ GPU support detected!
)
echo.

echo 🚀 Starting Docker containers...
echo 🌐 Once ready, open http://localhost:7860 in your browser.
echo.

docker-compose up

if %errorlevel% neq 0 (
    echo.
    echo ❌ Docker Compose failed to start.
    echo Tip: Try running 'docker-compose up --build' if you just updated the code.
    pause
    exit /b 1
)

pause
