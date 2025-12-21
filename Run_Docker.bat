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

:: Check for NVIDIA Container Toolkit (Optional but recommended for GPU)
docker info | findstr /C:"Runtimes: nvidia" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: NVIDIA Container Toolkit not detected. 
    echo If you have an NVIDIA GPU, please install it for 100x faster generation.
    echo Running in CPU mode by default...
    echo.
)

:: Check for checkpoints
if not exist "checkpoints\scalp_diffusion.pth" (
    echo ❌ Error: Checkpoints not found!
    echo Please download them manually and place them in the 'checkpoints' folder.
    echo Refer to README.md for instructions.
    pause
    exit /b 1
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
