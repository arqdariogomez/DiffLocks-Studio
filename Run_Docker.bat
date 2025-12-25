@echo off
setlocal enabledelayedexpansion

echo DiffLocks Studio - Docker Launcher
echo ========================================

:: Check for Docker
docker --version
if %errorlevel% neq 0 (
    echo Error: Docker not found.
    pause
    exit /b 1
)

echo.
echo Checking for GPU support...
echo.

:: Run the test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo.
echo Starting Docker containers...
:: Normal start (no build)
echo.
echo Starting container in background...
docker compose up -d
echo Container started.
echo.
echo To see logs, run: docker logs -f difflocks_studio
echo.
echo Waiting for Gradio to be ready...
timeout /t 5 >nul
start http://localhost:7860

if %errorlevel% neq 0 (
    echo.
    echo Docker Compose failed.
    pause
    exit /b 1
)
