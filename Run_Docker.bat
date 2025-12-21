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
docker compose down --remove-orphans
docker rm -f difflocks_studio >nul 2>&1
docker compose up --build

if %errorlevel% neq 0 (
    echo.
    echo Docker Compose failed.
    pause
    exit /b 1
)
