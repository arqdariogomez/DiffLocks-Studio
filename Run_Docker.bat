@echo off
setlocal enabledelayedexpansion

echo DiffLocks Studio - Docker Launcher (DEBUG MODE)
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
echo (This might take a few seconds)
echo.

:: Run the test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo.
echo [DEBUG] GPU test finished. ErrorLevel: %errorlevel%
echo Press any key to continue to Docker Compose...
pause

echo.
echo Starting Docker containers...
docker compose up

if %errorlevel% neq 0 (
    echo.
    echo Docker Compose failed.
    pause
    exit /b 1
)

pause
