@echo off
setlocal

:: Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker not found! Please install Docker Desktop for Windows.
    pause
    exit /b 1
)

:: Check if Docker is running
docker info >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker is not running! Please start Docker Desktop.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   DiffLocks Studio - BUILD AND RUN
echo ==========================================================
echo.
echo Rebuilding Docker image to include latest code changes...
echo This might take a few minutes depending on cache...
echo.

docker compose up --build -d

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker Compose failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Container rebuilt and started in background.
echo.
echo To see logs, run: docker logs -f difflocks_studio
echo.
echo Waiting for Gradio to be ready (this may take 10-20 seconds)...
timeout /t 15 >nul
start http://localhost:7860

exit /b 0