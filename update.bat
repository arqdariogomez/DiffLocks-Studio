@echo off
echo ==========================================
echo DIFFLOCKS STUDIO UPDATER
echo ==========================================

:: 1. Check if GIT is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] GIT not found on your system.
    echo Please install Git for Windows from: https://git-scm.com/
    pause
    exit
)

:: 2. Go to local directory
if exist "C:\difflocks_local" (
    cd /d "C:\difflocks_local"
) else (
    echo [INFO] Directory does not exist, creating it...
    mkdir "C:\difflocks_local"
    cd /d "C:\difflocks_local"
)

:: 3. Check if it is already a Git repository; if not, initialize it
if not exist .git (
    echo [INFO] No Git repository detected. Initializing...
    git init
    git remote add origin https://github.com/arqdariogomez/DiffLocks-Studio.git
) else (
    echo [INFO] Repository detected.
    :: Ensuring the origin is correct just in case
    git remote set-url origin https://github.com/arqdariogomez/DiffLocks-Studio.git
)

:: 4. Download and force update
echo.
echo [INFO] Downloading the latest version from GitHub...
git fetch origin

echo [INFO] Applying changes (this will overwrite local changes)...
:: Try 'main' branch first (current standard)
git reset --hard origin/main

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] It seems the main branch is not 'main', trying 'master'...
    git reset --hard origin/master
)

echo.
echo ==========================================
echo UPDATE COMPLETED SUCCESSFULLY
echo ==========================================
pause