@echo off
chcp 65001 >nul
echo ====================================
echo   YOLO Web Toolkit Setup
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not detected. Please install Python 3.8 or higher
    echo [INFO] Download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Detected Python version:
python --version
echo.

REM Check if virtual environment already exists
if exist "venv\" (
    echo [WARNING] Virtual environment already exists
    set /p overwrite="Do you want to recreate the virtual environment? (Y/N): "
    if /i "%overwrite%"=="Y" (
        echo [INFO] Deleting old virtual environment...
        rmdir /s /q venv
    ) else (
        echo [INFO] Skipping virtual environment creation, using existing environment
        goto install_deps
    )
)

REM Create virtual environment
echo [INFO] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created
echo.

:install_deps
REM Activate virtual environment and upgrade pip
echo [INFO] Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing with dependency installation...
) else (
    echo [SUCCESS] Pip upgraded successfully
)
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt file not found
    pause
    exit /b 1
)

REM Install dependencies
echo [INFO] Installing dependencies...
echo [INFO] PyTorch will be installed with CUDA 12.6 support (configured in requirements.txt)
echo [INFO] This may take a few minutes, please wait...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo [INFO] Please check the error messages, some dependencies may need manual installation
    pause
    exit /b 1
)

echo.
echo ====================================
echo   Setup Complete!
echo ====================================
echo.
echo [INFO] Virtual environment created and configured
echo [INFO] To activate virtual environment manually, use:
echo        venv\Scripts\activate
echo.
echo [INFO] Or run start.bat to launch the application
echo.
pause

