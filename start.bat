@echo off
chcp 65001 >nul
echo ====================================
echo   YOLO Web Toolkit Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not detected. Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment detected, activating...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [WARNING] Failed to activate virtual environment, using system Python
    ) else (
        echo [SUCCESS] Virtual environment activated
    )
) else (
    echo [INFO] No virtual environment detected, using system Python
    echo [INFO] It is recommended to run setup.bat first to create a virtual environment
)

echo.
echo [INFO] Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Dependencies not installed, installing...
    echo [INFO] PyTorch will be installed with CUDA 12.6 support (configured in requirements.txt)
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo [INFO] Please run setup.bat first to set up the environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Dependencies installed
)

echo.
echo [INFO] Starting server...
echo [INFO] Access http://127.0.0.1:5000 to use the interface
echo [INFO] Press Ctrl+C to stop the server
echo.

python backend\app.py

pause

