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

echo [INFO] Upgrading pip...
python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing with setup...
) else (
    echo [SUCCESS] Pip upgraded successfully
)
echo.

echo [INFO] Please select setup type:
echo.
echo   1. Setup CUDA (for NVIDIA GPU)
echo   2. Setup DirectML (for systems without NVIDIA GPU)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" goto setup_cuda
if "%choice%"=="2" goto setup_directml

echo [ERROR] Invalid choice. Please enter 1 or 2.
pause
exit /b 1

:setup_cuda
echo.
echo [INFO] Running CUDA setup...
echo [INFO] PyTorch will be installed with CUDA 12.6 support
echo [INFO] This may take a few minutes, please wait...
echo.

REM Check if cuda_setup.py exists
if not exist "cuda_setup.py" (
    echo [ERROR] cuda_setup.py file not found
    echo [INFO] Please ensure cuda_setup.py is in the current directory
    pause
    exit /b 1
)

REM Run cuda_setup.py
python cuda_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] CUDA setup failed
    echo [INFO] Please check the error messages above
    pause
    exit /b 1
)

echo.
echo [INFO] CUDA setup completed successfully!
pause
exit /b 0

:setup_directml
echo.
echo [INFO] Running DirectML setup...
echo [INFO] PyTorch will be installed with DirectML support
echo [INFO] This is for systems without NVIDIA GPU
echo [INFO] This may take a few minutes, please wait...
echo.

REM Check if directml_setup.py exists
if not exist "directml_setup.py" (
    echo [ERROR] directml_setup.py file not found
    echo [INFO] Please ensure directml_setup.py is in the current directory
    pause
    exit /b 1
)

REM Run directml_setup.py
python directml_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] DirectML setup failed
    echo [INFO] Please check the error messages above
    pause
    exit /b 1
)

echo.
echo [INFO] DirectML setup completed successfully!
pause
exit /b 0

