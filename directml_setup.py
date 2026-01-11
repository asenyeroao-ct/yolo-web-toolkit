"""
YOLO Web Toolkit Environment Setup Script (DirectML Version)
Automatically creates virtual environment and installs all dependencies (including DirectML version of PyTorch)
For systems without NVIDIA GPU (uses DirectML for GPU acceleration)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_info(msg):
    """Print info message"""
    print(f"[INFO] {msg}")


def print_error(msg):
    """Print error message"""
    print(f"[ERROR] {msg}")


def print_success(msg):
    """Print success message"""
    print(f"[SUCCESS] {msg}")


def print_warning(msg):
    """Print warning message"""
    print(f"[WARNING] {msg}")


def check_python():
    """Check if Python is installed"""
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_info(f"Detected Python version: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    
    print_error("Python not detected. Please install Python 3.8 or higher")
    print_info("Download Python from https://www.python.org/downloads/")
    return False


def create_venv():
    """Create virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Do you want to recreate the virtual environment? (Y/N): ")
        if response.upper() == "Y":
            print_info("Deleting old virtual environment...")
            shutil.rmtree(venv_path)
        else:
            print_info("Skipping virtual environment creation, using existing environment")
            return True
    
    print_info("Creating virtual environment...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            check=True
        )
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to create virtual environment")
        return False


def get_pip_command():
    """Get pip command path"""
    if sys.platform == "win32":
        return str(Path("venv/Scripts/pip.exe"))
    else:
        return str(Path("venv/bin/pip"))


def get_python_command():
    """Get Python command path"""
    if sys.platform == "win32":
        return str(Path("venv/Scripts/python.exe"))
    else:
        return str(Path("venv/bin/python"))


def upgrade_pip():
    """Upgrade pip"""
    print_info("Upgrading pip...")
    python_cmd = get_python_command()
    
    try:
        # Use python -m pip instead of pip.exe to avoid permission issues
        subprocess.run(
            [python_cmd, "-m", "pip", "install", "--upgrade", "pip"],
            check=True
        )
        print_success("Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print_warning("Failed to upgrade pip, continuing with dependency installation...")
        return False


def install_pytorch_directml():
    """Install PyTorch DirectML version"""
    print_info("Installing PyTorch (DirectML)...")
    print_info("This may take a few minutes, please wait...")
    print()
    
    python_cmd = get_python_command()
    
    # Install PyTorch DirectML related packages
    # DirectML version of PyTorch uses standard PyPI source
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "torch-directml"
    ]
    
    print_info(f"Installing packages: {', '.join(packages)}")
    print_info("Downloading and installing (this may take a while)...")
    print()
    
    try:
        # Use python -m pip and show real-time output
        result = subprocess.run(
            [python_cmd, "-m", "pip", "install"] + packages,
            check=True
        )
        print()
        print_success("PyTorch (DirectML) installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print_error("Failed to install PyTorch DirectML")
        return False


def install_other_dependencies():
    """Install other dependencies"""
    print_info("Installing other dependencies...")
    print()
    
    python_cmd = get_python_command()
    
    # Other dependency packages
    packages = [
        "flask==3.0.0",
        "flask-cors==4.0.0",
        "werkzeug==3.0.1",
        "ultralytics>=8.0.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0"
    ]
    
    # Note: DirectML version does not include TensorRT and PyCUDA (these require NVIDIA GPU)
    # If users need these features, they should use CUDA version
    
    print_info(f"Installing {len(packages)} packages...")
    print_info("Downloading and installing (this may take a while)...")
    print()
    
    try:
        # Use python -m pip and show real-time output
        result = subprocess.run(
            [python_cmd, "-m", "pip", "install"] + packages,
            check=True
        )
        print()
        print_success("Other dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print_error("Failed to install some dependencies")
        print_warning("Please check the error messages above")
        return False


def run_command(cmd):
    """Execute command and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def verify_installation():
    """Verify installation"""
    print()
    print_info("Verifying installation...")
    
    python_cmd = get_python_command()
    
    # Check PyTorch and DirectML
    check_script = """
import torch
print(f"PyTorch version: {torch.__version__}")

# Check DirectML support
try:
    import torch_directml
    print(f"DirectML available: True")
    print(f"DirectML device: {torch_directml.device()}")
except ImportError:
    print("DirectML available: False (torch-directml not installed)")

# Check CUDA (may not be available)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("CUDA not available (expected for DirectML setup)")
"""
    
    try:
        result = subprocess.run(
            [python_cmd, "-c", check_script],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print_warning(f"Could not verify installation: {e}")


def main():
    """Main function"""
    print("=" * 60)
    print("  YOLO Web Toolkit Setup (DirectML)")
    print("=" * 60)
    print()
    print_info("DirectML setup is for systems without NVIDIA GPU")
    print_info("It uses DirectML for GPU acceleration on Windows")
    print()
    
    # Check Python
    if not check_python():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print()
    
    # Create virtual environment
    if not create_venv():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print()
    
    # Upgrade pip
    upgrade_pip()
    print()
    
    # Install PyTorch (DirectML)
    if not install_pytorch_directml():
        print_warning("PyTorch DirectML installation failed, but continuing...")
    print()
    
    # Install other dependencies
    if not install_other_dependencies():
        print_warning("Some dependencies may not be installed correctly")
    print()
    
    # Verify installation
    verify_installation()
    print()
    
    print("=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print()
    print_info("Virtual environment created and configured (DirectML)")
    print_warning("Note: TensorRT and PyCUDA are not included in DirectML setup")
    print_warning("For full CUDA support, use cuda_setup.py instead")
    print()
    print_info("To activate virtual environment manually, use:")
    if sys.platform == "win32":
        print("        venv\\Scripts\\activate")
    else:
        print("        source venv/bin/activate")
    print()
    print_info("Or run start.bat to launch the application")
    print()
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
