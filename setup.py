"""
Setup script for PowerPoint Gesture Control Assistant

This script helps with the initial setup of the application.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    print("\nCreating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")

    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:
        pip_path = os.path.join("venv", "bin", "pip")

    try:
        # Upgrade pip
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("✓ Pip upgraded")

        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print("\nVerifying installation...")

    if sys.platform == "win32":
        python_path = os.path.join("venv", "Scripts", "python")
    else:
        python_path = os.path.join("venv", "bin", "python")

    test_code = """
import cv2
import mediapipe
import numpy
import PIL
print("✓ All core packages imported successfully")
"""

    try:
        subprocess.run([python_path, "-c", test_code], check=True)
        return True
    except subprocess.CalledProcessError:
        print("✗ Package verification failed")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   Windows (CMD):     venv\\Scripts\\activate")
        print("   Windows (PS):      venv\\Scripts\\Activate.ps1")
    else:
        print("   macOS/Linux:       source venv/bin/activate")

    print("\n2. Run the application:")
    print("   python main.py")

    print("\n3. First time usage:")
    print("   - Register your face in the 'User Management' tab")
    print("   - Click 'Authenticate' to verify")
    print("   - Click 'Start Detection' to begin")

    print("\nFor more information, see:")
    print("   - README.md")
    print("   - docs/INSTALLATION.md")
    print("   - docs/USER_GUIDE.md")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("="*60)
    print("PowerPoint Gesture Control Assistant - Setup")
    print("="*60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create virtual environment
    if os.path.exists("venv"):
        response = input("\nVirtual environment already exists. Recreate? (y/N): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree("venv")
            if not create_virtual_environment():
                sys.exit(1)
    else:
        if not create_virtual_environment():
            sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("\nWarning: Package verification failed. You may need to install some packages manually.")

    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
