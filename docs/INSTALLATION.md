# Installation Guide

This guide provides detailed installation instructions for the PowerPoint Gesture Control Assistant.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## System Requirements

### Hardware Requirements
- **Webcam**: Built-in or USB webcam (720p or higher recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Processor**: Intel i5 or equivalent (for real-time processing)
- **Disk Space**: 2GB free space for dependencies

### Software Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **Operating System**:
  - Windows 10/11 (full PowerPoint integration)
  - macOS 10.14+ (keyboard simulation only)
  - Linux Ubuntu 18.04+ (keyboard simulation only)
- **PowerPoint**: Microsoft PowerPoint 2016 or later (Windows only, optional)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/powerpoint-gesture-control.git
cd powerpoint-gesture-control

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Detailed Installation

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH"
4. Verify installation:
   ```bash
   python --version
   ```

#### macOS
```bash
# Using Homebrew
brew install python@3.9

# Verify installation
python3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
python3 --version
```

### Step 2: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/yourusername/powerpoint-gesture-control.git

# Or using SSH
git clone git@github.com:yourusername/powerpoint-gesture-control.git

cd powerpoint-gesture-control
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import cv2; import mediapipe; import deepface; print('All modules imported successfully!')"
```

## Platform-Specific Instructions

### Windows

#### PowerPoint COM API Setup
1. Ensure Microsoft PowerPoint is installed
2. The application will automatically connect to PowerPoint
3. If connection fails, the system falls back to keyboard simulation

#### Camera Permissions
1. Go to Settings > Privacy > Camera
2. Enable camera access for Python applications

### macOS

#### Additional Dependencies
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system libraries
brew install cmake
```

#### Camera Permissions
1. Go to System Preferences > Security & Privacy > Camera
2. Allow access for Terminal or your Python IDE

#### Note on PowerPoint
- PowerPoint COM API is not available on macOS
- The system will use keyboard simulation instead
- Ensure accessibility permissions are granted

### Linux

#### Additional System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-tk \
    portaudio19-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    cmake

# Fedora
sudo dnf install python3-tkinter portaudio-devel cmake
```

#### Camera Access
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Reboot or log out and log back in
```

## Troubleshooting

### Common Installation Issues

#### Issue: pip install fails with "No module named pip"
**Solution:**
```bash
python -m ensurepip --upgrade
```

#### Issue: "Permission denied" errors
**Solution:**
```bash
# Don't use sudo with pip in virtual environment
# Make sure virtual environment is activated
pip install --user -r requirements.txt
```

#### Issue: OpenCV import fails
**Solution:**
```bash
# Uninstall and reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python --no-cache-dir
```

#### Issue: MediaPipe installation fails
**Solution:**
```bash
# Install build dependencies first
pip install cmake
pip install mediapipe --no-cache-dir
```

#### Issue: TensorFlow compatibility errors
**Solution:**
```bash
# Install specific TensorFlow version
pip install tensorflow==2.15.0
```

#### Issue: DeepFace models download fails
**Solution:**
- Ensure stable internet connection
- Models will download on first use
- Check firewall settings
- Manually download models if needed

#### Issue: Camera not detected
**Solution:**
1. Check camera is not being used by another application
2. Verify camera permissions
3. Test camera with:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   print(cap.isOpened())
   ```

### Windows-Specific Issues

#### Issue: PowerPoint connection fails
**Solution:**
1. Ensure PowerPoint is installed and updated
2. Run Python as Administrator
3. Check COM security settings
4. The system will automatically fall back to keyboard simulation

### macOS-Specific Issues

#### Issue: "xcrun: error: invalid active developer path"
**Solution:**
```bash
xcode-select --install
```

### Linux-Specific Issues

#### Issue: Tkinter import fails
**Solution:**
```bash
sudo apt-get install python3-tk
```

## Verification

### Test Camera
```python
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### Test MediaPipe
```python
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

### Test DeepFace
```python
python -c "from deepface import DeepFace; print('DeepFace imported successfully')"
```

### Run Full Application
```bash
python main.py
```

## Post-Installation Steps

1. **Register Your Face**
   - Open the application
   - Go to User Management tab
   - Click "Register New User"
   - Follow the prompts

2. **Test Gestures**
   - Click "Authenticate"
   - Click "Start Detection"
   - Try the built-in gestures

3. **Camera Calibration** (Optional but Recommended)
   - Print a chessboard calibration pattern
   - Click "Calibrate Camera"
   - Follow the instructions

## Need Help?

If you encounter issues not covered here:
1. Check the [main README](../README.md)
2. Search [existing issues](https://github.com/yourusername/powerpoint-gesture-control/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

## Next Steps

- Read the [User Guide](USER_GUIDE.md)
- Learn about [Custom Gestures](CUSTOM_GESTURES.md)
- Explore [API Documentation](API.md)
