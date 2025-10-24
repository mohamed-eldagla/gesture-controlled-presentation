# Quick Start Guide

Get up and running with PowerPoint Gesture Control Assistant in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam
- Windows 10/11 (recommended) or macOS/Linux

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/powerpoint-gesture-control.git
cd powerpoint-gesture-control

# Run setup script
python setup.py
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/powerpoint-gesture-control.git
cd powerpoint-gesture-control

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## First Run

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Register your face** (first time only)
   - Click on "User Management" tab
   - Click "Register New User"
   - Enter your name
   - Look at the camera
   - Click OK

3. **Authenticate**
   - Return to "Main Control" tab
   - Click "Authenticate"
   - Wait for confirmation

4. **Test gestures**
   - Click "Start Detection"
   - Try these gestures:
     - **Open palm** → Should say "Start Presentation"
     - **Point index finger** → Should say "Next Slide"
     - **Extend pinky** → Should say "Previous Slide"
     - **Closed fist** → Should say "End Presentation"

## Using with PowerPoint

1. Open your PowerPoint presentation
2. Authenticate in the app
3. Click "Start Detection"
4. Use gestures to control:
   - **Open palm** → Start slideshow
   - **Index finger** → Next slide
   - **Pinky finger** → Previous slide
   - **Closed fist** → End slideshow

## Built-in Gestures

| Gesture | Action | Use |
|---------|--------|-----|
| ✋ Open palm | Start Presentation | Begin slideshow |
| ☝️ Index finger | Next Slide | Advance slides |
| 🤙 Pinky finger | Previous Slide | Go back |
| ✊ Closed fist | End Presentation | Exit slideshow |

## Tips for Best Results

1. **Lighting**: Ensure your face and hands are well-lit
2. **Distance**: Position yourself 2-4 feet from the camera
3. **Background**: Use a plain background with good contrast
4. **Steady gestures**: Hold gestures for 1-2 seconds
5. **Cooldown**: Wait 1.5 seconds between gestures

## Common Issues

### Camera not detected
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Error')"
```

### Face not recognized
- Improve lighting
- Move closer to camera
- Re-register in current lighting conditions

### Gestures not working
- Click "Start Detection"
- Ensure hand is clearly visible
- Try adjusting lighting
- Increase cooldown period in Settings

## What's Next?

- **Settings**: Adjust cooldown period and voice feedback
- **Custom Gestures**: Create your own gestures in "Gesture Training" tab
- **Camera Calibration**: Print a chessboard pattern and calibrate for better accuracy
- **Documentation**: Read the [full user guide](docs/USER_GUIDE.md)

## Getting Help

- **Documentation**: See [docs/](docs/) folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/powerpoint-gesture-control/issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Video Tutorial

*(Add link to video tutorial when available)*

---

**Enjoy hands-free presentations!** 🎉
