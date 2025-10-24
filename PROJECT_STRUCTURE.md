# Project Structure

Complete overview of the PowerPoint Gesture Control Assistant repository structure.

## Directory Tree

```
powerpoint-gesture-control/
│
├── .github/                          # GitHub-specific files
│   ├── ISSUE_TEMPLATE/               # Issue templates
│   │   ├── bug_report.md             # Bug report template
│   │   └── feature_request.md        # Feature request template
│   └── pull_request_template.md      # Pull request template
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System architecture overview
│   ├── INSTALLATION.md               # Detailed installation guide
│   └── USER_GUIDE.md                 # Complete user manual
│
├── .gitignore                        # Git ignore patterns
├── CHANGELOG.md                      # Version history and changes
├── CONTRIBUTING.md                   # Contributing guidelines
├── LICENSE                           # MIT License
├── QUICKSTART.md                     # Quick start guide
├── README.md                         # Main project documentation
│
├── main.py                           # Main application file
├── requirements.txt                  # Python dependencies
└── setup.py                          # Automated setup script
```

## Generated Files (Not in Git)

These files are generated during runtime and excluded from version control:

```
powerpoint-gesture-control/
│
├── venv/                             # Virtual environment (created by setup)
│
├── .deepface/                        # DeepFace model cache
│   └── weights/                      # Pre-trained model weights
│
├── gesture_control.log               # Application log file
├── users_db.pkl                      # User face database (pickled)
├── custom_gestures.json              # Custom gesture definitions
└── camera_calibration.json           # Camera calibration data
```

## File Descriptions

### Root Level Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Main project documentation | Everyone |
| `QUICKSTART.md` | 5-minute quick start guide | New users |
| `CONTRIBUTING.md` | Contribution guidelines | Contributors |
| `CHANGELOG.md` | Version history | Users, developers |
| `LICENSE` | MIT License | Everyone |
| `main.py` | Main application code | Developers |
| `requirements.txt` | Python dependencies | Installation |
| `setup.py` | Automated setup script | Installation |
| `.gitignore` | Git ignore patterns | Git |

### Documentation (`docs/`)

| File | Purpose | Audience |
|------|---------|----------|
| `INSTALLATION.md` | Detailed installation steps | Users |
| `USER_GUIDE.md` | Complete user manual | Users |
| `ARCHITECTURE.md` | Technical architecture | Developers |

### GitHub Templates (`.github/`)

| File | Purpose | Audience |
|------|---------|----------|
| `bug_report.md` | Bug report template | Issue reporters |
| `feature_request.md` | Feature request template | Feature suggesters |
| `pull_request_template.md` | PR template | Contributors |

## Code Structure (`main.py`)

```python
main.py
├── Module docstring                  # Project overview
├── Imports                           # Dependencies
├── Logging setup                     # Logger configuration
│
├── GestureControlSystem              # Core logic class
│   ├── __init__()                    # Initialize components
│   ├── load_user_data()              # User database loading
│   ├── save_user_data()              # User database saving
│   ├── register_new_user()           # User registration
│   ├── authenticate_user()           # Face authentication
│   ├── calculate_similarity()        # Embedding comparison
│   ├── detect_gestures()             # Hand gesture detection
│   ├── match_custom_gesture()        # Custom gesture matching
│   ├── normalize_landmarks()         # Landmark normalization
│   ├── classify_gesture()            # Built-in gesture classification
│   ├── execute_gesture_action()      # Action execution
│   ├── next_slide()                  # Next slide control
│   ├── previous_slide()              # Previous slide control
│   ├── start_presentation()          # Start presentation
│   ├── end_presentation()            # End presentation
│   ├── speak_feedback()              # Voice feedback
│   ├── detect_emotion()              # Emotion detection
│   ├── calibrate_camera()            # Camera calibration
│   ├── save_calibration_data()       # Save calibration
│   ├── load_calibration_data()       # Load calibration
│   ├── undistort_image()             # Apply calibration
│   ├── connect_to_powerpoint()       # PowerPoint COM connection
│   ├── save_custom_gesture()         # Save custom gesture
│   ├── save_custom_gestures()        # Save all gestures
│   ├── load_custom_gestures()        # Load gestures
│   └── remove_custom_gesture()       # Delete gesture
│
├── GestureControlUI                  # GUI class
│   ├── __init__()                    # Initialize UI
│   ├── setup_main_tab()              # Main control tab
│   ├── setup_settings_tab()          # Settings tab
│   ├── setup_user_tab()              # User management tab
│   ├── setup_training_tab()          # Gesture training tab
│   ├── setup_camera()                # Camera initialization
│   ├── update_camera()               # Camera feed update loop
│   ├── toggle_detection()            # Start/stop detection
│   ├── authenticate_user()           # UI authentication handler
│   ├── try_connect_powerpoint()      # PowerPoint connection
│   ├── start_calibration()           # Calibration UI
│   ├── run_calibration()             # Calibration execution
│   ├── apply_settings()              # Apply settings
│   ├── reset_settings()              # Reset to defaults
│   ├── register_new_user()           # User registration UI
│   ├── delete_user()                 # Delete user UI
│   ├── refresh_user_list()           # Update user list
│   ├── start_gesture_training()      # Training mode
│   ├── capture_gesture()             # Capture custom gesture
│   ├── delete_gesture()              # Delete custom gesture
│   ├── refresh_gesture_list()        # Update gesture list
│   ├── show_help()                   # Display help
│   └── on_closing()                  # Cleanup on exit
│
├── main()                            # Entry point
└── if __name__ == "__main__"         # Script execution
```

## Data Files

### User Database (`users_db.pkl`)

```python
{
    "username": {
        "embedding": [512-dimensional vector],
        "timestamp": 1234567890.123
    }
}
```

### Custom Gestures (`custom_gestures.json`)

```json
{
    "gesture_name": {
        "landmarks": [[x, y, z], ...],  // 21 normalized landmarks
        "action": "next",
        "timestamp": 1234567890.123
    }
}
```

### Camera Calibration (`camera_calibration.json`)

```json
{
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion_coefficients": [k1, k2, p1, p2, k3],
    "success": true
}
```

## Dependencies (`requirements.txt`)

### Core Dependencies
- **opencv-python**: Computer vision and camera handling
- **mediapipe**: Hand and face landmark detection
- **deepface**: Face recognition and emotion detection
- **tensorflow**: Deep learning backend
- **numpy**: Numerical operations

### UI and System Integration
- **Pillow**: Image processing for Tkinter
- **pyautogui**: Keyboard simulation
- **pyttsx3**: Text-to-speech
- **comtypes**: PowerPoint COM API (Windows only)

## Size Estimates

| Component | Approximate Size |
|-----------|------------------|
| Source code | ~60 KB |
| Documentation | ~50 KB |
| Dependencies (installed) | ~2 GB |
| User database (per user) | ~1 MB |
| DeepFace models (cached) | ~100 MB |
| Log file (per session) | ~1 MB |

## File Permissions

### Readable by Application
- All `.py`, `.md`, `.txt`, `.json` files
- User data files (`.pkl`, `.json`)
- Configuration files

### Writable by Application
- Log file (`gesture_control.log`)
- User database (`users_db.pkl`)
- Custom gestures (`custom_gestures.json`)
- Calibration data (`camera_calibration.json`)

### Executable
- `main.py` (with Python interpreter)
- `setup.py` (with Python interpreter)

## Version Control

### Tracked Files
- All source code (`.py`)
- Documentation (`.md`)
- Configuration (`requirements.txt`, `.gitignore`)
- License and templates

### Ignored Files (`.gitignore`)
- Virtual environment (`venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- User data files (`*.pkl`, `*.json`)
- Log files (`*.log`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Model cache (`.deepface/`)

## Development Setup

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd powerpoint-gesture-control
   ```

2. **Setup environment**
   ```bash
   python setup.py
   # or manually:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Run application**
   ```bash
   python main.py
   ```

## Production Deployment

### Minimal Required Files
```
main.py
requirements.txt
```

### Recommended Files
```
main.py
requirements.txt
README.md
LICENSE
docs/
```

### Optional for Users
```
setup.py
QUICKSTART.md
.github/
```

## Modular Future Structure (Proposed)

For future versions, consider this modular structure:

```
powerpoint-gesture-control/
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── vision/                   # Computer vision module
│   │   ├── __init__.py
│   │   ├── gesture_detector.py
│   │   ├── face_recognizer.py
│   │   └── emotion_detector.py
│   ├── control/                  # Presentation control
│   │   ├── __init__.py
│   │   ├── powerpoint.py
│   │   └── keyboard.py
│   ├── ui/                       # User interface
│   │   ├── __init__.py
│   │   └── main_window.py
│   └── data/                     # Data management
│       ├── __init__.py
│       └── database.py
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── setup.py                      # Installation script
```

## Navigation Guide

### For Users
Start with: `README.md` → `QUICKSTART.md` → `docs/USER_GUIDE.md`

### For Developers
Start with: `README.md` → `CONTRIBUTING.md` → `docs/ARCHITECTURE.md`

### For Contributors
Start with: `CONTRIBUTING.md` → `docs/ARCHITECTURE.md` → `main.py`

### For Installation
Start with: `QUICKSTART.md` or `docs/INSTALLATION.md`

---

**Note**: This structure is optimized for clarity, maintainability, and ease of contribution. Future versions may evolve based on community feedback and requirements.
