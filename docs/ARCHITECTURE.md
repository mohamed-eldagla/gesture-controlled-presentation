# System Architecture

Technical overview of the PowerPoint Gesture Control Assistant architecture.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Core Technologies](#core-technologies)
- [Class Structure](#class-structure)
- [Processing Pipeline](#processing-pipeline)
- [File Structure](#file-structure)

## Overview

The PowerPoint Gesture Control Assistant is built using a modular architecture that separates concerns between computer vision processing, user interface, data management, and presentation control.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface (Tkinter)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   Main   │ │ Settings │ │   User   │ │ Training │  │
│  │   Tab    │ │   Tab    │ │   Mgmt   │ │   Tab    │  │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘  │
└────────┼────────────┼────────────┼────────────┼────────┘
         │            │            │            │
         └────────────┴────────────┴────────────┘
                      │
         ┌────────────▼────────────┐
         │  GestureControlSystem   │
         │    (Core Logic)         │
         └────────┬────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼────┐
│Camera │    │ Face  │    │ Hand   │
│Input  │    │ Auth  │    │Gesture │
└───┬───┘    └───┬───┘    └───┬────┘
    │            │            │
    │      ┌─────▼────┐  ┌────▼────┐
    │      │ DeepFace │  │MediaPipe│
    │      └─────┬────┘  └────┬────┘
    │            │            │
    └────────────┴────────────┘
                 │
         ┌───────▼────────┐
         │  Presentation  │
         │    Control     │
         │ ┌────┐  ┌────┐ │
         │ │COM │  │ Key││ │
         │ │API │  │Sim ││ │
         │ └────┘  └────┘ │
         └────────────────┘
```

## System Components

### 1. User Interface Layer (`GestureControlUI`)

**Responsibilities**:
- Display camera feed with overlays
- Provide controls for system functions
- Manage user interactions
- Display system status

**Key Features**:
- Multi-tab interface (Main, Settings, User Management, Training)
- Real-time camera feed display
- Status indicators
- Configuration controls

### 2. Core Logic Layer (`GestureControlSystem`)

**Responsibilities**:
- Coordinate all system operations
- Manage system state
- Process computer vision data
- Execute presentation controls

**Key Components**:
- MediaPipe integration
- DeepFace integration
- User database management
- Custom gesture handling
- Calibration management

### 3. Computer Vision Layer

**Components**:

#### Hand Gesture Detection (MediaPipe Hands)
- Detects hand landmarks (21 points per hand)
- Tracks hand position and orientation
- Identifies finger positions

#### Face Recognition (DeepFace)
- Extracts facial embeddings
- Performs face matching
- Calculates similarity scores

#### Emotion Detection (DeepFace)
- Analyzes facial expressions
- Classifies emotions (7 categories)
- Triggers auto-pause on negative emotions

### 4. Presentation Control Layer

**Integration Methods**:

#### Primary: PowerPoint COM API (Windows)
- Direct PowerPoint control
- Reliable slide navigation
- Presentation state management

#### Fallback: Keyboard Simulation
- Cross-platform compatibility
- PyAutoGUI integration
- Standard keyboard shortcuts

### 5. Data Persistence Layer

**Storage Components**:

- **users_db.pkl**: Pickled user face embeddings
- **custom_gestures.json**: JSON custom gesture definitions
- **camera_calibration.json**: JSON camera calibration data
- **gesture_control.log**: Text application logs

## Data Flow

### Authentication Flow

```
1. Camera captures frame
   ↓
2. DeepFace detects face
   ↓
3. Extract face embedding (512-dim vector)
   ↓
4. Compare with stored embeddings
   ↓
5. Calculate cosine similarity
   ↓
6. Threshold check (>0.75)
   ↓
7. Authentication success/failure
```

### Gesture Detection Flow

```
1. Camera captures frame
   ↓
2. MediaPipe processes hand landmarks
   ↓
3. Extract 21 landmark coordinates
   ↓
4. Check custom gestures first
   │  ├─ Normalize landmarks
   │  ├─ Compare with templates
   │  └─ Calculate difference
   ↓
5. Check built-in gestures
   │  ├─ Analyze finger extensions
   │  ├─ Calculate finger states
   │  └─ Classify gesture
   ↓
6. Apply cooldown check
   ↓
7. Execute gesture action
   ↓
8. Provide feedback (voice/visual)
```

### Emotion Detection Flow

```
1. Camera captures frame
   ↓
2. DeepFace analyzes emotions
   ↓
3. Get dominant emotion
   ↓
4. Check if negative emotion
   ↓
5. Auto-pause if enabled
   ↓
6. Monitor for recovery
   ↓
7. Auto-resume when positive
```

## Core Technologies

### Computer Vision

#### OpenCV (cv2)
- **Version**: 4.8.0+
- **Usage**:
  - Camera capture (VideoCapture)
  - Image processing
  - Camera calibration
  - Display rendering

#### MediaPipe
- **Version**: 0.10.0+
- **Models Used**:
  - Hands: Hand landmark detection
  - Face Mesh: Facial landmark detection
- **Performance**: Real-time processing

#### DeepFace
- **Version**: 0.0.79+
- **Models Used**:
  - Facenet: Face recognition (128-dim embeddings)
  - Default: Emotion detection
- **Backend**: TensorFlow

### Machine Learning

#### TensorFlow
- **Version**: 2.15.0+
- **Purpose**: Backend for DeepFace models
- **Hardware**: CPU/GPU support

### GUI Framework

#### Tkinter
- **Purpose**: Cross-platform GUI
- **Components**:
  - ttk widgets for modern look
  - Canvas for camera display
  - Notebook for tabs

### System Integration

#### PyAutoGUI
- **Purpose**: Keyboard simulation
- **Actions**: Slide navigation, presentation control

#### pyttsx3
- **Purpose**: Text-to-speech feedback
- **Features**: Voice announcements

#### comtypes (Windows)
- **Purpose**: PowerPoint COM API
- **Platform**: Windows only

## Class Structure

### GestureControlSystem

```python
class GestureControlSystem:
    # MediaPipe components
    mp_hands: MediaPipe Hands solution
    mp_face_mesh: MediaPipe Face Mesh solution

    # Recognition components
    hands: Hands detector instance
    face_mesh: Face Mesh detector instance

    # State management
    recognized_user: str | None
    is_authenticated: bool
    is_presentation_active: bool
    current_gesture: str

    # Data storage
    users_db: dict[str, dict]
    custom_gestures: dict[str, dict]
    calibration_data: dict | None

    # Configuration
    cooldown_period: float
    emotion_state: str
```

### GestureControlUI

```python
class GestureControlUI:
    # Core components
    master: tk.Tk
    gc_system: GestureControlSystem

    # UI elements
    tab_control: ttk.Notebook
    camera_canvas: tk.Canvas
    status_labels: dict[str, ttk.Label]

    # Camera
    cap: cv2.VideoCapture
    current_frame: np.ndarray

    # State
    camera_active: bool
    detection_active: bool
    training_mode: bool
```

## Processing Pipeline

### Main Loop

1. **Camera Capture** (10ms interval)
   - Read frame from webcam
   - Store current frame

2. **Preprocessing**
   - Apply calibration if available
   - Convert color space (BGR to RGB)

3. **Detection** (if active)
   - Hand gesture detection
   - Emotion detection (if enabled)

4. **Action Execution**
   - Cooldown check
   - Execute gesture action
   - Provide feedback

5. **Display Update**
   - Draw overlays
   - Update canvas
   - Update status displays

### Threading Model

- **Main Thread**: UI and camera display
- **Worker Threads**:
  - Voice feedback (non-blocking)
  - Camera calibration (background)
  - Model downloads (first-time)

## File Structure

```
powerpoint-gesture-control/
├── main.py                      # Main application entry point
│   ├── GestureControlSystem     # Core logic class
│   └── GestureControlUI         # GUI class
│
├── Data Files (generated)
│   ├── users_db.pkl             # User face database
│   ├── custom_gestures.json    # Custom gestures
│   ├── camera_calibration.json # Camera calibration
│   └── gesture_control.log     # Application logs
│
├── Documentation
│   ├── README.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   └── docs/
│       ├── INSTALLATION.md
│       ├── USER_GUIDE.md
│       └── ARCHITECTURE.md
│
└── Configuration
    ├── requirements.txt         # Python dependencies
    └── .gitignore              # Git ignore patterns
```

## Performance Considerations

### Processing Bottlenecks

1. **Face Recognition**: CPU-intensive (1-2s per frame)
2. **Emotion Detection**: GPU-accelerated if available
3. **Hand Tracking**: Real-time (60+ FPS capable)

### Optimization Strategies

1. **Cooldown Period**: Prevents excessive processing
2. **Single Hand Tracking**: Reduces computational load
3. **Conditional Emotion Detection**: Only when enabled
4. **Frame Skipping**: Can be implemented if needed

### Memory Management

- **User Database**: Grows with registered users (≈1MB per user)
- **Model Cache**: DeepFace models (≈100MB)
- **Video Frames**: Cyclic buffer, minimal memory

## Security Considerations

### Data Privacy

- **Local Storage**: All data stored locally
- **No Cloud Upload**: No external data transmission
- **Encryption**: Consider encrypting user_db.pkl for production

### Access Control

- **Face Authentication**: Required before gesture control
- **User Isolation**: Each user has separate profile
- **Session Management**: Authentication state tracked

## Extensibility

### Adding New Gestures

1. Implement detection logic in `classify_gesture()`
2. Add action handler in `execute_gesture_action()`
3. Update documentation

### Adding New Actions

1. Create action method (e.g., `pause_presentation()`)
2. Link to gesture in `execute_gesture_action()`
3. Add PowerPoint COM API implementation

### Platform Support

Current: Windows (full), macOS/Linux (partial)

To add full macOS support:
- Implement AppleScript integration
- Replace comtypes with platform-specific APIs

## Future Architecture Enhancements

1. **Modular Design**: Split into separate modules
   - `vision/`: Computer vision components
   - `ui/`: User interface
   - `control/`: Presentation control
   - `data/`: Data management

2. **Plugin System**: Allow custom gesture plugins

3. **Web Interface**: Browser-based control panel

4. **Cloud Sync**: Optional cloud backup for user profiles

5. **Mobile Companion**: Remote control app

---

For implementation details, see the source code in [main.py](../main.py).
