# User Guide

Complete guide to using the PowerPoint Gesture Control Assistant.

## Table of Contents

- [Getting Started](#getting-started)
- [User Interface Overview](#user-interface-overview)
- [Authentication System](#authentication-system)
- [Gesture Control](#gesture-control)
- [Settings Configuration](#settings-configuration)
- [Custom Gestures](#custom-gestures)
- [Camera Calibration](#camera-calibration)
- [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### First Time Setup

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Register Your Face**
   - Navigate to the "User Management" tab
   - Click "Register New User"
   - Enter your name
   - Position your face clearly in the camera view
   - Click OK to save

3. **Authenticate**
   - Return to the "Main Control" tab
   - Click "Authenticate"
   - Wait for recognition confirmation

4. **Start Using Gestures**
   - Open your PowerPoint presentation
   - Click "Start Detection"
   - Use hand gestures to control your presentation

## User Interface Overview

### Main Control Tab

The main control tab is your primary interface:

- **Camera View**: Real-time video feed with gesture detection
- **Status Panel**: Shows current user, gesture, presentation state, and emotion
- **Control Buttons**:
  - Start/Stop Detection
  - Authenticate
  - Connect PowerPoint
  - Calibrate Camera
  - Help

### Settings Tab

Configure application behavior:

- **Gesture Cooldown**: Delay between gesture recognitions (0.5-5.0 seconds)
- **Voice Feedback**: Enable/disable audio announcements
- **Emotion Detection**: Toggle emotion monitoring
- **Auto-pause on Confusion**: Automatically pause when negative emotions detected

### User Management Tab

Manage registered users:

- **User List**: View all registered users
- **Register New User**: Add new face profiles
- **Delete User**: Remove user profiles
- **Refresh**: Update the user list

### Gesture Training Tab

Create custom gestures:

- **Gesture List**: View saved custom gestures
- **Gesture Name**: Name your custom gesture
- **Action**: Associate an action (next, previous, start, end)
- **Training Controls**: Capture and manage gestures

## Authentication System

### Registering Users

1. **Preparation**:
   - Ensure good lighting
   - Remove glasses if possible (or register with and without)
   - Look directly at the camera

2. **Registration**:
   - Click "Register New User"
   - Enter a unique username
   - Capture your face when the image is clear

3. **Best Practices**:
   - Register in similar lighting to where you'll present
   - Update registration if you change appearance significantly
   - Register multiple profiles for different conditions

### Authentication

1. **Manual Authentication**:
   - Click "Authenticate" button
   - System matches your face against registered users
   - Success message displays your username

2. **What Happens During Authentication**:
   - Face detection in the camera frame
   - Feature extraction using DeepFace
   - Comparison with stored profiles
   - Similarity threshold check (>75% match required)

### Troubleshooting Authentication

**Face Not Recognized**:
- Improve lighting conditions
- Move closer to the camera
- Remove obstructions (hands, objects)
- Re-register in current conditions

**Wrong User Recognized**:
- Increase distance from camera
- Adjust lighting
- Consider re-registering both users

## Gesture Control

### Built-in Gestures

#### 1. Start Presentation
- **Gesture**: Open palm (all fingers extended)
- **Action**: Starts the slideshow (F5)
- **Use Case**: Begin your presentation

#### 2. Next Slide
- **Gesture**: Index finger pointing up
- **Action**: Advances to next slide
- **Use Case**: Progress through presentation

#### 3. Previous Slide
- **Gesture**: Pinky finger extended
- **Action**: Returns to previous slide
- **Use Case**: Review previous content

#### 4. End Presentation
- **Gesture**: Closed fist
- **Action**: Exits the slideshow (ESC)
- **Use Case**: Finish presentation

### Using Gestures Effectively

1. **Start Detection**:
   - Click "Start Detection" button
   - Wait for system to initialize

2. **Perform Gestures**:
   - Hold gesture clearly for 1-2 seconds
   - Keep hand within camera frame
   - Maintain steady position
   - Wait for visual/audio feedback

3. **Gesture Tips**:
   - Hold gestures steady (avoid quick movements)
   - Position hand 1-2 feet from camera
   - Use gestures against contrasting background
   - Respect cooldown period between gestures

### Gesture Detection States

- **None**: No hand detected
- **Unknown Gesture**: Hand detected but gesture not recognized
- **[Gesture Name]**: Recognized gesture being executed

## Settings Configuration

### Gesture Cooldown Period

Controls how quickly gestures can be triggered:

- **Lower (0.5-1.0s)**: Fast response, may trigger accidentally
- **Medium (1.5-2.0s)**: Balanced, recommended for most users
- **Higher (2.5-5.0s)**: Slower, more deliberate control

**Recommendation**: Start with 1.5s and adjust based on preference

### Voice Feedback

Audio announcements when gestures are detected:

- **Enabled**: Hear gesture confirmations
- **Disabled**: Silent operation

**Use Cases**:
- Enable for training and testing
- Disable during actual presentations

### Emotion Detection

Monitors presenter's facial expressions:

- **Happy, Neutral**: Normal operation
- **Sad, Angry, Fear**: Triggers auto-pause (if enabled)
- **Surprise**: No action

**Benefits**:
- Automatically pause if presenter looks confused
- Resume when ready
- Improve presentation flow

### Auto-pause on Confusion

Automatically pause presentation when negative emotions detected:

- Monitors for: angry, fear, disgust, sad
- Pauses presentation automatically
- Resumes when emotion returns to neutral/positive

## Custom Gestures

### Creating Custom Gestures

1. **Navigate to Gesture Training Tab**

2. **Enter Gesture Details**:
   - Name: Descriptive name (e.g., "Peace Sign")
   - Action: Select from dropdown (next, previous, start, end)

3. **Start Training**:
   - Click "Start Training"
   - Position your hand in the desired gesture
   - Ensure hand is clearly visible

4. **Capture Gesture**:
   - Click "Capture Gesture" when ready
   - System saves the hand pose
   - Confirmation message appears

5. **Test Your Gesture**:
   - Return to Main Control tab
   - Start Detection
   - Perform your custom gesture

### Custom Gesture Tips

- **Unique Gestures**: Create distinct hand positions
- **Comfortable**: Use gestures you can hold comfortably
- **Consistent**: Practice to ensure reliable detection
- **Lighting**: Train in similar lighting to presentation environment

### Managing Custom Gestures

- **View**: See all custom gestures in the list
- **Delete**: Select and delete unwanted gestures
- **Refresh**: Update the gesture list
- **Overwrite**: Create new gesture with same name to replace

## Camera Calibration

### Why Calibrate?

Camera calibration improves:
- Gesture detection accuracy
- Hand landmark precision
- Overall system reliability

### Calibration Process

1. **Preparation**:
   - Print a chessboard calibration pattern (7x4 corners)
   - Download from: [OpenCV Calibration Patterns](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)

2. **Run Calibration**:
   - Click "Calibrate Camera"
   - Confirm to proceed
   - System captures 15 images

3. **During Calibration**:
   - Hold chessboard pattern in view
   - Move pattern to different positions/angles
   - Tilt and rotate for variety
   - Keep pattern flat and fully visible

4. **Completion**:
   - System processes images
   - Calibration data saved automatically
   - Confirmation message displayed

### Calibration Tips

- Use good lighting
- Keep pattern flat
- Cover different areas of the frame
- Avoid motion blur
- Press 'q' to cancel if needed

## Tips and Best Practices

### Optimal Setup

1. **Lighting**:
   - Face light source (avoid backlighting)
   - Consistent, diffused lighting
   - Avoid harsh shadows

2. **Camera Position**:
   - Eye level or slightly above
   - 2-4 feet away
   - Stable mount or placement

3. **Background**:
   - Plain, contrasting background
   - Minimal movement
   - Good contrast with hand/face

### Presentation Tips

1. **Before Presentation**:
   - Test all gestures
   - Adjust cooldown period
   - Verify authentication works
   - Close unnecessary applications

2. **During Presentation**:
   - Stay within camera view
   - Use deliberate gestures
   - Keep gestures clear and distinct
   - Have keyboard backup ready

3. **Practice**:
   - Practice gestures beforehand
   - Learn the timing
   - Test with your actual presentation
   - Have a backup plan

### Performance Optimization

1. **Close Background Apps**: Free up CPU/GPU resources
2. **Good Lighting**: Reduces processing load
3. **Lower Resolution**: If system is slow, reduce camera resolution
4. **Disable Emotion Detection**: If not needed, turn off to save resources

### Troubleshooting

**Gestures Not Detected**:
- Check hand is clearly visible
- Improve lighting
- Verify detection is started
- Try recalibrating camera

**Slow Performance**:
- Close other applications
- Reduce camera resolution
- Disable emotion detection
- Check CPU usage

**False Gesture Triggers**:
- Increase cooldown period
- Improve lighting
- Use more distinct gestures
- Keep hand out of frame when not gesturing

## Keyboard Shortcuts

While the system uses gestures, keyboard shortcuts still work:

- **F5**: Start presentation
- **Right Arrow**: Next slide
- **Left Arrow**: Previous slide
- **ESC**: End presentation
- **Space**: Pause/Resume

## Getting Help

- Click "Help" button in the application
- Check the documentation
- Review log file: `gesture_control.log`
- Open GitHub issue for bugs

## Advanced Features

### Log File Analysis

Location: `gesture_control.log`

Review logs for:
- Authentication attempts
- Gesture detections
- Error messages
- System events

### Data Files

Generated files:
- `users_db.pkl`: User face database
- `custom_gestures.json`: Custom gesture definitions
- `camera_calibration.json`: Calibration data
- `gesture_control.log`: Application logs

**Backup**: Regularly backup these files to preserve your configurations

## Safety and Privacy

- Face data stored locally (not sent to cloud)
- No video recording
- Data files can be deleted anytime
- Camera only active when application running

---

For additional help, see:
- [Installation Guide](INSTALLATION.md)
- [API Documentation](API.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
