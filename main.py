"""
PowerPoint Gesture Control Assistant

A sophisticated computer vision-based system that enables hands-free control of
PowerPoint presentations using hand gestures, facial recognition authentication,
and real-time emotion detection.

Features:
    - Hand gesture recognition for presentation control
    - Facial recognition authentication using DeepFace
    - Real-time emotion detection with auto-pause capability
    - Custom gesture training and configuration
    - Camera calibration for improved accuracy
    - PowerPoint COM API integration with keyboard fallback

Author: Mohamed Eldagla
License: MIT
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import pickle
import pyautogui
import pyttsx3
from deepface import DeepFace
from comtypes import client
import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import math

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='gesture_control.log')
logger = logging.getLogger('GestureControl')

class GestureControlSystem:
    """
    Core system for gesture recognition, face authentication, and presentation control.

    This class handles all the computer vision processing, user authentication,
    gesture detection, and PowerPoint control functionality.

    Attributes:
        mp_hands: MediaPipe hands solution
        mp_face_mesh: MediaPipe face mesh solution
        recognized_user: Currently authenticated username
        is_authenticated: Authentication status
        is_presentation_active: Whether presentation is currently running
        custom_gestures: Dictionary of user-defined custom gestures
        users_db: Database of registered user face embeddings
    """

    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                        max_num_hands=1, 
                                        min_detection_confidence=0.7,
                                        min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                   max_num_faces=1,
                                                   min_detection_confidence=0.7,
                                                   min_tracking_confidence=0.5)
        
        # Initialize variables
        self.recognized_user = None
        self.current_gesture = "None"
        self.last_gesture_time = time.time()
        self.is_authenticated = False
        self.is_presentation_active = False
        self.calibration_data = None
        self.emotion_state = "Neutral"
        self.is_paused_due_to_confusion = False
        self.custom_gestures = {}
        self.last_action_time = time.time()
        self.cooldown_period = 1.5  # seconds
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Load user data if exists
        self.users_db = {}
        self.load_user_data()
        
        # Load calibration data if exists
        #self.load_calibration_data()
        
        # Load custom gesture data if exists
        self.load_custom_gestures()
        
        # PowerPoint integration
        self.powerpoint_app = None
        
    def load_user_data(self):
        """Load user facial recognition data from disk"""
        if os.path.exists('users_db.pkl'):
            try:
                with open('users_db.pkl', 'rb') as f:
                    self.users_db = pickle.load(f)
                logger.info(f"Loaded {len(self.users_db)} user profiles")
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                self.users_db = {}
    
    def save_user_data(self):
        """Save user facial recognition data to disk"""
        try:
            with open('users_db.pkl', 'wb') as f:
                pickle.dump(self.users_db, f)
            logger.info("User data saved successfully")
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    def register_new_user(self, username, face_image):
        """Register a new user with face image"""
        try:
            # Extract face embedding using DeepFace
            embedding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=True)
            
            # Store user data
            self.users_db[username] = {
                'embedding': embedding,
                'timestamp': time.time()
            }
            
            self.save_user_data()
            return True
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False
    
    def authenticate_user(self, face_image):
        """Authenticate user using face recognition"""
        if not self.users_db:
            logger.warning("No users registered in the system")
            return False, "No users registered"
        
        try:
            # Get face embedding
            new_embedding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=True)
            
            # Compare with stored embeddings
            best_match = None
            best_similarity = -1
            
            for username, data in self.users_db.items():
                stored_embedding = data['embedding']
                
                # Calculate cosine similarity between embeddings
                similarity = self.calculate_similarity(new_embedding[0]["embedding"], 
                                                      stored_embedding[0]["embedding"])
                
                if similarity > 0.75 and similarity > best_similarity:  # Threshold for match
                    best_match = username
                    best_similarity = similarity
            
            if best_match:
                self.recognized_user = best_match
                self.is_authenticated = True
                logger.info(f"User authenticated: {best_match} (similarity: {best_similarity:.2f})")
                return True, best_match
            else:
                logger.warning("Authentication failed - no matching user found")
                return False, "Unknown user"
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False, str(e)
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def detect_gestures(self, image):
        """Detect hand gestures in the image"""
        if not self.is_authenticated:
            return "Not authenticated"
            
        # Check cooldown period to prevent rapid gesture triggering
        if time.time() - self.last_action_time < self.cooldown_period:
            return self.current_gesture
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Get landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Check custom gestures first
            if self.custom_gestures:
                for gesture_name, gesture_data in self.custom_gestures.items():
                    if self.match_custom_gesture(hand_landmarks, gesture_data):
                        if self.current_gesture != gesture_name:
                            self.current_gesture = gesture_name
                            self.last_action_time = time.time()
                            self.execute_gesture_action(gesture_name)
                        return gesture_name
            
            # Check built-in gestures if no custom gesture matched
            gesture = self.classify_gesture(hand_landmarks)
            
            if gesture != self.current_gesture:
                self.current_gesture = gesture
                self.last_action_time = time.time()
                self.execute_gesture_action(gesture)
                
            return gesture
        else:
            if self.current_gesture != "None":
                self.current_gesture = "None"
            return "None"
    
    def match_custom_gesture(self, landmarks, gesture_data):
        """Match hand landmarks against saved custom gesture data"""
        # Extract current landmark positions
        current_positions = []
        for landmark in landmarks.landmark:
            current_positions.append((landmark.x, landmark.y, landmark.z))
        
        # Compare with saved gesture template
        template = gesture_data['landmarks']
        
        # Calculate normalized positions for comparison
        norm_current = self.normalize_landmarks(current_positions)
        
        # Calculate difference between current pose and template
        total_diff = 0
        for i in range(len(norm_current)):
            diff = math.sqrt(sum((a - b)**2 for a, b in zip(norm_current[i], template[i])))
            total_diff += diff
        
        # Return true if the average difference is below threshold
        avg_diff = total_diff / len(norm_current)
        return avg_diff < 0.2  # Threshold for matching
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmark positions relative to wrist and middle finger positions"""
        # Use wrist as origin
        wrist = landmarks[0]
        
        # Find bounding box
        min_x = min(landmarks, key=lambda p: p[0])[0]
        max_x = max(landmarks, key=lambda p: p[0])[0]
        min_y = min(landmarks, key=lambda p: p[1])[1]
        max_y = max(landmarks, key=lambda p: p[1])[1]
        
        # Calculate scale factors
        scale_x = 1.0 / (max_x - min_x) if max_x != min_x else 1.0
        scale_y = 1.0 / (max_y - min_y) if max_y != min_y else 1.0
        
        # Normalize all points
        normalized = []
        for point in landmarks:
            nx = (point[0] - wrist[0]) * scale_x
            ny = (point[1] - wrist[1]) * scale_y
            nz = point[2] - wrist[2]
            normalized.append((nx, ny, nz))
            
        return normalized
        
    def classify_gesture(self, hand_landmarks):
        """Classify the detected hand gesture"""
        # Extract finger tip and base positions
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate if fingers are extended
        index_extended = index_tip.y < index_base.y
        middle_extended = middle_tip.y < middle_base.y
        ring_extended = ring_tip.y < ring_base.y
        pinky_extended = pinky_tip.y < pinky_base.y
        
        # Thumb extended (compare with wrist)
        thumb_extended = thumb_tip.x > wrist.x if wrist.x < 0.5 else thumb_tip.x < wrist.x
        
        # Gesture classification based on extended fingers
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "Start Presentation"
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Next Slide"
        elif pinky_extended and not index_extended and not middle_extended and not ring_extended:
            return "Previous Slide"
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "End Presentation"
        else:
            return "Unknown Gesture"
    
    def execute_gesture_action(self, gesture):
        """Execute the action associated with the detected gesture"""
        if not self.is_authenticated:
            return
            
        # Speak the gesture (voice feedback)
        threading.Thread(target=self.speak_feedback, args=(f"{gesture} activated",)).start()
        
        # Perform the appropriate action based on the gesture
        if gesture == "Next Slide":
            self.next_slide()
        elif gesture == "Previous Slide":
            self.previous_slide()
        elif gesture == "Start Presentation":
            self.start_presentation()
        elif gesture == "End Presentation":
            self.end_presentation()
        
        # Check if any custom gestures actions need to be executed
        if gesture in self.custom_gestures:
            action = self.custom_gestures[gesture].get('action')
            if action == "next":
                self.next_slide()
            elif action == "previous":
                self.previous_slide()
            elif action == "start":
                self.start_presentation()
            elif action == "end":
                self.end_presentation()
                
        logger.info(f"Executed gesture: {gesture}")
    
    def next_slide(self):
        """Navigate to the next slide"""
        if self.is_presentation_active:
            # Try PowerPoint integration first
            if self.powerpoint_app:
                try:
                    presentation = self.powerpoint_app.ActivePresentation
                    slideshow = presentation.SlideShowWindow.View
                    slideshow.Next()
                    return
                except Exception as e:
                    logger.error(f"PowerPoint integration error: {e}")
            
            # Fall back to keyboard simulation
            pyautogui.press('right')
            logger.info("Next slide command executed")
    
    def previous_slide(self):
        """Navigate to the previous slide"""
        if self.is_presentation_active:
            # Try PowerPoint integration first
            if self.powerpoint_app:
                try:
                    presentation = self.powerpoint_app.ActivePresentation
                    slideshow = presentation.SlideShowWindow.View
                    slideshow.Previous()
                    return
                except Exception as e:
                    logger.error(f"PowerPoint integration error: {e}")
            
            # Fall back to keyboard simulation
            pyautogui.press('left')
            logger.info("Previous slide command executed")
    
    def start_presentation(self):
        """Start the presentation"""
        # Try PowerPoint integration first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowSettings
                slideshow.Run()
                self.is_presentation_active = True
                logger.info("Presentation started via PowerPoint API")
                return
            except Exception as e:
                logger.error(f"PowerPoint start presentation error: {e}")
        
        # Fall back to keyboard simulation (F5 for PowerPoint)
        pyautogui.press('f5')
        self.is_presentation_active = True
        logger.info("Presentation started")
    
    def end_presentation(self):
        """End the presentation"""
        # Try PowerPoint integration first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowWindow.View
                slideshow.Exit()
                self.is_presentation_active = False
                logger.info("Presentation ended via PowerPoint API")
                return
            except Exception as e:
                logger.error(f"PowerPoint end presentation error: {e}")
        
        # Fall back to keyboard simulation (ESC for PowerPoint)
        pyautogui.press('esc')
        self.is_presentation_active = False
        logger.info("Presentation ended")
    
    def speak_feedback(self, text):
        """Provide voice feedback"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
    
    def detect_emotion(self, image):
        """Detect facial emotion of the presenter"""
        if not self.is_authenticated:
            return "Not authenticated"
            
        try:
            # Analyze emotion using DeepFace
            analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            
            # Get dominant emotion
            if isinstance(analysis, list):
                emotion = analysis[0]["dominant_emotion"]
            else:
                emotion = analysis["dominant_emotion"]
                
            # Check if confused/angry and should pause
            if emotion in ["angry", "fear", "disgust", "sad"] and not self.is_paused_due_to_confusion:
                if self.is_presentation_active:
                    # Pause presentation by simulating space key
                    pyautogui.press('space')
                    self.is_paused_due_to_confusion = True
                    logger.info(f"Presentation paused due to detected {emotion}")
                    threading.Thread(target=self.speak_feedback, 
                                    args=(f"Presentation paused due to detected {emotion}",)).start()
            
            # Resume if no longer confused
            elif emotion not in ["angry", "fear", "disgust", "sad"] and self.is_paused_due_to_confusion:
                if self.is_presentation_active:
                    # Resume presentation by simulating space key
                    pyautogui.press('space')
                    self.is_paused_due_to_confusion = False
                    logger.info("Presentation resumed")
                    threading.Thread(target=self.speak_feedback, 
                                    args=("Presentation resumed",)).start()
            
            self.emotion_state = emotion
            return emotion
            
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return "Error"
    
    def calibrate_camera(self, num_images=15):
        """Calibrate the camera using a chessboard pattern"""
        logger.info("Starting camera calibration...")
        
        # Chessboard dimensions
        chessboard_size = (7, 4)  # Interior corners
        square_size = 0.025  # meters
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Convert to real-world measurements
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return False
            
        frames_captured = 0
        
        while frames_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            # If corners found, add to our arrays
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display the corners
                cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
                cv2.putText(frame, f"Capturing: {frames_captured+1}/{num_images}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Calibration', frame)
                
                frames_captured += 1
                cv2.waitKey(500)  # Wait half a second between captures
            else:
                # Show the frame even if no corners detected
                cv2.putText(frame, "No chessboard detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Calibration', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frames_captured < 3:
            logger.error("Not enough frames captured for calibration")
            return False
            
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
            
        if ret:
            # Save calibration data
            self.calibration_data = {
                'camera_matrix': mtx.tolist(),
                'distortion_coefficients': dist.tolist(),
                'success': True
            }
            self.save_calibration_data()
            logger.info("Camera calibration completed successfully")
            print("\n=== Calibration Results ===")
            print("Camera Matrix:\n", mtx)
            print("Distortion Coefficients:\n", dist)
            return True
        else:
            logger.error("Camera calibration failed")
            return False
    
    def save_calibration_data(self):
        """Save camera calibration data to disk"""
        if self.calibration_data:
            try:
                with open('camera_calibration.json', 'w') as f:
                    json.dump(self.calibration_data, f)
                logger.info("Camera calibration data saved")
            except Exception as e:
                logger.error(f"Error saving calibration data: {e}")
    
    def load_calibration_data(self):
        """Load camera calibration data from disk"""
        if os.path.exists('camera_calibration.json'):
            try:
                with open('camera_calibration.json', 'r') as f:
                    self.calibration_data = json.load(f)
                    
                # Convert lists back to numpy arrays
                if self.calibration_data.get('success', False):
                    self.calibration_data['camera_matrix'] = np.array(
                        self.calibration_data['camera_matrix'])
                    self.calibration_data['distortion_coefficients'] = np.array(
                        self.calibration_data['distortion_coefficients'])
                    
                logger.info("Camera calibration data loaded")
            except Exception as e:
                logger.error(f"Error loading calibration data: {e}")
                self.calibration_data = None
    
    def undistort_image(self, image):
        """Apply camera calibration to undistort the image"""
        if self.calibration_data and self.calibration_data.get('success', False):
            try:
                h, w = image.shape[:2]
                mtx = self.calibration_data['camera_matrix']
                dist = self.calibration_data['distortion_coefficients']
                
                # Get optimal new camera matrix
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # Undistort
                dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
                
                # Crop the image (optional)
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                
                return dst
            except Exception as e:
                logger.error(f"Error undistorting image: {e}")
                return image
        else:
            return image
        
    
    def save_calibration_data(self):
        """Save camera calibration data to disk"""
        if self.calibration_data:
            try:
                with open('camera_calibration.json', 'w') as f:
                    json.dump(self.calibration_data, f)
                logger.info("Camera calibration data saved")
            except Exception as e:
                logger.error(f"Error saving calibration data: {e}")
    
    def load_calibration_data(self):
        """Load camera calibration data from disk"""
        if os.path.exists('camera_calibration.json'):
            try:
                with open('camera_calibration.json', 'r') as f:
                    self.calibration_data = json.load(f)
                    
                # Convert lists back to numpy arrays
                if self.calibration_data.get('success', False):
                    self.calibration_data['camera_matrix'] = np.array(
                        self.calibration_data['camera_matrix'])
                    self.calibration_data['distortion_coefficients'] = np.array(
                        self.calibration_data['distortion_coefficients'])
                    
                logger.info("Camera calibration data loaded")
            except Exception as e:
                logger.error(f"Error loading calibration data: {e}")
                self.calibration_data = None
    
    def undistort_image(self, image):
        """Apply camera calibration to undistort the image"""
        if self.calibration_data and self.calibration_data.get('success', False):
            try:
                h, w = image.shape[:2]
                mtx = self.calibration_data['camera_matrix']
                dist = self.calibration_data['distortion_coefficients']
                
                # Get optimal new camera matrix
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # Undistort
                dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
                
                # Crop the image (optional)
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                
                return dst
            except Exception as e:
                logger.error(f"Error undistorting image: {e}")
                return image
        else:
            return image
    
    def connect_to_powerpoint(self):
        """Connect to PowerPoint application via COM API"""
        try:
            self.powerpoint_app = client.CreateObject("PowerPoint.Application")
            self.powerpoint_app.Visible = True
            logger.info("Connected to PowerPoint successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PowerPoint: {e}")
            self.powerpoint_app = None
            return False
    
    def save_custom_gesture(self, gesture_name, hand_landmarks, action):
        """Save a custom gesture configuration"""
        if not hand_landmarks:
            return False
            
        # Extract landmark positions
        positions = []
        for landmark in hand_landmarks.landmark:
            positions.append((landmark.x, landmark.y, landmark.z))
            
        # Normalize positions
        normalized_positions = self.normalize_landmarks(positions)
        
        # Save the custom gesture
        self.custom_gestures[gesture_name] = {
            'landmarks': normalized_positions,
            'action': action,
            'timestamp': time.time()
        }
        
        self.save_custom_gestures()
        logger.info(f"Custom gesture '{gesture_name}' saved")
        return True
    
    def save_custom_gestures(self):
        """Save custom gestures to disk"""
        try:
            with open('custom_gestures.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_gestures = {}
                for name, data in self.custom_gestures.items():
                    serializable_gestures[name] = {
                        'landmarks': [list(p) for p in data['landmarks']],
                        'action': data['action'],
                        'timestamp': data['timestamp']
                    }
                json.dump(serializable_gestures, f)
            logger.info("Custom gestures saved")
        except Exception as e:
            logger.error(f"Error saving custom gestures: {e}")
    
    def load_custom_gestures(self):
        """Load custom gestures from disk"""
        if os.path.exists('custom_gestures.json'):
            try:
                with open('custom_gestures.json', 'r') as f:
                    self.custom_gestures = json.load(f)
                logger.info(f"Loaded {len(self.custom_gestures)} custom gestures")
            except Exception as e:
                logger.error(f"Error loading custom gestures: {e}")
                self.custom_gestures = {}
    
    def remove_custom_gesture(self, gesture_name):
        """Remove a custom gesture from the system"""
        if gesture_name in self.custom_gestures:
            del self.custom_gestures[gesture_name]
            self.save_custom_gestures()
            return True
        return False


class GestureControlUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PowerPoint Gesture Control Assistant")
        self.master.geometry("1200x800")
        self.master.minsize(1000, 700)
        
        # Initialize the gesture control system
        self.gc_system = GestureControlSystem()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Main tab
        self.main_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.main_tab, text="Main Control")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Settings")
        
        # User Management tab
        self.user_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.user_tab, text="User Management")
        
        # Gesture Training tab
        self.training_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.training_tab, text="Gesture Training")
        
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Initialize each tab
        self.setup_main_tab()
        self.setup_settings_tab()
        self.setup_user_tab()
        self.setup_training_tab()
        
        # Initialize variables
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.detection_active = False
        self.training_mode = False
        
        # Set up camera
        self.setup_camera()
        
        # Connect to PowerPoint if possible
        self.try_connect_powerpoint()
    
    def setup_main_tab(self):
        """Set up the main control tab"""
        # Top frame for camera view
        self.camera_frame = ttk.Frame(self.main_tab)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(self.camera_frame, bg="black", width=640, height=480)
        self.camera_canvas.pack(padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(self.main_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Status frame
        self.status_frame = ttk.LabelFrame(self.main_tab, text="Status")
        self.status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Status labels
        self.user_status = ttk.Label(self.status_frame, text="User: Not authenticated")
        self.user_status.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.gesture_status = ttk.Label(self.status_frame, text="Gesture: None")
        self.gesture_status.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.presentation_status = ttk.Label(self.status_frame, text="Presentation: Not active")
        self.presentation_status.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.emotion_status = ttk.Label(self.status_frame, text="Emotion: Not detected")
        self.emotion_status.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="Start Detection", command=self.toggle_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.auth_btn = ttk.Button(control_frame, text="Authenticate", command=self.authenticate_user)
        self.auth_btn.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = ttk.Button(control_frame, text="Connect PowerPoint", command=self.try_connect_powerpoint)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        self.calibrate_btn = ttk.Button(control_frame, text="Calibrate Camera", command=self.start_calibration)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        self.help_btn = ttk.Button(control_frame, text="Help", command=self.show_help)
        self.help_btn.pack(side=tk.RIGHT, padx=5)
    
    def setup_settings_tab(self):
        """Set up the settings tab"""
        # Create settings frame
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Application Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Cooldown period setting
        ttk.Label(settings_frame, text="Gesture Cooldown (seconds):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.cooldown_var = tk.DoubleVar(value=self.gc_system.cooldown_period)
        cooldown_spin = ttk.Spinbox(settings_frame, from_=0.5, to=5.0, increment=0.5, textvariable=self.cooldown_var)
        cooldown_spin.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Voice feedback setting
        self.voice_feedback_var = tk.BooleanVar(value=True)
        voice_check = ttk.Checkbutton(settings_frame, text="Enable Voice Feedback", variable=self.voice_feedback_var)
        voice_check.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # Emotion detection setting
        self.emotion_detection_var = tk.BooleanVar(value=True)
        emotion_check = ttk.Checkbutton(settings_frame, text="Enable Emotion Detection", variable=self.emotion_detection_var)
        emotion_check.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        
        # Auto-pause on confusion setting
        self.autopause_var = tk.BooleanVar(value=True)
        autopause_check = ttk.Checkbutton(settings_frame, text="Auto-pause on Confusion", variable=self.autopause_var)
        autopause_check.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        
        # Apply button
        apply_btn = ttk.Button(settings_frame, text="Apply Settings", command=self.apply_settings)
        apply_btn.grid(row=4, column=0, padx=10, pady=20, sticky="w")
        
        # Reset to defaults button
        reset_btn = ttk.Button(settings_frame, text="Reset to Defaults", command=self.reset_settings)
        reset_btn.grid(row=4, column=1, padx=10, pady=20, sticky="w")
    
    def setup_user_tab(self):
        """Set up the user management tab"""
        # User list frame
        list_frame = ttk.LabelFrame(self.user_tab, text="Registered Users")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # User listbox
        self.user_listbox = tk.Listbox(list_frame, width=30, height=15)
        self.user_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.user_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.user_listbox.configure(yscrollcommand=scrollbar.set)
        
        # User action frame
        action_frame = ttk.Frame(self.user_tab)
        action_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Register new user button
        register_btn = ttk.Button(action_frame, text="Register New User", command=self.register_new_user)
        register_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Delete user button
        delete_btn = ttk.Button(action_frame, text="Delete Selected User", command=self.delete_user)
        delete_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Refresh user list button
        refresh_btn = ttk.Button(action_frame, text="Refresh User List", command=self.refresh_user_list)
        refresh_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Load initial user list
        self.refresh_user_list()
    
    def setup_training_tab(self):
        """Set up the gesture training tab"""
        # Training frame
        training_frame = ttk.LabelFrame(self.training_tab, text="Custom Gesture Training")
        training_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - gesture list
        list_frame = ttk.Frame(training_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Custom Gestures:").pack(anchor="w", padx=5, pady=5)
        
        self.gesture_listbox = tk.Listbox(list_frame, width=30, height=15)
        self.gesture_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for listbox
        g_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.gesture_listbox.yview)
        g_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gesture_listbox.configure(yscrollcommand=g_scrollbar.set)
        
        # Right panel - training controls
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Gesture name entry
        ttk.Label(control_frame, text="Gesture Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.gesture_name_var = tk.StringVar()
        gesture_entry = ttk.Entry(control_frame, textvariable=self.gesture_name_var, width=20)
        gesture_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Gesture action selection
        ttk.Label(control_frame, text="Action:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.gesture_action_var = tk.StringVar(value="next")
        actions = ["next", "previous", "start", "end"]
        action_combo = ttk.Combobox(control_frame, textvariable=self.gesture_action_var, values=actions, state="readonly")
        action_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Start training button
        train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_gesture_training)
        train_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=15, sticky="we")
        
        # Capture gesture button
        self.capture_btn = ttk.Button(control_frame, text="Capture Gesture", command=self.capture_gesture, state=tk.DISABLED)
        self.capture_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="we")
        
        # Delete gesture button
        delete_g_btn = ttk.Button(control_frame, text="Delete Selected Gesture", command=self.delete_gesture)
        delete_g_btn.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="we")
        
        # Refresh gesture list button
        refresh_g_btn = ttk.Button(control_frame, text="Refresh Gesture List", command=self.refresh_gesture_list)
        refresh_g_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="we")
        
        # Load initial gesture list
        self.refresh_gesture_list()
    
    def setup_camera(self):
        """Initialize the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return False
            
            # Set resolution to 640x480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera_active = True
            self.update_camera()
            return True
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error initializing camera: {e}")
            return False
    
    def update_camera(self):
        """Update camera feed"""
        if self.camera_active and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Store the current frame for processing
                self.current_frame = frame.copy()
                
                # Apply calibration if available
                if self.gc_system.calibration_data and self.gc_system.calibration_data.get('success', False):
                    frame = self.gc_system.undistort_image(frame)
                
                # Process frame if detection is active
                if self.detection_active:
                    # Process for hand gestures
                    gesture = self.gc_system.detect_gestures(frame)
                    self.gesture_status.config(text=f"Gesture: {gesture}")
                    
                    # Process for emotion if enabled
                    if self.emotion_detection_var.get():
                        emotion = self.gc_system.detect_emotion(frame)
                        self.emotion_status.config(text=f"Emotion: {emotion}")
                    
                    # Draw gesture feedback on frame
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw authentication status
                    auth_text = f"User: {self.gc_system.recognized_user}" if self.gc_system.is_authenticated else "Not authenticated"
                    cv2.putText(frame, auth_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw presentation status
                    pres_text = "Presentation: Active" if self.gc_system.is_presentation_active else "Presentation: Not active"
                    cv2.putText(frame, pres_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process frame if in training mode
                if self.training_mode:
                    # Draw training mode indicator
                    cv2.putText(frame, "TRAINING MODE", (frame.shape[1]//2 - 80, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Process the frame to detect hands for visual feedback
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = self.gc_system.hands.process(image_rgb)
                    
                    # Draw hand landmarks
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            self.gc_system.mp_drawing.draw_landmarks(
                                frame, 
                                hand_landmarks, 
                                self.gc_system.mp_hands.HAND_CONNECTIONS)
                
                # Convert to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.camera_canvas.image = img_tk  # Keep a reference
            
            # Schedule the next update
            self.master.after(10, self.update_camera)
    
    def toggle_detection(self):
        """Toggle gesture detection on/off"""
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            self.start_btn.config(text="Stop Detection")
            # Update status display
            self.presentation_status.config(text=f"Presentation: {'Active' if self.gc_system.is_presentation_active else 'Not active'}")
        else:
            self.start_btn.config(text="Start Detection")
            self.gesture_status.config(text="Gesture: None")
            self.emotion_status.config(text="Emotion: Not detected")
    
    def authenticate_user(self):
        """Authenticate the current user"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available")
            return
            
        # Try to authenticate
        success, user_info = self.gc_system.authenticate_user(self.current_frame)
        
        if success:
            self.user_status.config(text=f"User: {user_info}")
            messagebox.showinfo("Authentication", f"Successfully authenticated as {user_info}")
        else:
            self.user_status.config(text="User: Not authenticated")
            messagebox.showerror("Authentication", f"Authentication failed: {user_info}")
    
    def try_connect_powerpoint(self):
        """Try to connect to PowerPoint"""
        if self.gc_system.connect_to_powerpoint():
            self.connect_btn.config(text="PowerPoint Connected", state=tk.DISABLED)
            messagebox.showinfo("PowerPoint", "Successfully connected to PowerPoint")
        else:
            messagebox.showwarning("PowerPoint", "Failed to connect to PowerPoint. Will use keyboard simulation instead.")
    
    #-------------------------------------------------------------------------
    def start_calibration(self):
        """Start camera calibration process"""
        if messagebox.askyesno("Camera Calibration", 
                              "This will start the camera calibration process. You will need a chessboard pattern. Proceed?"):
            # Disable UI during calibration
            self.calibrate_btn.config(state=tk.DISABLED)
            
            # Start calibration in a separate thread
            threading.Thread(target=self.run_calibration).start()
    
    def run_calibration(self):
        """Run the calibration process"""
        success = self.gc_system.calibrate_camera()
        
        # Re-enable UI when done
        self.master.after(0, lambda: self.calibrate_btn.config(state=tk.NORMAL))
        
        if success:
            self.master.after(0, lambda: messagebox.showinfo("Calibration", 
                                                           "Camera calibration completed successfully"))
        else:
            self.master.after(0, lambda: messagebox.showerror("Calibration", 
                                                            "Camera calibration failed. Please try again."))
    
    def apply_settings(self):
        """Apply settings from the settings tab"""
        # Update cooldown period
        self.gc_system.cooldown_period = self.cooldown_var.get()
        
        # Update other settings as needed
        messagebox.showinfo("Settings", "Settings applied successfully")
    
    def reset_settings(self):
        """Reset settings to defaults"""
        self.cooldown_var.set(1.5)
        self.voice_feedback_var.set(True)
        self.emotion_detection_var.set(True)
        self.autopause_var.set(True)
        
        # Reset system settings
        self.gc_system.cooldown_period = 1.5
        
        messagebox.showinfo("Settings", "Settings reset to defaults")
    
    def register_new_user(self):
        """Register a new user"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available")
            return
            
        # Ask for username
        username = simpledialog.askstring("Register User", "Enter username:")
        if not username:
            return
            
        # Check if username already exists
        if username in self.gc_system.users_db:
            if not messagebox.askyesno("Warning", 
                                    f"User '{username}' already exists. Do you want to overwrite?"):
                return
        
        # Register the user
        success = self.gc_system.register_new_user(username, self.current_frame)
        
        if success:
            messagebox.showinfo("Registration", f"User '{username}' registered successfully")
            self.refresh_user_list()
        else:
            messagebox.showerror("Registration", "Failed to register user. Please try again.")
    
    def delete_user(self):
        """Delete the selected user"""
        selected = self.user_listbox.curselection()
        if not selected:
            messagebox.showwarning("Delete User", "No user selected")
            return
            
        username = self.user_listbox.get(selected[0])
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user '{username}'?"):
            if username in self.gc_system.users_db:
                del self.gc_system.users_db[username]
                self.gc_system.save_user_data()
                self.refresh_user_list()
                messagebox.showinfo("Delete User", f"User '{username}' deleted successfully")
    
    def refresh_user_list(self):
        """Refresh the user list"""
        self.user_listbox.delete(0, tk.END)
        
        for username in self.gc_system.users_db:
            self.user_listbox.insert(tk.END, username)
    
    def start_gesture_training(self):
        """Start gesture training mode"""
        gesture_name = self.gesture_name_var.get().strip()
        if not gesture_name:
            messagebox.showwarning("Training", "Please enter a gesture name")
            return
        
        # Enable training mode
        self.training_mode = True
        self.capture_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Training", "Training mode activated.\n\nPosition your hand in front of the camera and click 'Capture Gesture' when ready.")
    
    def capture_gesture(self):
        """Capture the current hand pose as a custom gesture"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available")
            return
            
        gesture_name = self.gesture_name_var.get().strip()
        action = self.gesture_action_var.get()
        
        # Process the frame to detect hands
        image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        hand_results = self.gc_system.hands.process(image_rgb)
        
        if not hand_results.multi_hand_landmarks:
            messagebox.showwarning("Training", "No hand detected. Please position your hand in front of the camera.")
            return
        
        # Save the gesture
        success = self.gc_system.save_custom_gesture(
            gesture_name, 
            hand_results.multi_hand_landmarks[0], 
            action
        )
        
        if success:
            messagebox.showinfo("Training", f"Gesture '{gesture_name}' saved successfully")
            self.training_mode = False
            self.capture_btn.config(state=tk.DISABLED)
            self.refresh_gesture_list()
        else:
            messagebox.showerror("Training", "Failed to save gesture. Please try again.")
    
    def delete_gesture(self):
        """Delete the selected custom gesture"""
        selected = self.gesture_listbox.curselection()
        if not selected:
            messagebox.showwarning("Delete Gesture", "No gesture selected")
            return
            
        gesture_name = self.gesture_listbox.get(selected[0])
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete gesture '{gesture_name}'?"):
            if self.gc_system.remove_custom_gesture(gesture_name):
                self.refresh_gesture_list()
                messagebox.showinfo("Delete Gesture", f"Gesture '{gesture_name}' deleted successfully")
    
    def refresh_gesture_list(self):
        """Refresh the custom gesture list"""
        self.gesture_listbox.delete(0, tk.END)
        
        for gesture_name in self.gc_system.custom_gestures:
            self.gesture_listbox.insert(tk.END, gesture_name)
    
    def show_help(self):
        """Show help information"""
        help_text = """
        PowerPoint Gesture Control Assistant - Help
        
        Default Gestures:
        - Open palm (all fingers extended): Start Presentation
        - Index finger only: Next Slide
        - Pinky finger only: Previous Slide
        - Closed fist: End Presentation
        
        Usage Instructions:
        1. Register your face in the User Management tab
        2. Click 'Authenticate' to verify your identity
        3. Click 'Start Detection' to begin gesture recognition
        4. Use hand gestures to control your presentation
        
        Custom Gestures:
        - Create custom gestures in the Gesture Training tab
        - Each gesture can be associated with an action
        
        Camera Calibration:
        - Use a printed chessboard pattern for calibration
        - Follow on-screen instructions during calibration
        """
        messagebox.showinfo("Help", help_text)
        
    def on_closing(self):
        """Handle window closing"""
        if self.camera_active and self.cap is not None:
            self.camera_active = False
            self.cap.release()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = GestureControlUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
    