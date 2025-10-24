"""
Hand gesture detection using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from typing import Optional, Dict, Tuple, List

from ..utils.logger import get_logger

logger = get_logger()


class GestureDetector:
    """
    Detects and classifies hand gestures using MediaPipe Hands.

    Supports both built-in gestures and custom user-defined gestures.
    """

    def __init__(self, cooldown_period: float = 1.5):
        """
        Initialize gesture detector.

        Args:
            cooldown_period: Minimum time between gesture detections (seconds)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.current_gesture = "None"
        self.last_action_time = time.time()
        self.cooldown_period = cooldown_period
        self.custom_gestures: Dict[str, Dict] = {}

    def detect(self, image: np.ndarray, is_authenticated: bool = True) -> str:
        """
        Detect hand gesture in the image.

        Args:
            image: Input image (BGR format)
            is_authenticated: Whether user is authenticated

        Returns:
            Detected gesture name
        """
        if not is_authenticated:
            return "Not authenticated"

        # Check cooldown period
        if time.time() - self.last_action_time < self.cooldown_period:
            return self.current_gesture

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Check custom gestures first
            if self.custom_gestures:
                for gesture_name, gesture_data in self.custom_gestures.items():
                    if self.match_custom_gesture(hand_landmarks, gesture_data):
                        if self.current_gesture != gesture_name:
                            self.current_gesture = gesture_name
                            self.last_action_time = time.time()
                        return gesture_name

            # Check built-in gestures
            gesture = self.classify_gesture(hand_landmarks)

            if gesture != self.current_gesture:
                self.current_gesture = gesture
                self.last_action_time = time.time()

            return gesture
        else:
            if self.current_gesture != "None":
                self.current_gesture = "None"
            return "None"

    def classify_gesture(self, hand_landmarks) -> str:
        """
        Classify built-in hand gesture.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Gesture name
        """
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

        # Gesture classification
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

    def match_custom_gesture(self, landmarks, gesture_data: Dict) -> bool:
        """
        Match hand landmarks against saved custom gesture.

        Args:
            landmarks: Current hand landmarks
            gesture_data: Saved gesture template data

        Returns:
            True if gesture matches
        """
        # Extract current landmark positions
        current_positions = []
        for landmark in landmarks.landmark:
            current_positions.append((landmark.x, landmark.y, landmark.z))

        # Get saved template
        template = gesture_data['landmarks']

        # Normalize current positions
        norm_current = self.normalize_landmarks(current_positions)

        # Calculate difference
        total_diff = 0
        for i in range(len(norm_current)):
            diff = math.sqrt(sum((a - b)**2 for a, b in zip(norm_current[i], template[i])))
            total_diff += diff

        # Check if average difference is below threshold
        avg_diff = total_diff / len(norm_current)
        return avg_diff < 0.2

    def normalize_landmarks(self, landmarks: List[Tuple]) -> List[Tuple]:
        """
        Normalize landmark positions relative to wrist.

        Args:
            landmarks: List of (x, y, z) landmark coordinates

        Returns:
            Normalized landmark positions
        """
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

    def save_custom_gesture(self, gesture_name: str, hand_landmarks, action: str) -> bool:
        """
        Save a custom gesture configuration.

        Args:
            gesture_name: Name for the gesture
            hand_landmarks: MediaPipe hand landmarks
            action: Action to associate (next, previous, start, end)

        Returns:
            True if successful
        """
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

        logger.info(f"Custom gesture '{gesture_name}' saved")
        return True

    def remove_custom_gesture(self, gesture_name: str) -> bool:
        """
        Remove a custom gesture.

        Args:
            gesture_name: Name of gesture to remove

        Returns:
            True if successful
        """
        if gesture_name in self.custom_gestures:
            del self.custom_gestures[gesture_name]
            return True
        return False

    def set_cooldown_period(self, period: float):
        """Set gesture cooldown period."""
        self.cooldown_period = period
        logger.info(f"Cooldown period set to {period}s")

    def draw_landmarks(self, image: np.ndarray, hand_landmarks):
        """
        Draw hand landmarks on image.

        Args:
            image: Image to draw on
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Image with landmarks drawn
        """
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        return image
