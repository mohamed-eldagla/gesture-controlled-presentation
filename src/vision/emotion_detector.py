"""
Emotion detection using DeepFace.
"""

import numpy as np
from typing import Optional
from deepface import DeepFace

from ..utils.logger import get_logger

logger = get_logger()


class EmotionDetector:
    """
    Detects facial emotions using DeepFace.
    """

    def __init__(self):
        """Initialize emotion detector."""
        self.current_emotion = "Neutral"
        self.is_paused_due_to_confusion = False
        self.negative_emotions = ["angry", "fear", "disgust", "sad"]

    def detect(self, image: np.ndarray, is_authenticated: bool = True) -> str:
        """
        Detect emotion in the image.

        Args:
            image: Input image (BGR format)
            is_authenticated: Whether user is authenticated

        Returns:
            Detected emotion
        """
        if not is_authenticated:
            return "Not authenticated"

        try:
            # Analyze emotion using DeepFace
            analysis = DeepFace.analyze(
                image,
                actions=['emotion'],
                enforce_detection=False
            )

            # Get dominant emotion
            if isinstance(analysis, list):
                emotion = analysis[0]["dominant_emotion"]
            else:
                emotion = analysis["dominant_emotion"]

            self.current_emotion = emotion
            return emotion

        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return "Error"

    def should_pause(self) -> bool:
        """
        Check if presentation should be paused based on emotion.

        Returns:
            True if should pause
        """
        return (self.current_emotion in self.negative_emotions and
                not self.is_paused_due_to_confusion)

    def should_resume(self) -> bool:
        """
        Check if presentation should be resumed.

        Returns:
            True if should resume
        """
        return (self.current_emotion not in self.negative_emotions and
                self.is_paused_due_to_confusion)

    def set_paused_state(self, paused: bool):
        """Set paused due to confusion state."""
        self.is_paused_due_to_confusion = paused

    def is_negative_emotion(self) -> bool:
        """Check if current emotion is negative."""
        return self.current_emotion in self.negative_emotions
