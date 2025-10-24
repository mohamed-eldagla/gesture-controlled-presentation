"""
Presentation control for PowerPoint and other presentation software.
"""

import pyautogui
import pyttsx3
import threading
from typing import Optional
import sys

from ..utils.logger import get_logger

logger = get_logger()

# Try to import comtypes for Windows PowerPoint integration
if sys.platform == "win32":
    try:
        from comtypes import client
        COM_AVAILABLE = True
    except ImportError:
        COM_AVAILABLE = False
        logger.warning("comtypes not available - using keyboard simulation only")
else:
    COM_AVAILABLE = False


class PresentationController:
    """
    Controls presentation software (PowerPoint, Keynote, etc.) using
    COM API on Windows or keyboard simulation as fallback.
    """

    def __init__(self, voice_feedback: bool = True):
        """
        Initialize presentation controller.

        Args:
            voice_feedback: Enable voice feedback for actions
        """
        self.powerpoint_app: Optional[object] = None
        self.is_presentation_active = False
        self.voice_feedback_enabled = voice_feedback

        # Initialize text-to-speech
        if voice_feedback:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.voice_feedback_enabled = False

    def connect_to_powerpoint(self) -> bool:
        """
        Connect to PowerPoint application via COM API (Windows only).

        Returns:
            True if connection successful
        """
        if not COM_AVAILABLE:
            logger.info("PowerPoint COM API not available - using keyboard simulation")
            return False

        try:
            self.powerpoint_app = client.CreateObject("PowerPoint.Application")
            self.powerpoint_app.Visible = True
            logger.info("Connected to PowerPoint successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PowerPoint: {e}")
            self.powerpoint_app = None
            return False

    def next_slide(self):
        """Navigate to the next slide."""
        if not self.is_presentation_active:
            return

        # Try PowerPoint COM API first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowWindow.View
                slideshow.Next()
                logger.info("Next slide (PowerPoint API)")
                return
            except Exception as e:
                logger.error(f"PowerPoint API error: {e}")

        # Fall back to keyboard simulation
        pyautogui.press('right')
        logger.info("Next slide (keyboard)")

    def previous_slide(self):
        """Navigate to the previous slide."""
        if not self.is_presentation_active:
            return

        # Try PowerPoint COM API first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowWindow.View
                slideshow.Previous()
                logger.info("Previous slide (PowerPoint API)")
                return
            except Exception as e:
                logger.error(f"PowerPoint API error: {e}")

        # Fall back to keyboard simulation
        pyautogui.press('left')
        logger.info("Previous slide (keyboard)")

    def start_presentation(self):
        """Start the presentation."""
        # Try PowerPoint COM API first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowSettings
                slideshow.Run()
                self.is_presentation_active = True
                logger.info("Presentation started (PowerPoint API)")
                return
            except Exception as e:
                logger.error(f"PowerPoint start error: {e}")

        # Fall back to keyboard simulation (F5 for PowerPoint)
        pyautogui.press('f5')
        self.is_presentation_active = True
        logger.info("Presentation started (keyboard)")

    def end_presentation(self):
        """End the presentation."""
        # Try PowerPoint COM API first
        if self.powerpoint_app:
            try:
                presentation = self.powerpoint_app.ActivePresentation
                slideshow = presentation.SlideShowWindow.View
                slideshow.Exit()
                self.is_presentation_active = False
                logger.info("Presentation ended (PowerPoint API)")
                return
            except Exception as e:
                logger.error(f"PowerPoint end error: {e}")

        # Fall back to keyboard simulation (ESC for PowerPoint)
        pyautogui.press('esc')
        self.is_presentation_active = False
        logger.info("Presentation ended (keyboard)")

    def pause_presentation(self):
        """Pause/unpause the presentation."""
        if self.is_presentation_active:
            pyautogui.press('space')
            logger.info("Presentation paused/resumed")

    def speak_feedback(self, text: str):
        """
        Provide voice feedback.

        Args:
            text: Text to speak
        """
        if not self.voice_feedback_enabled:
            return

        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"Text-to-speech error: {e}")

        # Run in separate thread to avoid blocking
        threading.Thread(target=_speak, daemon=True).start()

    def execute_gesture_action(self, gesture: str, custom_gestures: dict = None):
        """
        Execute action based on detected gesture.

        Args:
            gesture: Gesture name
            custom_gestures: Dictionary of custom gesture configurations
        """
        # Provide voice feedback
        self.speak_feedback(f"{gesture} activated")

        # Execute built-in gesture actions
        if gesture == "Next Slide":
            self.next_slide()
        elif gesture == "Previous Slide":
            self.previous_slide()
        elif gesture == "Start Presentation":
            self.start_presentation()
        elif gesture == "End Presentation":
            self.end_presentation()

        # Execute custom gesture actions
        if custom_gestures and gesture in custom_gestures:
            action = custom_gestures[gesture].get('action')
            if action == "next":
                self.next_slide()
            elif action == "previous":
                self.previous_slide()
            elif action == "start":
                self.start_presentation()
            elif action == "end":
                self.end_presentation()

    def handle_emotion(self, emotion: str, auto_pause: bool = True):
        """
        Handle emotion detection for auto-pause functionality.

        Args:
            emotion: Detected emotion
            auto_pause: Whether auto-pause is enabled

        Returns:
            Action taken (None, 'paused', or 'resumed')
        """
        if not auto_pause or not self.is_presentation_active:
            return None

        negative_emotions = ["angry", "fear", "disgust", "sad"]

        if emotion in negative_emotions:
            self.pause_presentation()
            self.speak_feedback(f"Presentation paused due to detected {emotion}")
            logger.info(f"Presentation paused due to detected {emotion}")
            return 'paused'
        elif emotion not in negative_emotions:
            # Could add logic to track pause state and resume
            return None

    def set_voice_feedback(self, enabled: bool):
        """Enable or disable voice feedback."""
        self.voice_feedback_enabled = enabled
        logger.info(f"Voice feedback {'enabled' if enabled else 'disabled'}")
