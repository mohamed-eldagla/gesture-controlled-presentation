"""
Database classes for storing user data, gestures, and calibration.
"""

import os
import pickle
import json
from typing import Dict, Optional, Any

from ..utils.logger import get_logger

logger = get_logger()


class UserDatabase:
    """
    Manages user face recognition database.
    """

    def __init__(self, filepath: str = 'users_db.pkl'):
        """
        Initialize user database.

        Args:
            filepath: Path to database file
        """
        self.filepath = filepath
        self.users: Dict[str, Dict] = {}
        self.load()

    def load(self) -> bool:
        """
        Load user data from disk.

        Returns:
            True if successful
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    self.users = pickle.load(f)
                logger.info(f"Loaded {len(self.users)} user profiles")
                return True
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                self.users = {}
                return False
        return False

    def save(self) -> bool:
        """
        Save user data to disk.

        Returns:
            True if successful
        """
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.users, f)
            logger.info("User data saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            return False

    def add_user(self, username: str, user_data: Dict) -> bool:
        """
        Add or update user in database.

        Args:
            username: Username
            user_data: User data dictionary

        Returns:
            True if successful
        """
        self.users[username] = user_data
        return self.save()

    def remove_user(self, username: str) -> bool:
        """
        Remove user from database.

        Args:
            username: Username to remove

        Returns:
            True if successful
        """
        if username in self.users:
            del self.users[username]
            return self.save()
        return False

    def get_user(self, username: str) -> Optional[Dict]:
        """Get user data by username."""
        return self.users.get(username)

    def get_all_users(self) -> Dict[str, Dict]:
        """Get all users."""
        return self.users

    def user_exists(self, username: str) -> bool:
        """Check if user exists."""
        return username in self.users

    def count(self) -> int:
        """Get number of registered users."""
        return len(self.users)


class GestureDatabase:
    """
    Manages custom gesture database.
    """

    def __init__(self, filepath: str = 'custom_gestures.json'):
        """
        Initialize gesture database.

        Args:
            filepath: Path to database file
        """
        self.filepath = filepath
        self.gestures: Dict[str, Dict] = {}
        self.load()

    def load(self) -> bool:
        """
        Load gesture data from disk.

        Returns:
            True if successful
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.gestures = json.load(f)
                logger.info(f"Loaded {len(self.gestures)} custom gestures")
                return True
            except Exception as e:
                logger.error(f"Error loading custom gestures: {e}")
                self.gestures = {}
                return False
        return False

    def save(self) -> bool:
        """
        Save gesture data to disk.

        Returns:
            True if successful
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_gestures = {}
            for name, data in self.gestures.items():
                serializable_gestures[name] = {
                    'landmarks': [list(p) if isinstance(p, (list, tuple)) else p
                                 for p in data['landmarks']],
                    'action': data['action'],
                    'timestamp': data['timestamp']
                }

            with open(self.filepath, 'w') as f:
                json.dump(serializable_gestures, f, indent=2)
            logger.info("Custom gestures saved")
            return True
        except Exception as e:
            logger.error(f"Error saving custom gestures: {e}")
            return False

    def add_gesture(self, gesture_name: str, gesture_data: Dict) -> bool:
        """
        Add or update gesture in database.

        Args:
            gesture_name: Gesture name
            gesture_data: Gesture data dictionary

        Returns:
            True if successful
        """
        self.gestures[gesture_name] = gesture_data
        return self.save()

    def remove_gesture(self, gesture_name: str) -> bool:
        """
        Remove gesture from database.

        Args:
            gesture_name: Gesture name to remove

        Returns:
            True if successful
        """
        if gesture_name in self.gestures:
            del self.gestures[gesture_name]
            return self.save()
        return False

    def get_gesture(self, gesture_name: str) -> Optional[Dict]:
        """Get gesture data by name."""
        return self.gestures.get(gesture_name)

    def get_all_gestures(self) -> Dict[str, Dict]:
        """Get all gestures."""
        return self.gestures

    def gesture_exists(self, gesture_name: str) -> bool:
        """Check if gesture exists."""
        return gesture_name in self.gestures

    def count(self) -> int:
        """Get number of custom gestures."""
        return len(self.gestures)


class CalibrationDatabase:
    """
    Manages camera calibration data.
    """

    def __init__(self, filepath: str = 'camera_calibration.json'):
        """
        Initialize calibration database.

        Args:
            filepath: Path to calibration file
        """
        self.filepath = filepath
        self.calibration_data: Optional[Dict] = None
        self.load()

    def load(self) -> bool:
        """
        Load calibration data from disk.

        Returns:
            True if successful
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.calibration_data = json.load(f)
                logger.info("Camera calibration data loaded")
                return True
            except Exception as e:
                logger.error(f"Error loading calibration data: {e}")
                self.calibration_data = None
                return False
        return False

    def save(self) -> bool:
        """
        Save calibration data to disk.

        Returns:
            True if successful
        """
        if self.calibration_data:
            try:
                with open(self.filepath, 'w') as f:
                    json.dump(self.calibration_data, f, indent=2)
                logger.info("Camera calibration data saved")
                return True
            except Exception as e:
                logger.error(f"Error saving calibration data: {e}")
                return False
        return False

    def set_calibration(self, calibration_data: Dict) -> bool:
        """
        Set and save calibration data.

        Args:
            calibration_data: Calibration data dictionary

        Returns:
            True if successful
        """
        self.calibration_data = calibration_data
        return self.save()

    def get_calibration(self) -> Optional[Dict]:
        """Get calibration data."""
        return self.calibration_data

    def is_calibrated(self) -> bool:
        """Check if calibration data exists."""
        return (self.calibration_data is not None and
                self.calibration_data.get('success', False))

    def clear(self) -> bool:
        """
        Clear calibration data.

        Returns:
            True if successful
        """
        self.calibration_data = None
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
                logger.info("Calibration data cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing calibration data: {e}")
                return False
        return True
