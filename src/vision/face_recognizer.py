"""
Face recognition and authentication using DeepFace.
"""

import numpy as np
import time
from typing import Tuple, Optional
from deepface import DeepFace

from ..utils.logger import get_logger

logger = get_logger()


class FaceRecognizer:
    """
    Handles face recognition and user authentication using DeepFace.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize face recognizer.

        Args:
            similarity_threshold: Minimum similarity score for authentication
        """
        self.similarity_threshold = similarity_threshold
        self.recognized_user: Optional[str] = None
        self.is_authenticated = False

    def register_user(self, username: str, face_image: np.ndarray) -> Tuple[bool, Optional[dict]]:
        """
        Register a new user with their face image.

        Args:
            username: Username for registration
            face_image: Face image (BGR format)

        Returns:
            Tuple of (success, embedding_data)
        """
        try:
            # Extract face embedding using DeepFace
            embedding = DeepFace.represent(
                face_image,
                model_name="Facenet",
                enforce_detection=True
            )

            # Store user data
            user_data = {
                'embedding': embedding,
                'timestamp': time.time()
            }

            logger.info(f"User '{username}' registered successfully")
            return True, user_data

        except Exception as e:
            logger.error(f"Error registering user '{username}': {e}")
            return False, None

    def authenticate(self, face_image: np.ndarray, users_db: dict) -> Tuple[bool, str]:
        """
        Authenticate user using face recognition.

        Args:
            face_image: Face image to authenticate (BGR format)
            users_db: Database of registered users

        Returns:
            Tuple of (success, username or error message)
        """
        if not users_db:
            logger.warning("No users registered in the system")
            return False, "No users registered"

        try:
            # Get face embedding
            new_embedding = DeepFace.represent(
                face_image,
                model_name="Facenet",
                enforce_detection=True
            )

            # Compare with stored embeddings
            best_match = None
            best_similarity = -1

            for username, data in users_db.items():
                stored_embedding = data['embedding']

                # Calculate cosine similarity
                similarity = self.calculate_similarity(
                    new_embedding[0]["embedding"],
                    stored_embedding[0]["embedding"]
                )

                if similarity > self.similarity_threshold and similarity > best_similarity:
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

    def calculate_similarity(self, embedding1: list, embedding2: list) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        return dot_product / (norm1 * norm2)

    def reset_authentication(self):
        """Reset authentication state."""
        self.recognized_user = None
        self.is_authenticated = False
        logger.info("Authentication state reset")
