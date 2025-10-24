"""
Camera calibration using chessboard pattern.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from ..utils.logger import get_logger

logger = get_logger()


class CameraCalibrator:
    """
    Handles camera calibration for improved accuracy.
    """

    def __init__(self, chessboard_size: Tuple[int, int] = (7, 4), square_size: float = 0.025):
        """
        Initialize camera calibrator.

        Args:
            chessboard_size: Interior corners in chessboard (columns, rows)
            square_size: Size of chessboard squares in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.calibration_data: Optional[Dict] = None

    def calibrate(self, num_images: int = 15) -> bool:
        """
        Calibrate camera using chessboard pattern.

        Args:
            num_images: Number of calibration images to capture

        Returns:
            True if calibration successful
        """
        logger.info("Starting camera calibration...")

        # Prepare object points
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

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
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            # If corners found, add to arrays
            if ret:
                objpoints.append(objp)

                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners2, ret)
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
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            # Save calibration data
            self.calibration_data = {
                'camera_matrix': mtx.tolist(),
                'distortion_coefficients': dist.tolist(),
                'success': True
            }
            logger.info("Camera calibration completed successfully")
            print("\n=== Calibration Results ===")
            print("Camera Matrix:\n", mtx)
            print("Distortion Coefficients:\n", dist)
            return True
        else:
            logger.error("Camera calibration failed")
            return False

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply calibration to undistort image.

        Args:
            image: Input image

        Returns:
            Undistorted image
        """
        if self.calibration_data and self.calibration_data.get('success', False):
            try:
                h, w = image.shape[:2]
                mtx = np.array(self.calibration_data['camera_matrix'])
                dist = np.array(self.calibration_data['distortion_coefficients'])

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

    def load_calibration(self, data: Dict):
        """
        Load calibration data.

        Args:
            data: Calibration data dictionary
        """
        if data and data.get('success', False):
            # Convert lists back to numpy arrays
            self.calibration_data = {
                'camera_matrix': np.array(data['camera_matrix']),
                'distortion_coefficients': np.array(data['distortion_coefficients']),
                'success': True
            }
            logger.info("Camera calibration data loaded")

    def get_calibration_data(self) -> Optional[Dict]:
        """Get current calibration data."""
        return self.calibration_data

    def is_calibrated(self) -> bool:
        """Check if camera is calibrated."""
        return self.calibration_data is not None and self.calibration_data.get('success', False)
