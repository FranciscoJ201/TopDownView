# transformer.py
import cv2
import numpy as np
import os
import config

class PerspectiveManager:
    def __init__(self):
        self.matrix = None

    def load_matrix(self):
        """Attempts to load an existing matrix file."""
        if os.path.exists(config.MATRIX_FILE):
            try:
                self.matrix = np.load(config.MATRIX_FILE)
                print(f"[INFO] Loaded matrix from {config.MATRIX_FILE}")
                return True
            except Exception as e:
                print(f"[ERROR] Loading matrix failed: {e}")
                return False
        return False

    def compute_and_save_matrix(self, src_points):
        """
        Takes 4 source points (from GUI clicks), computes the matrix,
        and saves it to disk.
        """
        if len(src_points) != 4:
            raise ValueError("Need exactly 4 points to calibrate.")

        # Source: User clicks
        src_pts = np.float32(src_points)

        # Destination: Perfect Square Map
        dst_pts = np.float32([
            [0, 0], 
            [config.MAP_SIZE, 0], 
            [config.MAP_SIZE, config.MAP_SIZE], 
            [0, config.MAP_SIZE]
        ])

        try:
            self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            np.save(config.MATRIX_FILE, self.matrix)
            print(f"[INFO] Matrix saved to {config.MATRIX_FILE}")
            return True
        except Exception as e:
            print(f"[ERROR] Matrix computation failed: {e}")
            return False

    def transform_point(self, x, y):
        """Maps video (x,y) -> map (x,y). Returns (0,0) if no matrix."""
        if self.matrix is None:
            return 0, 0
            
        # Reshape to (1, 1, 2) for OpenCV
        point_vec = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_vec, self.matrix)
        
        return int(transformed[0][0][0]), int(transformed[0][0][1])