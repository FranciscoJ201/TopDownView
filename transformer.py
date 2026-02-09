# transformer.py
import cv2
import numpy as np
import config

class PerspectiveManager:
    def __init__(self):
        self.matrix = None
        self.points = []
        self.temp_img = None

    def load_matrix(self):
        try:
            self.matrix = np.load(config.MATRIX_FILE)
            print(f"[INFO] Loaded existing matrix from {config.MATRIX_FILE}")
            return True
        except FileNotFoundError:
            return False

    def click_event(self, event, x, y, flags, params):
        # Only allow clicks if we have a locked frame (temp_img)
        if event == cv2.EVENT_LBUTTONDOWN and self.temp_img is not None:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Draw visual feedback
                cv2.circle(self.temp_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.temp_img, str(len(self.points)), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Calibrate: Click 4 Corners', self.temp_img)

    def select_frame(self, video_source):
        """Plays video and lets user hit ENTER to pick a frame."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_source}")

        print("[INFO] Controls: [SPACE] = Pause/Play, [ENTER] = Select Frame")
        
        paused = False
        selected_frame = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video if it ends before selection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            display = frame.copy()
            msg = "PAUSED - Press ENTER to Select" if paused else "Press SPACE to Pause"
            color = (0, 0, 255) if paused else (0, 255, 0)
            
            cv2.putText(display, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Select Frame', display)

            key = cv2.waitKey(30) & 0xFF

            if key == 32: # SPACE to pause/play
                paused = not paused
            elif key == 13: # ENTER to select
                selected_frame = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise Exception("Calibration cancelled.")

        cap.release()
        cv2.destroyWindow('Select Frame')
        return selected_frame

    def calibrate(self, video_source):
        print("[INFO] No matrix found. Starting calibration...")
        
        # 1. Let user pick the frame
        self.temp_img = self.select_frame(video_source)
        
        # 2. Start clicking loop
        cv2.imshow('Calibrate: Click 4 Corners', self.temp_img)
        cv2.setMouseCallback('Calibrate: Click 4 Corners', self.click_event)

        print("Please click the 4 corners of the mat in this order:")
        print("1. Top-Left -> 2. Top-Right -> 3. Bottom-Right -> 4. Bottom-Left")
        
        # Wait until 4 points are clicked
        while len(self.points) < 4:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                 raise Exception("Calibration cancelled.")

        cv2.destroyWindow('Calibrate: Click 4 Corners')

        # 3. Calculate Matrix
        src_pts = np.float32(self.points)
        dst_pts = np.float32([
            [0, 0], 
            [config.MAP_SIZE, 0], 
            [config.MAP_SIZE, config.MAP_SIZE], 
            [0, config.MAP_SIZE]
        ])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        np.save(config.MATRIX_FILE, self.matrix)
        print(f"[INFO] Matrix saved to {config.MATRIX_FILE}")

    def transform_point(self, x, y):
        if self.matrix is None: return 0, 0
        point_vec = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_vec, self.matrix)
        return int(transformed[0][0][0]), int(transformed[0][0][1])