import sys
import os
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, 
                             QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# --- CONFIGURATION ---
MAP_SIZE = 600
SMOOTHING_WINDOW = 5
MATRIX_FILE = "homography_matrix.npy"
MODEL_PATH = "yolo26n.pt"  # Replace with your fine-tuned model path

# --- CUSTOM WIDGETS ---

class ClickableVideoLabel(QLabel):
    """Custom Label to capture mouse clicks for calibration."""
    click_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # We need the click relative to the image, not the widget
            # (Assuming the image is scaled to fit, this might need ratio calc)
            # For simplicity, we assume fixed size or handle ratio in main
            self.click_signal.emit(event.pos().x(), event.pos().y())

class JudoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Match Tracker & Mapper")
        self.resize(1400, 800)

        # --- STATE ---
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.setInterval(30) # ~30 FPS
        self.timer.timeout.connect(self.next_frame)

        self.current_frame = None   # The raw CV2 frame
        self.total_frames = 0
        self.is_playing = False
        
        # Homography / Calibration State
        self.homography_matrix = None
        self.calibration_points = []
        self.is_calibrating = False
        
        # Smoothing History: { track_id : deque([(x,y), ...]) }
        self.history = {}

        # --- UI SETUP ---
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)

        # 1. TOP PANEL (Video + Map)
        display_layout = QHBoxLayout()
        
        # Left: Video Feed
        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        self.lbl_video = ClickableVideoLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.lbl_video.setFixedSize(800, 600) # Fixed size ensures click coords map easily
        self.lbl_video.click_signal.connect(self.handle_video_click)
        v_layout.addWidget(QLabel("<b>Camera Feed</b> (Click to Calibrate)"))
        v_layout.addWidget(self.lbl_video)
        display_layout.addWidget(self.video_container)

        # Right: Top-Down Map
        self.map_container = QWidget()
        m_layout = QVBoxLayout(self.map_container)
        self.lbl_map = QLabel()
        self.lbl_map.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_map.setFixedSize(600, 600)
        self.lbl_map.setStyleSheet("background-color: #FFF; border: 1px solid #555;")
        m_layout.addWidget(QLabel("<b>Top-Down View</b>"))
        m_layout.addWidget(self.lbl_map)
        display_layout.addWidget(self.map_container)

        self.main_layout.addLayout(display_layout)

        # 2. CONTROLS PANEL
        controls = QFrame()
        controls.setStyleSheet("background-color: #333; color: white; border-radius: 5px;")
        c_layout = QHBoxLayout(controls)

        self.btn_load = QPushButton("📂 Load Video")
        self.btn_load.clicked.connect(self.load_video)
        c_layout.addWidget(self.btn_load)

        self.btn_calibrate = QPushButton("📐 Calibrate Mat")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        c_layout.addWidget(self.btn_calibrate)

        c_layout.addStretch()

        self.btn_prev = QPushButton("⏮")
        self.btn_prev.clicked.connect(self.prev_frame)
        c_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        c_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("⏭")
        self.btn_next.clicked.connect(self.next_frame)
        c_layout.addWidget(self.btn_next)

        c_layout.addStretch()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.seek_frame)
        c_layout.addWidget(self.slider, stretch=2)

        self.main_layout.addWidget(controls)
        
        # --- INITIALIZATION ---
        self.init_map_bg()
        self.load_saved_matrix()
        
        # Load Model in background (or on startup)
        print("Loading YOLO...")
        self.model = YOLO(MODEL_PATH) 
        print("YOLO Loaded.")

    # --- INITIALIZATION HELPERS ---
    def init_map_bg(self):
        """Creates the default white mat image."""
        self.map_bg = np.ones((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) * 255
        center = MAP_SIZE // 2
        # Draw Red Circle (Combat Area)
        cv2.circle(self.map_bg, (center, center), int(MAP_SIZE * 0.4), (0, 0, 255), 2)
        # Draw Starting Marks
        cv2.line(self.map_bg, (center - 20, center), (center + 20, center), (0,0,255), 2)
        cv2.line(self.map_bg, (center, center - 20), (center, center + 20), (0,0,255), 2)
        self.update_map_display(self.map_bg)

    def load_saved_matrix(self):
        if os.path.exists(MATRIX_FILE):
            try:
                self.homography_matrix = np.load(MATRIX_FILE)
                self.btn_calibrate.setStyleSheet("background-color: #4CAF50; color: white;") # Green
                self.btn_calibrate.setText("📐 Re-Calibrate")
                print("Loaded saved Homography Matrix.")
            except:
                self.homography_matrix = None
        else:
            self.btn_calibrate.setStyleSheet("background-color: #FF9800; color: black;") # Orange
            self.btn_calibrate.setText("📐 Calibrate Needed")

    # --- VIDEO HANDLING ---
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video (*.mp4 *.avi *.mov)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.next_frame() # Show first frame
            self.btn_play.setEnabled(True)

    def update_frame(self):
        if self.cap is None: return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_play()
            return

        self.current_frame = cv2.resize(frame, (800, 600)) # Resize to match widget
        processed_frame = self.current_frame.copy()
        
        # --- LOGIC BRANCHING ---
        if self.is_calibrating:
            self.draw_calibration_points(processed_frame)
        elif self.homography_matrix is not None:
            processed_frame = self.run_tracking(processed_frame)

        # Convert to Qt Image for display
        self.display_image(processed_frame, self.lbl_video)

        # Update slider without triggering seek
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.slider.blockSignals(False)

    # --- TRACKING LOGIC ---
    def run_tracking(self, frame):
        # Run YOLO with Tracking
        results = self.model.track(frame, classes=[0], persist=True, verbose=False)
        
        map_display = self.map_bg.copy()

        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    
                    # 1. Calc Foot (Bottom Center)
                    foot_x = (x1 + x2) / 2
                    foot_y = y2

                    # 2. Smoothing
                    if track_id not in self.history:
                        self.history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                    self.history[track_id].append((foot_x, foot_y))
                    
                    avg_x = sum(p[0] for p in self.history[track_id]) / len(self.history[track_id])
                    avg_y = sum(p[1] for p in self.history[track_id]) / len(self.history[track_id])

                    # 3. Transform
                    map_pt = self.transform_point(avg_x, avg_y)

                    # 4. Draw
                    # On Map
                    if 0 <= map_pt[0] <= MAP_SIZE and 0 <= map_pt[1] <= MAP_SIZE:
                        color = (255, 0, 0) # Blue (BGR)
                        cv2.circle(map_display, map_pt, 10, color, -1)
                        cv2.putText(map_display, f"ID:{track_id}", (map_pt[0]+10, map_pt[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                    
                    # On Video
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.update_map_display(map_display)
        return frame

    def transform_point(self, x, y):
        # Apply Homography
        pt_vec = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_vec, self.homography_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    # --- CALIBRATION LOGIC ---
    def start_calibration(self):
        if self.cap is None: return
        
        self.stop_play()
        self.is_calibrating = True
        self.calibration_points = []
        self.homography_matrix = None # Reset
        
        self.btn_calibrate.setText("Click 4 Corners (TL -> TR -> BR -> BL)")
        self.btn_calibrate.setStyleSheet("background-color: #F44336; color: white;") # Red
        
        # Force redraw to show instructions/clear boxes
        self.update_frame()

    def handle_video_click(self, x, y):
        if not self.is_calibrating: return

        if len(self.calibration_points) < 4:
            self.calibration_points.append((x, y))
            self.update_frame() # Redraw to show the new dot

            if len(self.calibration_points) == 4:
                self.finalize_calibration()

    def draw_calibration_points(self, frame):
        for i, pt in enumerate(self.calibration_points):
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw instructions on frame
        cv2.putText(frame, "CALIBRATION MODE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def finalize_calibration(self):
        src_pts = np.float32(self.calibration_points)
        dst_pts = np.float32([
            [0, 0], 
            [MAP_SIZE, 0], 
            [MAP_SIZE, MAP_SIZE], 
            [0, MAP_SIZE]
        ])

        try:
            self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            np.save(MATRIX_FILE, self.homography_matrix)
            
            self.is_calibrating = False
            self.btn_calibrate.setText("📐 Re-Calibrate")
            self.btn_calibrate.setStyleSheet("background-color: #4CAF50; color: white;") # Green
            QMessageBox.information(self, "Success", "Calibration Complete! Matrix Saved.")
            self.update_frame()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration failed: {e}")
            self.start_calibration()

    # --- VIDEO UTILS ---
    def display_image(self, img, label_widget):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        label_widget.setPixmap(QPixmap.fromImage(qt_img))

    def update_map_display(self, img):
        self.display_image(img, self.lbl_map)

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        self.is_playing = True
        self.btn_play.setText("|| Pause")
        self.timer.start()

    def stop_play(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()

    def next_frame(self):
        if self.cap:
            self.update_frame()

    def prev_frame(self):
        if self.cap:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
            self.update_frame()

    def seek_frame(self, value):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.update_frame()

    def slider_pressed(self):
        self.timer.stop()

    def slider_released(self):
        if self.is_playing:
            self.timer.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JudoTrackerApp()
    window.show()
    sys.exit(app.exec())