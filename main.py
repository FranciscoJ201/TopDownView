import sys
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, 
                             QMessageBox, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import config
from transformer import PerspectiveManager

class ClickableVideoLabel(QLabel):
    """Custom Label to capture mouse clicks for calibration."""
    click_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480) 

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.click_signal.emit(event.pos().x(), event.pos().y())

class JudoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Match Tracker & Mapper")
        self.resize(1450, 900)
        
        # --- LOGIC MODULES ---
        self.pm = PerspectiveManager()
        self.pm.load_matrix() 
        
        # --- STATE ---
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.setInterval(30) 
        self.timer.timeout.connect(self.next_frame)

        self.current_frame = None
        self.display_pixmap = None 
        self.total_frames = 0
        self.is_playing = False
        
        self.calibration_points = []
        self.is_calibrating = False
        self.history = {}

        # --- UI SETUP ---
        self.setup_ui()
        self.init_map_bg()
        self.update_calibration_status()
        
        print(f"[INFO] Loading YOLO model: {config.MODEL_PATH}...")
        try:
            self.model = YOLO(config.MODEL_PATH) 
            print("[INFO] YOLO Loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")

        if config.VIDEO_SOURCE:
            self.load_video(config.VIDEO_SOURCE)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)

        # 1. VISUALIZATION AREA
        viz_layout = QHBoxLayout()
        
        # Left: Video (Scalable)
        video_container = QWidget()
        v_layout = QVBoxLayout(video_container)
        self.lbl_video = ClickableVideoLabel()
        # Simple border to show where the video area is
        self.lbl_video.setStyleSheet("border: 1px solid #555;") 
        self.lbl_video.click_signal.connect(self.handle_video_click)
        
        lbl_cam_title = QLabel("<b>Camera Feed</b> (Click to Calibrate)")
        v_layout.addWidget(lbl_cam_title)
        v_layout.addWidget(self.lbl_video)
        viz_layout.addWidget(video_container, stretch=2)

        # Right: Map (Fixed Size)
        map_container = QWidget()
        m_layout = QVBoxLayout(map_container)
        self.lbl_map = QLabel()
        self.lbl_map.setFixedSize(config.MAP_SIZE, config.MAP_SIZE)
        self.lbl_map.setStyleSheet("border: 1px solid #555; background-color: white;")
        self.lbl_map.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_map_title = QLabel("<b>Top-Down Analysis</b>")
        m_layout.addWidget(lbl_map_title)
        m_layout.addWidget(self.lbl_map)
        m_layout.addStretch() 
        viz_layout.addWidget(map_container, stretch=1) 

        self.main_layout.addLayout(viz_layout)

        # 2. CONTROLS AREA
        controls = QFrame()
        c_layout = QHBoxLayout(controls)

        # Load
        self.btn_load = QPushButton("📂 Open Video")
        self.btn_load.clicked.connect(lambda: self.load_video())
        c_layout.addWidget(self.btn_load)

        # Calibrate
        self.btn_calibrate = QPushButton("📐 Calibrate")
        self.btn_calibrate.setCheckable(True)
        self.btn_calibrate.clicked.connect(self.toggle_calibration_mode)
        c_layout.addWidget(self.btn_calibrate)

        c_layout.addSpacing(30)

        # Media Controls
        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.clicked.connect(self.prev_frame)
        c_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        c_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("Next ⏭")
        self.btn_next.clicked.connect(self.next_frame)
        c_layout.addWidget(self.btn_next)

        c_layout.addSpacing(20)

        # Scrubber
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.seek_frame)
        c_layout.addWidget(self.slider)

        self.main_layout.addWidget(controls)

    def init_map_bg(self):
        self.map_bg = np.ones((config.MAP_SIZE, config.MAP_SIZE, 3), dtype=np.uint8) * 255
        center = config.MAP_SIZE // 2
        cv2.circle(self.map_bg, (center, center), int(config.MAP_SIZE * 0.4), (0, 0, 255), 2)
        cv2.line(self.map_bg, (center - 20, center), (center + 20, center), (0,0,255), 2)
        cv2.line(self.map_bg, (center, center - 20), (center, center + 20), (0,0,255), 2)
        self.update_map_display(self.map_bg)

    def update_calibration_status(self):
        if self.pm.matrix is not None:
            self.btn_calibrate.setText("📐 Re-Calibrate (Ready)")
        else:
            self.btn_calibrate.setText("📐 Calibrate Needed")

    # --- VIDEO LOGIC ---
    def load_video(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.next_frame() 
            self.btn_play.setEnabled(True)

    def update_frame(self):
        if self.cap is None: return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_play()
            return

        self.current_frame = frame 
        viz_frame = self.current_frame.copy()

        if self.is_calibrating:
            self.draw_calibration_ui(viz_frame)
        elif self.pm.matrix is not None and self.model is not None:
            viz_frame = self.run_inference_and_map(viz_frame)

        self.display_video_frame(viz_frame)
        
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.slider.blockSignals(False)

    def display_video_frame(self, cv_img):
        """Scales the video frame to fit the GUI label while keeping aspect ratio."""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        
        # 1. Convert to Qt Image
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.display_pixmap = QPixmap.fromImage(qt_img)

        # 2. Scale to fit the label (Keep Aspect Ratio)
        scaled_pixmap = self.display_pixmap.scaled(
            self.lbl_video.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_video.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle window resizing to re-scale the video frame."""
        if self.display_pixmap is not None:
             scaled_pixmap = self.display_pixmap.scaled(
                self.lbl_video.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
             self.lbl_video.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def run_inference_and_map(self, frame):
        # Run Tracking
        results = self.model.track(frame, classes=[0], persist=True, verbose=False, conf=config.CONF_THRESHOLD)
        map_viz = self.map_bg.copy()

        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    foot_x, foot_y = (x1 + x2) / 2, y2

                    if track_id not in self.history:
                        self.history[track_id] = deque(maxlen=config.SMOOTHING_WINDOW)
                    self.history[track_id].append((foot_x, foot_y))
                    
                    avg_x = sum(p[0] for p in self.history[track_id]) / len(self.history[track_id])
                    avg_y = sum(p[1] for p in self.history[track_id]) / len(self.history[track_id])

                    mx, my = self.pm.transform_point(avg_x, avg_y)

                    if 0 <= mx <= config.MAP_SIZE and 0 <= my <= config.MAP_SIZE:
                        cv2.circle(map_viz, (mx, my), 8, config.COLOR_PLAYER, -1)
                        cv2.putText(map_viz, f"ID:{track_id}", (mx+10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.circle(map_viz, (mx, my), 6, config.COLOR_REF_OTHER, -1)

        self.update_map_display(map_viz)
        return frame

    # --- CALIBRATION HANDLERS ---
    def toggle_calibration_mode(self):
        if self.btn_calibrate.isChecked():
            self.stop_play()
            self.is_calibrating = True
            self.calibration_points = []
            self.pm.matrix = None 
            self.btn_calibrate.setText("Click TL -> TR -> BR -> BL")
            self.update_frame()
        else:
            self.is_calibrating = False
            self.pm.load_matrix() 
            self.update_calibration_status()
            self.update_frame()

    def handle_video_click(self, x, y):
        # Scale Click Logic
        if not self.is_calibrating or self.current_frame is None: return

        pixmap = self.lbl_video.pixmap()
        if not pixmap: return
        
        # Calculate offsets (Image is centered)
        img_w = pixmap.width()
        img_h = pixmap.height()
        lbl_w = self.lbl_video.width()
        lbl_h = self.lbl_video.height()
        
        offset_x = (lbl_w - img_w) // 2
        offset_y = (lbl_h - img_h) // 2

        rel_x = x - offset_x
        rel_y = y - offset_y

        if 0 <= rel_x < img_w and 0 <= rel_y < img_h:
            orig_h, orig_w = self.current_frame.shape[:2]
            scale_x = orig_w / img_w
            scale_y = orig_h / img_h
            
            final_x = int(rel_x * scale_x)
            final_y = int(rel_y * scale_y)

            self.add_calibration_point(final_x, final_y)

    def add_calibration_point(self, x, y):
        if len(self.calibration_points) < 4:
            self.calibration_points.append((x, y))
            self.update_frame()

            if len(self.calibration_points) == 4:
                success = self.pm.compute_and_save_matrix(self.calibration_points)
                if success:
                    QMessageBox.information(self, "Success", "Calibration Matrix Saved!")
                else:
                    QMessageBox.warning(self, "Error", "Failed to compute matrix.")
                
                self.is_calibrating = False
                self.btn_calibrate.setChecked(False)
                self.update_calibration_status()
                self.update_frame()

    def draw_calibration_ui(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        for i, pt in enumerate(self.calibration_points):
            cv2.circle(frame, pt, 8, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (pt[0]+15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, "CALIBRATION MODE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # --- HELPER UTILS ---
    def update_map_display(self, img):
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.lbl_map.setPixmap(QPixmap.fromImage(qt_img))

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
        self.update_frame()

    def prev_frame(self):
        if self.cap:
            curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr - 2))
            self.update_frame()

    def seek_frame(self, val):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
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