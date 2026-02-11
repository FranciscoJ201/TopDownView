# main_gui.py
import sys
import cv2
import numpy as np
from collections import deque

# PyQt Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, 
                             QMessageBox, QFrame, QSizePolicy, QTextEdit, QProgressDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

import config
from transformer import PerspectiveManager
from analyzer import MatchAnalyzer

# --- WORKER THREAD FOR ANALYSIS (Prevents GUI Freeze) ---
class AnalysisWorker(QThread):
    progress_sig = pyqtSignal(int)
    finished_sig = pyqtSignal(dict)

    def __init__(self, analyzer, video_path):
        super().__init__()
        self.analyzer = analyzer
        self.video_path = video_path

    def run(self):
        tracks = self.analyzer.analyze_video(self.video_path, self.update_progress)
        self.analyzer.save_tracks(tracks, self.video_path)
        self.finished_sig.emit(tracks)

    def update_progress(self, val):
        self.progress_sig.emit(val)

# --- MAIN GUI ---
class ClickableVideoLabel(QLabel):
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
        self.setWindowTitle("Judo Match Tracker (Pre-Processed)")
        self.resize(1450, 950)
        
        self.pm = PerspectiveManager()
        self.pm.load_matrix()
        
        # Initialize Analyzer (Loads YOLO once)
        print(f"[INFO] Initializing Analyzer with {config.MODEL_PATH}...")
        self.analyzer = MatchAnalyzer(config.MODEL_PATH)
        
        self.cap = None
        self.tracks = None # Dictionary of pre-computed tracks
        
        self.timer = QTimer()
        self.timer.setInterval(30) 
        self.timer.timeout.connect(self.next_frame)

        self.current_frame = None
        self.display_pixmap = None 
        self.is_playing = False
        self.is_calibrating = False
        self.calibration_points = []
        
        # Stats & History
        self.history = {} 

        self.setup_ui()
        self.init_map_bg()
        self.update_calibration_status()

        if config.VIDEO_SOURCE:
            self.load_video(config.VIDEO_SOURCE)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Viz Area
        viz_layout = QHBoxLayout()
        
        # Video
        vid_container = QWidget()
        v_layout = QVBoxLayout(vid_container)
        self.lbl_video = ClickableVideoLabel()
        self.lbl_video.setStyleSheet("border: 1px solid #555;")
        self.lbl_video.click_signal.connect(self.handle_video_click)
        v_layout.addWidget(QLabel("<b>Camera Feed</b>"))
        v_layout.addWidget(self.lbl_video)
        viz_layout.addWidget(vid_container, stretch=2)

        # Map
        map_container = QWidget()
        m_layout = QVBoxLayout(map_container)
        self.lbl_map = QLabel()
        self.lbl_map.setFixedSize(config.MAP_SIZE, config.MAP_SIZE)
        self.lbl_map.setStyleSheet("border: 1px solid #555; background: white;")
        m_layout.addWidget(QLabel("<b>Top-Down Analysis</b>"))
        m_layout.addWidget(self.lbl_map)
        
        # Stats
        m_layout.addSpacing(10)
        m_layout.addWidget(QLabel("<b>Live Radial Stats</b>"))
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMaximumHeight(150)
        m_layout.addWidget(self.txt_stats)
        m_layout.addStretch()
        viz_layout.addWidget(map_container, stretch=1)

        layout.addLayout(viz_layout)

        # Controls
        controls = QFrame()
        c_layout = QHBoxLayout(controls)
        
        self.btn_load = QPushButton("📂 Open Video")
        self.btn_load.clicked.connect(lambda: self.load_video())
        c_layout.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("⚙️ Re-Analyze")
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        c_layout.addWidget(self.btn_analyze)

        self.btn_calibrate = QPushButton("📐 Calibrate")
        self.btn_calibrate.setCheckable(True)
        self.btn_calibrate.clicked.connect(self.toggle_calibration_mode)
        c_layout.addWidget(self.btn_calibrate)

        c_layout.addSpacing(20)
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        c_layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.seek_frame)
        c_layout.addWidget(self.slider)

        layout.addWidget(controls)

    # --- VIDEO LOADING & ANALYSIS ---
    def load_video(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi)")
        if not path: return

        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.btn_analyze.setEnabled(True)

        # Check for existing tracks
        loaded_tracks = self.analyzer.load_tracks(path)
        if loaded_tracks:
            self.tracks = loaded_tracks
            print(f"[INFO] Tracks loaded for {len(self.tracks)} frames.")
            self.next_frame()
        else:
            # Prompt for analysis
            reply = QMessageBox.question(self, "Analysis Needed", 
                                         "No pre-computed tracks found. Analyze video now?\n(This allows perfect scrubbing)",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.run_analysis()
            else:
                self.tracks = {} # Empty, will show nothing
                self.next_frame()

    def run_analysis(self):
        self.btn_analyze.setEnabled(False)
        self.pbar = QProgressDialog("Analyzing Video... (YOLO + ByteTrack)", "Cancel", 0, 100, self)
        self.pbar.setWindowModality(Qt.WindowModality.WindowModal)
        self.pbar.show()

        self.worker = AnalysisWorker(self.analyzer, self.video_path)
        self.worker.progress_sig.connect(self.pbar.setValue)
        self.worker.finished_sig.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, tracks):
        self.tracks = tracks
        self.btn_analyze.setEnabled(True)
        self.pbar.close()
        QMessageBox.information(self, "Done", "Analysis Complete! Tracks saved.")
        self.next_frame()

    # --- PLAYBACK & DISPLAY ---
    def update_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: 
            self.stop_play()
            return

        self.current_frame = frame
        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 # 0-based index

        viz_frame = frame.copy()

        if self.is_calibrating:
            self.draw_calibration_ui(viz_frame)
        elif self.tracks and self.pm.matrix is not None:
            # LOOKUP MODE: No Inference, just drawing!
            self.draw_tracks_from_memory(frame_idx, viz_frame)

        self.display_video_frame(viz_frame)
        
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

    def draw_tracks_from_memory(self, frame_idx, frame):
        # Safety check: if we skipped past analysis limit
        if frame_idx not in self.tracks: return

        frame_data = self.tracks[frame_idx]
        map_viz = self.map_bg.copy()
        center = (config.MAP_SIZE//2, config.MAP_SIZE//2)
        stats_text = ""

        for obj in frame_data:
            tid = obj['id']
            box = obj['box']
            foot = obj['foot']

            # Update History (For smoothing if you want, or just raw)
            # Since we pre-processed, we can assume data is decent, 
            # but we can still smooth if we keep a history buffer relevant to frame_idx
            
            # Transform
            mx, my = self.pm.transform_point(foot[0], foot[1])

            if 0 <= mx <= config.MAP_SIZE and 0 <= my <= config.MAP_SIZE:
                # Dist Calc
                dist_px = np.sqrt((mx - center[0])**2 + (my - center[1])**2)
                dist_m = dist_px * config.METERS_PER_PIXEL
                stats_text += f"ID {tid}: {dist_m:.2f}m\n"

                # Draw Map
                cv2.circle(map_viz, (mx, my), 8, config.COLOR_PLAYER, -1)
                cv2.putText(map_viz, str(tid), (mx+10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
                # Draw Video
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                 cv2.circle(map_viz, (mx, my), 6, config.COLOR_REF_OTHER, -1)

        self.update_map_display(map_viz)
        self.txt_stats.setText(stats_text)

    # --- UTILS (Same as before) ---
    def init_map_bg(self):
        self.map_bg = np.ones((config.MAP_SIZE, config.MAP_SIZE, 3), dtype=np.uint8) * 255
        center = (config.MAP_SIZE // 2, config.MAP_SIZE // 2)
        r_danger = int((4.0 / config.MAT_REAL_DIM_METERS) * config.MAP_SIZE)
        r_safe = int((2.0 / config.MAT_REAL_DIM_METERS) * config.MAP_SIZE)
        cv2.circle(self.map_bg, center, r_danger, (0, 0, 255), 2) 
        cv2.circle(self.map_bg, center, r_safe, (0, 200, 0), 1)
        self.update_map_display(self.map_bg)

    def display_video_frame(self, cv_img):
        h, w, ch = cv_img.shape
        qt_img = QImage(cv_img.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def update_map_display(self, img):
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.lbl_map.setPixmap(QPixmap.fromImage(qt_img))

    # --- CALIBRATION ---
    def update_calibration_status(self):
        if self.pm.matrix is not None:
            self.btn_calibrate.setText("📐 Re-Calibrate")
        else:
            self.btn_calibrate.setText("📐 Calibrate Needed")

    def toggle_calibration_mode(self):
        if self.btn_calibrate.isChecked():
            self.stop_play()
            self.is_calibrating = True
            self.calibration_points = []
            self.pm.matrix = None
            self.btn_calibrate.setText("Click 4 Corners")
            self.update_frame()
        else:
            self.is_calibrating = False
            self.pm.load_matrix()
            self.update_calibration_status()
            self.update_frame()

    def handle_video_click(self, x, y):
        # ... (Same scaling logic as your previous file) ...
        # For brevity, reusing standard scaling logic:
        if not self.is_calibrating or self.current_frame is None: return
        pixmap = self.lbl_video.pixmap()
        if not pixmap: return
        img_w, img_h = pixmap.width(), pixmap.height()
        lbl_w, lbl_h = self.lbl_video.width(), self.lbl_video.height()
        off_x, off_y = (lbl_w - img_w)//2, (lbl_h - img_h)//2
        rel_x, rel_y = x - off_x, y - off_y
        if 0 <= rel_x < img_w and 0 <= rel_y < img_h:
            orig_h, orig_w = self.current_frame.shape[:2]
            fx = int(rel_x * (orig_w/img_w))
            fy = int(rel_y * (orig_h/img_h))
            if len(self.calibration_points) < 4:
                self.calibration_points.append((fx, fy))
                if len(self.calibration_points) == 4:
                    if self.pm.compute_and_save_matrix(self.calibration_points):
                        QMessageBox.information(self, "Success", "Matrix Saved!")
                    self.is_calibrating = False
                    self.btn_calibrate.setChecked(False)
                    self.update_calibration_status()
                self.update_frame()

    def draw_calibration_ui(self, frame):
        for i, pt in enumerate(self.calibration_points):
            cv2.circle(frame, pt, 8, (0,0,255), -1)
            cv2.putText(frame, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # --- CONTROLS ---
    def toggle_play(self):
        if self.is_playing: self.stop_play()
        else: self.start_play()
    def start_play(self):
        self.is_playing = True
        self.btn_play.setText("|| Pause")
        self.timer.start()
    def stop_play(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()
    def next_frame(self): self.update_frame()
    def prev_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-2))
            self.update_frame()
    def seek_frame(self, val):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.update_frame()
    def slider_pressed(self): self.timer.stop()
    def slider_released(self): 
        if self.is_playing: self.timer.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JudoTrackerApp()
    window.show()
    sys.exit(app.exec())