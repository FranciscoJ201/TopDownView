import cv2
import json
import os
import torch
import numpy as np
from ultralytics import YOLO
import config

class MatchAnalyzer:
    def __init__(self, model_path):
        self.model = self._load_model_smart(model_path)

    def _load_model_smart(self, pt_path):
        """
        Smart loader that prioritizes GPU/TensorRT (.engine) for speed.
        If optimization fails (missing libraries/drivers), it falls back to standard (.pt).
        """
        # Define the target engine path (e.g., 'yolo26x.engine')
        base_name = os.path.splitext(pt_path)[0]
        engine_path = base_name + ".engine"
        
        # 1. Try High-Speed GPU (TensorRT)
        if torch.cuda.is_available():
            try:
                # A. Check if engine already exists
                if os.path.exists(engine_path):
                    print(f"[INFO] 🚀 Loading TensorRT Engine: {engine_path}")
                    return YOLO(engine_path)
                
                # B. Try to Export (Requires 'tensorrt' library)
                else:
                    print(f"[INFO] ⚡ GPU Detected! Attempting export to TensorRT...")
                    print(f"       (This requires specific libraries. If it fails, we fallback to standard GPU/CPU.)")
                    
                    if os.path.exists(pt_path):
                        # Load pure PyTorch model first
                        temp_model = YOLO(pt_path)
                        # Attempt Export
                        temp_model.export(format='engine', half=True)
                        
                        print(f"[INFO] Export Complete. Loading Engine...")
                        return YOLO(engine_path)
                    else:
                        print(f"[ERROR] Source PT file missing: {pt_path}")

            except Exception as e:
                # C. SAFETY NET: If export fails (missing libs), catch it here.
                print(f"[WARNING] GPU Optimization failed: {e}")
                print("[INFO] Falling back to standard model loader.")
        
        # 2. Fallback (Standard PyTorch)
        # This runs on CPU (or standard GPU if drivers are okay but TensorRT is missing)
        print(f"[INFO] Loading Standard Model: {pt_path}")
        return YOLO(pt_path)

    def analyze_video(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        tracks_data = {}
        frame_idx = 0

        # Use custom config if available
        tracker_config = "judo_tracker.yaml" if os.path.exists("judo_tracker.yaml") else "bytetrack.yaml"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run Tracking
            results = self.model.track(
                frame, 
                classes=[0], 
                persist=True, 
                verbose=False, 
                tracker=tracker_config,
                conf=config.CONF_THRESHOLD
            )

            frame_tracks = []
            
            for result in results:
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(float, box)
                        foot_x = (x1 + x2) / 2
                        foot_y = y2
                        
                        track_info = {
                            "id": int(track_id),
                            "box": [x1, y1, x2, y2],
                            "foot": [foot_x, foot_y]
                        }
                        frame_tracks.append(track_info)

            tracks_data[frame_idx] = frame_tracks
            
            frame_idx += 1
            if progress_callback:
                progress_callback(int((frame_idx / total_frames) * 100))

        cap.release()
        return tracks_data

    def save_tracks(self, tracks, video_path):
        base_name = os.path.splitext(video_path)[0]
        json_path = base_name + "_tracks.json"
        
        with open(json_path, 'w') as f:
            json.dump(tracks, f)
        print(f"[INFO] Tracks saved to {json_path}")
        return json_path

    def load_tracks(self, video_path):
        base_name = os.path.splitext(video_path)[0]
        json_path = base_name + "_tracks.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        return None