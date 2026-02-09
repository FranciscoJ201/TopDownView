import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import config
from transformer import PerspectiveManager

# --- CONFIGURATION FOR SMOOTHING ---
# How many frames to average (Higher = Smoother but more "lag")
SMOOTHING_WINDOW = 5 

def create_default_map():
    img = np.ones((config.MAP_SIZE, config.MAP_SIZE, 3), dtype=np.uint8) * 255
    center = config.MAP_SIZE // 2
    cv2.circle(img, (center, center), int(config.MAP_SIZE * 0.4), (0, 0, 255), 2)
    cv2.line(img, (center - 20, center), (center + 20, center), (0,0,255), 2)
    cv2.line(img, (center, center - 20), (center, center + 20), (0,0,255), 2)
    return img

def main():
    # 1. Setup
    pm = PerspectiveManager()
    if not pm.load_matrix():
        pm.calibrate(config.VIDEO_SOURCE)

    print(f"[INFO] Loading YOLO model: {config.MODEL_PATH}...")
    model = YOLO(config.MODEL_PATH) 
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if config.MAP_IMAGE_PATH:
        base_map = cv2.imread(config.MAP_IMAGE_PATH)
        base_map = cv2.resize(base_map, (config.MAP_SIZE, config.MAP_SIZE))
    else:
        base_map = create_default_map()

    # Dictionary to store history for smoothing: { track_id : deque([(x,y), ...]) }
    history = {}

    print("[INFO] Starting Tracking. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 2. Run TRACKING (persist=True is critical for ID stability)
        results = model.track(frame, classes=[0], conf=config.CONF_THRESHOLD, persist=True, verbose=False)

        current_map = base_map.copy()

        for result in results:
            if result.boxes.id is not None:
                # Get the IDs and Boxes (Move to CPU numpy)
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()

                for id, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    
                    # Raw Foot Position (Bottom-Center)
                    raw_foot_x = (x1 + x2) / 2
                    raw_foot_y = y2

                    # --- SMOOTHING LOGIC ---
                    if id not in history:
                        history[id] = deque(maxlen=SMOOTHING_WINDOW)
                    
                    history[id].append((raw_foot_x, raw_foot_y))

                    # Calculate Average of the history buffer
                    avg_x = sum(p[0] for p in history[id]) / len(history[id])
                    avg_y = sum(p[1] for p in history[id]) / len(history[id])

                    # Transform the SMOOTHED point
                    map_x, map_y = pm.transform_point(avg_x, avg_y)

                    # --- DRAWING ---
                    # Logic: Filter Out-of-Bounds
                    if 0 <= map_x <= config.MAP_SIZE and 0 <= map_y <= config.MAP_SIZE:
                        dot_color = config.COLOR_PLAYER
                        # Add ID text on map for debugging
                        cv2.putText(current_map, f"ID:{id}", (map_x+10, map_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                    else:
                        dot_color = config.COLOR_REF_OTHER

                    # Draw on Camera View
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {id}", (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw on Map
                    cv2.circle(current_map, (map_x, map_y), 8, dot_color, -1)

        cv2.imshow("Judo Tracker - Smoothed", frame)
        cv2.imshow("Top-Down Map", current_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()