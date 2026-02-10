# config.py

# --- INPUTS ---
# Path to your video file
VIDEO_SOURCE = "/Users/franciscojimenez/Desktop/recordings/GX011125.MP4"   
# Path to your fine-tuned YOLO model
MODEL_PATH = "yolo26n.pt"         
# Where the matrix will be saved
MATRIX_FILE = "homography_matrix.npy"

# --- MAP SETTINGS ---
# Dimensions of the top-down map window (Square)
MAP_SIZE = 400  

# --- VISUALIZATION ---
# Colors (B, G, R)
COLOR_PLAYER = (255, 0, 0)        # Blue dots
COLOR_REF_OTHER = (128, 128, 128) # Grey dots
COLOR_TEXT = (0, 0, 0)

# Detection Settings
CONF_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5  # Frames to average for smoother dots