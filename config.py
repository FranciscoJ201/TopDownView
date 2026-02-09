# config.py

# --- INPUTS ---
VIDEO_SOURCE = "/Users/franciscojimenez/Desktop/recordings/GX011125.MP4"   # Path to your video file
MODEL_PATH = "yolo26n.pt"         # Path to your YOLO model (use your fine-tuned one here)
MATRIX_FILE = "homography_matrix.npy"

# --- MAP SETTINGS ---
# Dimensions of the top-down map window
MAP_SIZE = 600  

# Path to a top-down image of a mat (optional). 
# If None, the script generates a white mat with a red circle.
MAP_IMAGE_PATH = None 

# --- VISUALIZATION ---
# Colors (B, G, R)
COLOR_PLAYER = (255, 0, 0)      # Blue dots for players
COLOR_REF_OTHER = (128, 128, 128) # Grey for people outside the mat
COLOR_TEXT = (0, 0, 0)

# Confidence threshold to detect a person
CONF_THRESHOLD = 0.5