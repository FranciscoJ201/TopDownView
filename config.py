# config.py

# --- INPUTS ---
VIDEO_SOURCE = "standardizedjudo.mp4"   
MODEL_PATH = "yolo26x.pt"         
MATRIX_FILE = "homography_matrix.npy"

# --- MAP SETTINGS ---
MAP_SIZE = 600  

# --- REAL WORLD PHYSICS ---
# The width of the square area you are clicking in meters.
# International Standard: 10m x 10m (including the red danger zone)
MAT_REAL_DIM_METERS = 10.0 
METERS_PER_PIXEL = MAT_REAL_DIM_METERS / MAP_SIZE

# --- VISUALIZATION ---
COLOR_PLAYER = (255, 0, 0)        # Blue
COLOR_REF_OTHER = (128, 128, 128) # Grey
COLOR_TEXT = (0, 0, 0)

# Detection Settings
CONF_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5