import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# DeepFace
FACE_MODEL = os.getenv("FACE_MODEL", "Facenet512")
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "yunet")  # yunet = fast on CPU, retinaface = more accurate
DISTANCE_METRIC = "cosine"
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.35"))  # cosine distance threshold for Facenet512

# Anti-spoofing
ANTI_SPOOF_ENABLED = os.getenv("ANTI_SPOOF_ENABLED", "true").lower() == "true"
ANTI_SPOOF_THRESHOLD = float(os.getenv("ANTI_SPOOF_THRESHOLD", "0.5"))

# Multi-frame
MIN_FRAMES = 3
MAX_FRAMES = 5

# Adaptive blend ratio
BLEND_OLD_RATIO = 0.8
BLEND_NEW_RATIO = 0.2

# API Security
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")  # Required: shared secret with Next.js

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Server
PORT = int(os.getenv("PORT", "8000"))
