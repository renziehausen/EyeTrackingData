import numpy as np
import torch
import cv2
from ultralytics import YOLO
from transformers import pipeline

##### Blink Detection Models #####

# Load Yolov8
try:
    blink_detector = YOLO("yolov8n-eyes.pt")  # Replace with correct model path
    USE_YOLO = True
    print("✅ YOLOv8 Model Loaded")
except:
    print("⚠️ YOLOv8 model not found, disabling YOLO")
    USE_YOLO = False

# load ViT Transformer (used in the eye dentify paper)
try:
    blink_pipe = pipeline("image-classification", model="dima806/closed_eyes_image_detection")
    USE_VIT = True
    print("✅ ViT Transformer Model Loaded")
except:
    print("⚠️ Could not load online ViT model")
    USE_VIT = False

##### Ear based blink detection #####

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_blinks_ear(landmarks, left_eye_idx, right_eye_idx):
    """Detect blinks using EAR."""
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_idx])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_idx])

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)

    left_blink = left_EAR < 0.2
    right_blink = right_EAR < 0.2

    return left_blink, right_blink

##### ViT Blink Detecion #####
def detect_blinks_vit(image):
    """Detect blinks using ViT (Hugging Face)."""
    img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds = blink_pipe(img_pil)
    return any(pred["label"] == "closeEye" and pred["score"] >= 0.5 for pred in preds)

##### yolov8 based blink detection #####
def detect_blinks_yolo(frame):
    """Detect blinks using YOLOv8."""
    results = blink_detector(frame)
    closed_eyes = sum(1 for r in results if r.probs[0] > 0.8)  # If YOLO detects closed eyes
    return closed_eyes > 0  # True if detected

##### Final blink detection #####

def detect_blinks(image, landmarks, left_eye_idx, right_eye_idx):
    """Run all available blink detection methods and return results."""
    
    blink_detected = False

    # 1️⃣ Try YOLO first (fastest)
    if USE_YOLO:
        blink_detected = detect_blinks_yolo(image)

    # 2️⃣ Try ViT if available
    elif USE_VIT:
        blink_detected = detect_blinks_vit(image)

    # 3️⃣ Fallback to EAR
    if not blink_detected:
        left_blink, right_blink = detect_blinks_ear(landmarks, left_eye_idx, right_eye_idx)
        blink_detected = left_blink or right_blink

    return blink_detected