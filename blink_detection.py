import numpy as np
import torch
import cv2
from ultralytics import YOLO
from transformers import pipeline

##### Blink Detection #####

#ensure blink detection model is loaded
try:
    blink_pipe = pipeline("image-classification", model="dima806/closed_eyes_image_detection")
    USE_VIT = True
    print("ViT Transformer Model Loaded")
except:
    print("Could not load ViT model")
    USE_VIT = False

def detect_blinks_vit(image, left_eye, right_eye):
    if not USE_VIT:
        return False, False, 0.0, 0.0  # No detection if ViT is unavailable

    # Ensure valid eye regions
    if left_eye is None or right_eye is None:
        return False, False, 0.0, 0.0  # Return default values if eyes are not detected
    
    # Convert OpenCV (BGR) image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop left and right eye regions
    try:
        left_eye_crop = img_rgb[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]]
        right_eye_crop = img_rgb[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]]
    except:
        return False, False, 0.0, 0.0  # Return default values if cropping fails

    # Make predictions
    left_preds = blink_pipe(left_eye_crop)
    right_preds = blink_pipe(right_eye_crop)

    # Get probabilities for closed eyes
    left_eye_prob = max([pred["score"] for pred in left_preds if pred["label"] == "closeEye"], default=0.0)
    right_eye_prob = max([pred["score"] for pred in right_preds if pred["label"] == "closeEye"], default=0.0)

    # Threshold for blink detection (like Eye-Dentify paper)
    left_blink = left_eye_prob >= 0.5
    right_blink = right_eye_prob >= 0.5

    return left_blink, right_blink, left_eye_prob, right_eye_prob