import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import mediapipe as mp

# If these constants and face_mesh are defined in your preprocessing file,
# you can import them. For example:
# from preprocessing import LEFT_EYE, RIGHT_EYE, face_mesh, extract_eye_regions
# For this example, we redefine them here:

# Define landmark indices (as used in Pupilsense)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Initialize MediaPipe face mesh (you may want to create only one instance for your app)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# EAR computation functions
# ---------------------------
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def compute_EAR(landmarks, eye_indices, image_shape):
    """
    Computes the Eye Aspect Ratio (EAR) given the landmarks for an eye.
    landmarks: list of MediaPipe landmarks
    eye_indices: list of indices for the eye (e.g. LEFT_EYE)
    image_shape: tuple (height, width, channels)
    Returns a float EAR value.
    """
    h, w = image_shape[:2]
    # Convert normalized landmarks to pixel coordinates.
    points = [ (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices ]
    points = np.array(points)
    
    # Using a common EAR formula:
    # horizontal distance between landmark[0] and landmark[8]
    horizontal = euclidean_distance(points[0], points[8])
    # vertical distances between landmark[4] & landmark[12] and between landmark[5] & landmark[11]
    vertical1 = euclidean_distance(points[4], points[12])
    vertical2 = euclidean_distance(points[5], points[11])
    
    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_blink_with_EAR(frame):
    """
    Runs MediaPipe face mesh on the frame to compute EAR for each eye.
    Returns:
    avg_EAR (float), left_EAR (float), right_EAR (float), landmarks (if detected)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    left_EAR = compute_EAR(landmarks, LEFT_EYE, frame.shape)
    right_EAR = compute_EAR(landmarks, RIGHT_EYE, frame.shape)
    avg_EAR = (left_EAR + right_EAR) / 2.0
    return avg_EAR, left_EAR, right_EAR, landmarks

# ---------------------------
# ViT-based blink detection
# ---------------------------
try:
    blink_pipe = pipeline("image-classification", model="dima806/closed_eyes_image_detection")
    USE_VIT = True
    print("ViT Transformer Model Loaded")
except Exception as e:
    print("Could not load ViT model:", e)
    USE_VIT = False

def detect_blinks_vit(frame, left_eye, right_eye):
    """
    Detects blinks using a Vision Transformer model.
    Returns:
    left_blink (bool), right_blink (bool),
    left_eye_prob (float), right_eye_prob (float)
    """
    if not USE_VIT or left_eye is None or right_eye is None:
        return False, False, 0.0, 0.0

    desired_size = (224, 224)
    # Convert eye images from BGR to RGB and resize.
    left_eye_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
    right_eye_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
    left_eye_resized = cv2.resize(left_eye_rgb, desired_size, interpolation=cv2.INTER_CUBIC)
    right_eye_resized = cv2.resize(right_eye_rgb, desired_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert to PIL images.
    left_eye_pil = Image.fromarray(left_eye_resized)
    right_eye_pil = Image.fromarray(right_eye_resized)
    
    # Run the ViT model.
    left_preds = blink_pipe(left_eye_pil)
    right_preds = blink_pipe(right_eye_pil)
    
    left_eye_prob = max([pred["score"] for pred in left_preds if pred["label"] == "closeEye"], default=0.0)
    right_eye_prob = max([pred["score"] for pred in right_preds if pred["label"] == "closeEye"], default=0.0)
    
    left_blink = left_eye_prob >= 0.5
    right_blink = right_eye_prob >= 0.5
    
    return left_blink, right_blink, left_eye_prob, right_eye_prob

# ---------------------------
# Combined blink detection function
# ---------------------------
# Set EAR thresholds (adjust as needed)
BLINK_LOWER_THRESH = 0.22
BLINK_UPPER_THRESH = 0.25

def detect_blink(frame, extract_eye_regions_func):
    """
    Detects blink for a given frame by first computing the EAR.
    If EAR is ambiguous (between BLINK_LOWER_THRESH and BLINK_UPPER_THRESH),
    it uses the ViT-based model to confirm blink detection.
    
    extract_eye_regions_func: a function that takes a frame and returns (left_eye, right_eye, left_EAR, right_EAR)
    Returns:
    blinked (bool), avg_EAR (float), left_eye_prob (float), right_eye_prob (float)
    """
    avg_EAR, left_EAR, right_EAR, landmarks = detect_blink_with_EAR(frame)
    if avg_EAR is None:
        # No face detected; assume no blink.
        return False, None, 0.0, 0.0
    
    # Quick decision based on EAR.
    if avg_EAR <= BLINK_LOWER_THRESH:
        print("Blink detected based on EAR (below lower threshold).")
        return True, avg_EAR, 0.0, 0.0
    elif avg_EAR > BLINK_UPPER_THRESH:
        return False, avg_EAR, 0.0, 0.0
    else:
        # Ambiguous EAR; use ViT-based detection.
        # Use the provided extraction function to get eye crops.
        left_eye, right_eye, _, _ = extract_eye_regions_func(frame)
        vit_left_blink, vit_right_blink, left_eye_prob, right_eye_prob = detect_blinks_vit(frame, left_eye, right_eye)
        blinked = vit_left_blink or vit_right_blink
        print("Ambiguous EAR; confirmed blink via ViT:", blinked)
        return blinked, avg_EAR, left_eye_prob, right_eye_prob

# Example usage (this part would be in your main loop, not at module level):
# blinked, avg_EAR, left_prob, right_prob = detect_blink(frame, extract_eye_regions)