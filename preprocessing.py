import cv2
import mediapipe as mp
import numpy as np

# Define landmark indices (as used in Pupilsense)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Initialize MediaPipe face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def euclidean_distance(pt1, pt2):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def compute_EAR(landmarks, eye_indices, image_shape):
    """
    Computes the Eye Aspect Ratio (EAR) for one eye.
    Uses:
    - Horizontal distance: between points at index 0 and index 8 of the eye landmarks.
    - Vertical distances: between points at index 4 & 12 and between points at index 5 & 11.
    Returns a float EAR value.
    """
    h, w = image_shape[:2]
    # Convert the normalized landmark positions to pixel coordinates.
    points = [ (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices ]
    
    # Calculate horizontal distance (between first and ninth points)
    horizontal = euclidean_distance(points[0], points[8])
    # Calculate vertical distances
    vertical1 = euclidean_distance(points[4], points[12])
    vertical2 = euclidean_distance(points[5], points[11])
    
    # Avoid division by zero
    if horizontal == 0:
        return 0.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def extract_eye_regions(frame, padding=10):
    """
    Extracts left and right eye regions using MediaPipe face mesh.
    Also computes EAR for each eye.
    Returns:
      left_eye_crop, right_eye_crop, left_EAR, right_EAR
    """
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None, None, None  # No face detected

    landmarks = results.multi_face_landmarks[0].landmark

    def get_eye_bbox(eye_indices):
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                            for i in eye_indices])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        # Expand the box by the padding value
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, w)
        y_max = min(y_max + padding, h)
        return frame[y_min:y_max, x_min:x_max]

    left_eye_crop = get_eye_bbox(LEFT_EYE)
    right_eye_crop = get_eye_bbox(RIGHT_EYE)

    # Ensure three channels in case the crops are grayscale.
    if left_eye_crop is not None and left_eye_crop.ndim == 2:
        left_eye_crop = cv2.cvtColor(left_eye_crop, cv2.COLOR_GRAY2RGB)
    if right_eye_crop is not None and right_eye_crop.ndim == 2:
        right_eye_crop = cv2.cvtColor(right_eye_crop, cv2.COLOR_GRAY2RGB)

    # Compute EAR for each eye.
    left_EAR = compute_EAR(landmarks, LEFT_EYE, frame.shape)
    right_EAR = compute_EAR(landmarks, RIGHT_EYE, frame.shape)

    return left_eye_crop, right_eye_crop, left_EAR, right_EAR