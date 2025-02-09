import cv2
import mediapipe as mp
import numpy as np
import cv2

# Load MediaPipe models
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def extract_eye_regions(frame):
    """Extract left and right eye regions using MediaPipe (Eye-Dentify method)."""
    h, w, _ = frame.shape  # Get frame dimensions

    # Convert frame to RGB and process with MediaPipe
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None  # No face detected

    landmarks = results.multi_face_landmarks[0].landmark

    # Get eye bounding boxes
    def get_eye_bbox(eye_indices):
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Define fixed size (32×16 like in Eye-Dentify)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        target_width, target_height = 32, 16

        x1 = max(center_x - target_width // 2, 0)
        y1 = max(center_y - target_height // 2, 0)
        x2 = min(x1 + target_width, w)
        y2 = min(y1 + target_height, h)

        return frame[y1:y2, x1:x2]

    left_eye_crop = get_eye_bbox(LEFT_EYE)
    right_eye_crop = get_eye_bbox(RIGHT_EYE)

    # Convert to RGB if grayscale (some OpenCV operations return single-channel)
    if left_eye_crop.ndim == 2:
        left_eye_crop = cv2.cvtColor(left_eye_crop, cv2.COLOR_GRAY2RGB)
    if right_eye_crop.ndim == 2:
        right_eye_crop = cv2.cvtColor(right_eye_crop, cv2.COLOR_GRAY2RGB)

    return left_eye_crop, right_eye_crop