import cv2
import mediapipe as mp

# Load MediaPipe models
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def extract_eye_regions(frame):
    """Extract left and right eye regions."""
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in RIGHT_EYE]

        return left_eye, right_eye
    return None, None