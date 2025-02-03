import cv2

def capture_camera(feed_id = 0):
    #capture video from webcam
    cap = cv2.VideoCapture(feed_id)
    if not cap.isOpened():
        print("Camera not accessible!")
        return None

    return cap

def capture_dual_camera(camera_1=0, camera_2=None):
    """Capture from two cameras: webcam + iPhone (later)."""
    cap1 = capture_camera(camera_1)
    cap2 = capture_camera(camera_2) if camera_2 else None
    return cap1, cap2