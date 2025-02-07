import cv2
import os
import pandas as pd
from camera_input import capture_dual_camera
from preprocessing import extract_eye_regions
from models import predict_pupil_diameter, left_model_18, right_model_18, left_model_50, right_model_50
from blink_detection import detect_blinks
from preprocessing import extract_eye_regions  # Assuming this extracts landmarks
from blink_detection import detect_blinks

# Start dual camera feed (webcam + second feed placeholder)
cap1, cap2 = capture_dual_camera(0)  # Change second cam ID when needed

csv_file = "outputs/pupil_diameter_log.csv"
df = pd.DataFrame(columns=["Timestamp", "Left_ResNet18", "Right_ResNet18", "Left_ResNet50", "Right_ResNet50", "Left_Blink", "Right_Blink"])

while cap1.isOpened():
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    left_eye, right_eye = extract_eye_regions(frame1)
    
    # Predict pupil diameter using both models
    left_18 = predict_pupil_diameter(left_model_18, left_eye)
    right_18 = predict_pupil_diameter(right_model_18, right_eye)
    left_50 = predict_pupil_diameter(left_model_50, left_eye)
    right_50 = predict_pupil_diameter(right_model_50, right_eye)

    # Extract landmarks first
    eyes_data = extract_eye_regions(frame1)

    if eyes_data is not None:
        landmarks = eyes_data["landmarks"]  # Ensure extract_eye_regions returns landmarks
        left_eye_idx = eyes_data["left_eye_idx"]
        right_eye_idx = eyes_data["right_eye_idx"]

    # Now call detect_blinks
    left_blink, right_blink = detect_blinks(frame1, landmarks, left_eye_idx, right_eye_idx)

    # Blink detection
    left_blink, right_blink = detect_blinks(frame1, landmarks, left_eye_idx, right_eye_idx)

    # Display results on screen
    cv2.putText(frame1, f"L18: {left_18:.2f}mm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"R18: {right_18:.2f}mm", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Tracking", frame1)

    # Save results
    df.loc[len(df)] = [pd.Timestamp.now(), left_18, right_18, left_50, right_50, left_blink, right_blink]
    df.to_csv(csv_file, index=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()