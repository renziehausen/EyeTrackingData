import cv2
import os
import time
import pandas as pd

#own modules
from camera_input import capture_selected_camera_feed
from preprocessing import extract_eye_regions
from models import predict_pupil_diameter, left_model_18, right_model_18, left_model_50, right_model_50
from blink_detection import detect_blinks_vit
from preprocessing import extract_eye_regions  # Assuming this extracts landmarks

####################################################
# Enter particpant ID
participant_id = input("Enter participant ID: ").strip()

# make sure output directory exists
os.makedirs("outputs", exist_ok=True)

#defining file paths
csv_file = f"outputs/participant_{participant_id}.csv" #csv file for data that is captured on the go (depends on the hardware, does not always work perfectly fine)
processed_csv_file = f"outputs/participant_{participant_id}_processed.csv" #rerun the processing with more time on the saved video.
# video file names later as they need to contain the index of the camera

# Define CSV columns
df = pd.DataFrame(columns=[
    "Timestamp", "Frame Number",
    "Left_ResNet18", "Right_ResNet18", "Left_ResNet50", "Right_ResNet50",
    "ViT Left Blink", "ViT Right Blink", "ViT Left Prob", "ViT Right Prob",
    "Intervention"
])

# start camera feeds
caps = capture_selected_camera_feed() #this function is in the camera input file
if not caps:
    print("somethings wrong with the camera... :/")
    exit()

#### recording the camera feed
# initialising the video writers, one per camera
video_writers = {}
frame_counters = {i: 0 for i in range(len(caps))}
fps_target = 30  # target frames per second for capture
frame_interval = 1 / fps_target
last_frame_time = time.time()


fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i, cap in enumerate(caps):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640 #initial width and height of the input video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 # fall back to 30 if necessary

    video_file = f"outputs/participant_{participant_id}_cam{i}.mp4" # the recording itself
    print(f"[INFO] Initializing VideoWriter for Camera {i}: {frame_width}x{frame_height}, {fps} FPS -> {video_file}")
    video_writers[i] = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

print("Live preview running.")
print("Press 's' in any preview window to start recording, press space to mark an intervention event, or 'q' to quit.")

# global flags and counters
recording = False
frame_count = 0
intervention_counter = 0


while True:
    current_time = time.time()
    if current_time - last_frame_time < frame_interval:
        # Sleep a bit to avoid maxing out the CPU
        time.sleep(0.001)
        continue  # Maintain FPS

    last_frame_time = current_time  # Update frame time

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {i} feed unavailable. Exiting...")
            continue

        frame_counters[i] += 1  # Increment frame counter for this camera
        frame_count += 1

        left_eye, right_eye = extract_eye_regions(frame)
        # Ensure eyes are detected
        if left_eye is None or right_eye is None:
            continue  # Skip frame if eyes are not detected

        # Predict pupil diameter
        left_18 = predict_pupil_diameter(left_model_18, left_eye)
        right_18 = predict_pupil_diameter(right_model_18, right_eye)
        left_50 = predict_pupil_diameter(left_model_50, left_eye)
        right_50 = predict_pupil_diameter(right_model_50, right_eye)

        # Blink detection
        # Blink detection using ViT
        vit_left_blink, vit_right_blink, vit_left_prob, vit_right_prob = detect_blinks_vit(frame, left_eye, right_eye)

        # Display results on screen
        cv2.putText(frame, f"L18: {left_18:.2f}mm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"R18: {right_18:.2f}mm", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"L50: {left_50:.2f}mm", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"R50: {right_50:.2f}mm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"ViT Left Blink: {vit_left_blink}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"ViT Right Blink: {vit_right_blink}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # show recording status
        if recording:
            cv2.putText(frame, "REC", (frame.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "PREVIEW", (frame.shape[1]-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        cv2.imshow(f"Camera {i} Feed", frame)

        # Write frame to video file if recording is acvtive
        if recording:
            video_writers[i].write(frame)
            # Save results
            df.loc[len(df)] = [
            pd.Timestamp.now(), frame_count, left_18, right_18, left_50, right_50, 
            vit_left_blink, vit_right_blink, vit_left_prob, vit_right_prob, ""
            ]
            df.to_csv(csv_file, index=False)

    # Check for key presses.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        print("Recording started!")
        recording = True
    elif key == ord(' '):  # space bar pressed: mark an intervention event.
        if recording:
            intervention_counter += 1
            # Record a separate row for the intervention event.
            df.loc[len(df)] = [
                pd.Timestamp.now(), frame_count,
                None, None, None, None,
                None, None, None, None,
                intervention_counter  # record the event number
            ]
            print(f"Intervention event {intervention_counter} recorded at frame {frame_count}!")
            df.to_csv(csv_file, index=False)
    elif key == ord('q'):
        print("Exiting...")
        break

# Release all camera resources
for cap in caps:
    cap.release()
for writer in video_writers.values():
    writer.release()
cv2.destroyAllWindows()

print("Video saved successfully in 'outputs' folder.")