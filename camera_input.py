import cv2

def available_cameras(max_cameras = 20):
    print("Searching for cameras...")
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[{i}]Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
        else: 
            print(f"No camera fround at index[{i}]")

    if not available_cameras:
        print("no available cameras detected")
    return available_cameras

def select_camera():
    # get a list of all available cameras
    sel_available_cameras = available_cameras()
    if not sel_available_cameras:
        print("No available cameras detected")
        return []
    
    # give the user a selection of available cameras
    print("\nAvailable Cameras:")
    for cam in sel_available_cameras:
        print(f"[{cam}] Camera {cam}")
        
    # let the user select a camera
    while True:
        # enter the input
        selected_indices = input("Enter the camera indices you want to use (comma-separated): [Hit 'q' key to exit] ")
        # if one wants to quit
        if selected_indices.lower() == "q":
            print("Exiting camera selection")
            return []

        selected_cameras = [int(idx) for idx in selected_indices.split(',') if idx.strip().isdigit() and int(idx) in sel_available_cameras]

        # for invalid input
        if not selected_cameras:
            print("invalid camera choice, select a valid camera index")
            continue

        # checks if cameras are working
        working_cameras = []
        for cam in selected_cameras:
            cap = cv2.VideoCapture(cam)
            if cap.isOpened():
                working_cameras.append(cam)
                cap.release()
            else:
                print(f"Skipped {cam} because it could not be opened")

        if working_cameras:
            return working_cameras
        else:
            print("No working cameras available. Press q to quit or select again")

        

# captures the selected cameras and gives back the camera feed
def capture_camera(feed_id):
    print(f"Accessing camera at index {feed_id}")
    cap = cv2.VideoCapture(feed_id)

    if not cap.isOpened():
        print(f"Camera at inded {feed_id} is not accessible")
        return None
    
    print(f"Camera {feed_id} opened successfully")
    return cap

## captures the selected camera feed and gives them back as a list
def capture_selected_camera_feed():
    selected_camera = select_camera()
    caps = [capture_camera(cam) for cam in selected_camera]
    return [cap for cap in caps if cap is not None]

# displazs the camera
def display_camera_feed():
    caps = capture_selected_camera_feed()
    if not caps:
        return

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:  # If frame capture fails, exit the loop
                print(f"Failed to capture frame from camera {i}. Exiting...")
                break

            cv2.imshow(f"camera {i} feed", frame)
        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:  # Release all camera resources
        cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows