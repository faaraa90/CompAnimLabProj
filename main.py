import cv2
import os
import mediapipe as mp

# 1. Video file configuration
video_path = "data/videos/UmfangmaßeSenkenLeft01.mp4"

# Security check to ensure the video path is valid
if not os.path.exists(video_path):
    print(f"[ERROR] Target video file not found at: {video_path}")
    exit()

# 2. Initialize MediaPipe core structures
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Setup the Pose Estimation model config
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video stream via OpenCV
cap = cv2.VideoCapture(video_path)
print("[INFO] MediaPipe pipeline started successfully. Tracking video joints...")

# 3. Frame processing pipeline loop
while cap.isOpened():
    ret, frame = cap.read()
    
    # Break the loop safely at the end of the video file
    if not ret:
        print("[INFO] Video frame stream reached the end.")
        break

    # Convert image colors from BGR (OpenCV standard) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw the skeleton tracking landmarks onto the current frame
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Joint dots
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)                  # Bone lines
        )

    # Render frame in an interactive window
    cv2.imshow("MediaPipe Pose Tracking", frame)

    # Allow early interruption by pressing the 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("[INFO] Execution terminated by user.")
        break

# 4. Clean up operations and memory release
cap.release()
cv2.destroyAllWindows()
pose.close()