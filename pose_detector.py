import cv2
import os
import math
import mediapipe as mp

class AnkleClinicalAnalyzer:
    def __init__(self, model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6):
        """
        Initializes the MediaPipe Pose engine tailored for Ankle joint analysis (Abb. 6).
        """
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Clinical thresholds from Neutral-0-Method (Messblatt untere Gliedmassen - Abb. 6)
        # In Neutral position, the angle between lower leg and foot is roughly 90 degrees.
        # Extension (Plantarflexion/Senken): Normal range is up to 40-50 degrees from neutral (90 + 40 = 130 deg)
        # Flexion (Dorsiflexion/Heben): Normal range is up to 20-30 degrees from neutral (90 - 20 = 70 deg)
        self.NORMAL_MAX_EXTENSION = 130.0  # Plantarflexion peak threshold
        self.NORMAL_MAX_FLEXION = 70.0     # Dorsiflexion peak threshold

    def calculate_angle(self, p1, p2, p3):
        """
        Calculates the 2D geometric angle at the vertex (p2: Ankle) between p1 (Knee) and p3 (Toes).
        """
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        angle_rad = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle_deg = math.degrees(angle_rad)
        angle_deg = abs(angle_deg)
        
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
            
        return angle_deg

    def analyze_ankle_video(self, video_path, window_name="Ankle Clinical Screening"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []

        print(f"[INFO] Initializing kinematic pipeline for: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Mapping landmarks for Ankle Movement (Abb. 6)
                knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                toes = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

                # Calculate live kinematic angle
                live_angle = self.calculate_angle(knee, ankle, toes)
                captured_angles.append(live_angle)

                # Visual feedback: Render skeleton on screen
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Print live angle overlay on the frame
                cv2.putText(
                    frame, f"Ankle Angle: {int(live_angle)} Deg", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA
                )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("[INFO] Processing interrupted by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if not captured_angles:
            return None

        # Extract diagnostic metrics (Peak extension and flexion)
        max_angle_achieved = max(captured_angles)
        min_angle_achieved = min(captured_angles)
        
        return {
            "max_extension": max_angle_achieved,
            "max_flexion": min_angle_achieved
        }

    def generate_clinical_report(self, file_name, results):
        """
        Acts as the Report Generator module. Evaluates metrics against clinical norms
        and prints a diagnostic screening summary.
        """
        if not results:
            print(f"\n[REPORT] Failed to generate data for {file_name}")
            return

        print("\n" + "="*55)
        print(f"      CLINICAL SCREENING REPORT (Neutral-0-Method)  ")
        print(f"      Target File: {file_name}")
        print("="*55)
        print(f"Captured Range of Motion (ROM):")
        print(f" - Maximum Plantarflexion (Senken): {int(results['max_extension'])} Degrees")
        print(f" - Maximum Dorsiflexion (Heben):     {int(results['max_flexion'])} Degrees")
        print("-"*55)
        print("Diagnostic Evaluation:")

        # Diagnostic screening logic for Plantarflexion (Senken)
        if results['max_extension'] >= self.NORMAL_MAX_EXTENSION - 5:
            print(" [PASS] Plantarflexion (Senken): Normal mobility detected.")
        else:
            deficit = int(self.NORMAL_MAX_EXTENSION - results['max_extension'])
            print(f" [FAIL] Plantarflexion (Senken): Restricted mobility! Deficit of {deficit} degrees.")

        # Diagnostic screening logic for Dorsiflexion (Heben)
        if results['max_flexion'] <= self.NORMAL_MAX_FLEXION + 5:
            print(" [PASS] Dorsiflexion (Heben): Normal mobility detected.")
        else:
            deficit = int(results['max_flexion'] - self.NORMAL_MAX_FLEXION)
            print(f" [FAIL] Dorsiflexion (Heben): Restricted mobility! Deficit of {deficit} degrees.")
        
        print("="*55 + "\n")

    def close(self):
        self.pose.close()