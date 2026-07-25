import cv2
import os
import math
import mediapipe as mp
import numpy as np

import cv2
import os
import math
import mediapipe as mp
import numpy as np

class BiomechanicalAssessment:
    """
    Base Class (Parent) that handles core computer vision pipelines.
    Prevents code duplication by managing video streams and MediaPipe initialization.
    """
    def __init__(self, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def calculate_2d_angle(self, p1, p2, p3):
        """Standard vector math helper to calculate joint angles using atan2."""
        angle_rad = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
        angle_deg = abs(math.degrees(angle_rad))
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        return angle_deg

    def print_symmetry_report(self, left_metrics, right_metrics, metric_key, title="BILATERAL COMPARISON REPORT", unit="°"):
        """
        Generic helper method in base class to perform side-by-side comparison 
        and calculate symmetry index for any biomechanical assessment.
        """
        left_val = left_metrics.get(metric_key, 0.0)
        right_val = right_metrics.get(metric_key, 0.0)
        
        diff = abs(left_val - right_val)
        max_val = max(left_val, right_val)
        symmetry_index = (1.0 - (diff / max_val)) * 100.0 if max_val > 0 else 100.0

        print("\n" + "="*60)
        print(f"         {title.upper()}         ")
        print("="*60)
        print(f"LEFT SIDE:  {left_val:.1f}{unit} | Completed Reps: {left_metrics.get('repetitions', 0)}")
        print(f"RIGHT SIDE: {right_val:.1f}{unit} | Completed Reps: {right_metrics.get('repetitions', 0)}")
        print("-"*60)
        print(f"ABSOLUTE DIFFERENCE: {diff:.1f}{unit}")
        print(f"BILATERAL SYMMETRY INDEX: {symmetry_index:.1f}%")
        print("-"*60)
        print("CLINICAL EVALUATION:")
        if symmetry_index >= 85.0:
            print(" [PASS] Excellent symmetry between left and right sides.")
        else:
            print(" [FLAG] WARNING: Significant asymmetry detected between limbs/sides!")
        print("="*60 + "\n")

    def close(self):
        self.pose.close()

class AnkleAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Upper Ankle Joint Assessment (Abb. 6).
    Supports dual-direction tracking for both Dorsiflexion (Up) and Plantarflexion (Down).
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.NEUTRAL_ANGLE = 90.0
        self.CLINICAL_NORM_PLANTAR = 40.0  # Normal extension down
        self.CLINICAL_NORM_DORSI = 20.0    # Normal flexion up

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        """Draws a translucent HUD box in the Top Right corner with relative positioning."""
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"ANKLE COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Peak ROM:   {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        
        rep_count = 0
        direction = 0  # 0 = Downwards (Plantar), 1 = Upwards (Dorsi)
        prev_angle = None

        print(f"[INFO] Running Ankle Pipeline for {side.upper()} side...")

        window_name = f"Assessment Protocol - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                if side.lower() == "left":
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    toes = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                else:
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                    toes = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                live_angle = self.calculate_2d_angle(knee, ankle, toes)
                captured_angles.append(live_angle)
                current_max_rom = max(captured_angles) - min(captured_angles)

                # --- Dynamic Direction Detection & Adaptive Repetition Counter ---
                current_motion = "IDLE"
                if prev_angle is not None:
                    angle_diff = live_angle - prev_angle
                    
                    if angle_diff > 0.3:
                        current_motion = "DOWN"  # Plantarflexion (angle increasing)
                    elif angle_diff < -0.3:
                        current_motion = "UP"    # Dorsiflexion (angle decreasing)

                    # State machine for repetition counting
                    if live_angle > prev_angle + 0.3 and direction == 0:
                        if live_angle > 115:
                            direction = 1
                    elif live_angle < prev_angle - 0.3 and direction == 1:
                        if live_angle < 100:
                            rep_count += 1
                            direction = 0
                            
                prev_angle = live_angle

                # --- Biomechanical Deflections ---
                measured_plantar = max(0.0, live_angle - self.NEUTRAL_ANGLE)
                measured_dorsi = max(0.0, self.NEUTRAL_ANGLE - live_angle)

                # --- Dynamic Patient Coaching Feedbacks ---
                feedback_msg = "Keep moving your foot up and down."
                feedback_color = (255, 255, 255)

                if current_motion == "DOWN":
                    # Moving foot downwards (Plantarflexion)
                    if 5.0 < measured_plantar < self.CLINICAL_NORM_PLANTAR:
                        feedback_msg = "Pushing DOWN! Push a little further."
                        feedback_color = (0, 165, 255)
                    elif measured_plantar >= self.CLINICAL_NORM_PLANTAR:
                        feedback_msg = "Excellent Plantarflexion range reached! 🌟"
                        feedback_color = (0, 255, 0)

                elif current_motion == "UP":
                    # Pulling foot upwards towards the shin (Dorsiflexion)
                    if 5.0 < measured_dorsi < self.CLINICAL_NORM_DORSI:
                        feedback_msg = "Pulling UP! Pull closer towards your shin."
                        feedback_color = (0, 165, 255)
                    elif measured_dorsi >= self.CLINICAL_NORM_DORSI:
                        feedback_msg = "Excellent Dorsiflexion range achieved! 🌟"
                        feedback_color = (0, 255, 0)

                if rep_count >= 3:
                    feedback_msg = "🌟 STREAK ACTIVE! Patient level: PRO! 🌟"
                    feedback_color = (255, 51, 153)

                h, w, _ = frame.shape
                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                ankle_pixel = (int(ankle.x * w), int(ankle.y * h))
                cv2.circle(frame, ankle_pixel, int(w*0.006)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, current_max_rom, rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_angle = max(captured_angles)
        min_angle = min(captured_angles)
        
        return {
            "max_angle": max_angle,
            "min_angle": min_angle,
            "rom": max_angle - min_angle,
            "measured_plantar": max(0.0, max_angle - self.NEUTRAL_ANGLE),
            "measured_dorsi": max(0.0, self.NEUTRAL_ANGLE - min_angle),
            "repetitions": rep_count
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="rom", 
            title="ANKLE RANGE OF MOTION SYMMETRY REPORT"
        )


class HipAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Hip Abduction Assessment (Abb. 2).
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.NEUTRAL_ANGLE = 180.0
        self.CLINICAL_NORM_ABDUCTION = 40.0 

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"HIP COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Hip Angle: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Abduction: {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        
        rep_count = 0
        direction = 0  
        prev_angle = None

        print(f"[INFO] Running Hip Pipeline for {side.upper()} side...")

        # ایجاد نام پنجره یکتا برای مدیریت اوپن‌سی‌وی
        window_name = f"Assessment Protocol - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if Antiquities := results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                if side.lower() == "left":
                    shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                else:
                    shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

                live_angle = self.calculate_2d_angle(shoulder, hip, knee)
                captured_angles.append(live_angle)
                current_abduction = abs(self.NEUTRAL_ANGLE - live_angle)

                # --- Wave-Based Adaptive Repetition Counter for Hip ---
                if prev_angle is not None:
                    if live_angle < prev_angle - 0.3 and direction == 0:
                        if current_abduction > 15:
                            direction = 1
                    elif live_angle > prev_angle + 0.3 and direction == 1:
                        if current_abduction < 8:
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # --- Live Patient Coaching Feedbacks & Easter Egg ---
                feedback_msg = "Lift your leg out to the side slowly."
                feedback_color = (255, 255, 255)

                if 10.0 < current_abduction < self.CLINICAL_NORM_ABDUCTION:
                    feedback_msg = "Good! Keep lifting higher if you can."
                    feedback_color = (0, 165, 255)
                elif current_abduction >= self.CLINICAL_NORM_ABDUCTION:
                    feedback_msg = "Target reached! Excellent Hip Mobility! 🌟"
                    feedback_color = (0, 255, 0)

                if rep_count == 3:
                    feedback_msg = "🌟 HIP FREQUENCY STREAK ACTIVE! Great stamina! 🌟"
                    feedback_color = (255, 51, 153)

                h, w, _ = frame.shape
                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                hip_pixel = (int(hip.x * w), int(hip.y * h))
                cv2.circle(frame, hip_pixel, int(w*0.006)+1, (255, 0, 0), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max([abs(self.NEUTRAL_ANGLE - a) for a in captured_angles]), rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name) # بستن امن همان پنجره خاص بعد از پایان ویدیو

        if not captured_angles:
            return None

        all_deflections = [abs(self.NEUTRAL_ANGLE - a) for a in captured_angles]
        max_abduction = max(all_deflections)

        return {
            "max_abduction": max_abduction,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_abduction >= self.CLINICAL_NORM_ABDUCTION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_abduction", 
            title="HIP ABDUCTION SYMMETRY REPORT"
        )
class SpineAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Spine Lateral Flexion Assessment.
    Measures the spine deviation from the vertical axis using mid-shoulder and mid-hip.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.CLINICAL_NORM_FLEXION = 30.0 # Normal lateral bending degrees from vertical

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"SPINE COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Bending: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Bending:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        rep_count = 0
        direction = 0  # 0 = Bending outwards, 1 = Returning to center
        prev_angle = None

        print(f"[INFO] Running Spine Pipeline for {side.upper()} flexion...")

        window_name = f"Spine Assessment Suite - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract shoulders and hips
                l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

                # Calculate Midpoints to get a stable spine line
                mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
                mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                mid_hip_x = (l_hip.x + r_hip.x) / 2
                mid_hip_y = (l_hip.y + r_hip.y) / 2

                # Angle calculation relative to the true vertical axis
                spine_vector_x = mid_shoulder_x - mid_hip_x
                spine_vector_y = mid_shoulder_y - mid_hip_y
                
                # Angle with vertical line (0, -1)
                live_angle = math.degrees(math.atan2(abs(spine_vector_x), -spine_vector_y))
                captured_angles.append(live_angle)

                # --- Wave-Based Repetition Counter for Spine ---
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.2 and direction == 0:
                        if live_angle > 12: # Minimal bend detection
                            direction = 1
                    elif live_angle < prev_angle - 0.2 and direction == 1:
                        if live_angle < 5:  # Near neutral vertical line
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # --- Patient Coaching Feedback ---
                feedback_msg = f"Bend your torso slowly to your {side.upper()} side."
                feedback_color = (255, 255, 255)

                if 5.0 < live_angle < self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Good flex! Keep leaning without twisting."
                    feedback_color = (0, 165, 255)
                elif live_angle >= self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Excellent lateral range of motion! 🌟"
                    feedback_color = (0, 255, 0)

                h, w, _ = frame.shape
                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # Draw custom biomechanical spine line on frame
                ms_pixel = (int(mid_shoulder_x * w), int(mid_shoulder_y * h))
                mh_pixel = (int(mid_hip_x * w), int(mid_hip_y * h))
                cv2.line(frame, mh_pixel, ms_pixel, (0, 255, 255), int(w * 0.003) + 1)
                cv2.circle(frame, ms_pixel, int(w*0.006)+1, (0, 0, 255), -1)
                cv2.circle(frame, mh_pixel, int(w*0.006)+1, (255, 0, 0), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max(captured_angles), rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_flexion = max(captured_angles)

        return {
            "max_flexion": max_flexion,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_flexion", 
            title="SPINE LATERAL FLEXION SYMMETRY REPORT"
        )
class HipFlexionAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Hip Flexion Assessment (Sagittal Plane).
    Measures the hinge angle between Shoulder, Hip, and Knee.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.CLINICAL_NORM_FLEXION = 120.0 # Normal hip flexion degrees with bent knee

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"HIP FLEXION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle:   {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Flexion:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        rep_count = 0
        direction = 0  # 0 = Lifting leg, 1 = Lowering leg
        prev_angle = None

        print(f"[INFO] Running Hip Flexion Pipeline for {side.upper()} side...")

        window_name = f"Hip Flexion Assessment - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Select landmarks dynamically based on the side being tracked
                if side.lower() == "left":
                    shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                else:
                    shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

                # Convert to pixel coordinates for visualization
                h, w, _ = frame.shape
                s_pt = (int(shoulder.x * w), int(shoulder.y * h))
                h_pt = (int(hip.x * w), int(hip.y * h))
                k_pt = (int(knee.x * w), int(knee.y * h))

                # Calculate the inner hinge angle of the Hip
                angle_s_h_k = self.calculate_2d_angle(shoulder, hip, knee)
                
                # Biomechanical ROM: standard anatomical standing position is 0 degrees.
                # Flexion angle is how far the thigh moves away from the trunk line.
                live_angle = max(0.0, 180.0 - angle_s_h_k)
                captured_angles.append(live_angle)

                # --- Repetition Counter Logic ---
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.3 and direction == 0:
                        if live_angle > 30: # Threshold to start registering flexion
                            direction = 1
                    elif live_angle < prev_angle - 0.3 and direction == 1:
                        if live_angle < 15: # Returned near initial position
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # --- Clinical Feedback Messages ---
                feedback_msg = "Lift your knee up towards your chest."
                feedback_color = (255, 255, 255)

                if 45.0 < live_angle < self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Good execution, push for maximum height!"
                    feedback_color = (0, 165, 255)
                elif live_angle >= self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Target reached! Excellent Hip Mobility. 🌟"
                    feedback_color = (0, 255, 0)

                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # Visualizing the kinematic chain (Shoulder-Hip-Knee)
                cv2.line(frame, s_pt, h_pt, (0, 255, 0), int(w * 0.003) + 1)
                cv2.line(frame, h_pt, k_pt, (0, 255, 255), int(w * 0.003) + 1)
                for pt in [s_pt, h_pt, k_pt]:
                    cv2.circle(frame, pt, int(w*0.005)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max(captured_angles), rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_flexion = max(captured_angles)
        return {
            "max_flexion": max_flexion,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_flexion", 
            title="ACTIVE HIP FLEXION SYMMETRY REPORT"
        )    
class FingerFloorAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Finger-to-Floor (FFD) Assessment (Sagittal Plane).
    Measures hip flexion and monitors how close the hands get to the ankles/floor.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)

    def draw_dashboard(self, frame, live_angle, max_flexion, touch_detected, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        # ارتفاع کادر را کمی بیشتر می‌کنیم تا سطر جدید جا شود
        box_height = int(h * 0.18) if h > 1000 else 180 
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        status_str = "TOUCHED" if touch_detected else "DESCENDING..."
        status_color = (0, 255, 0) if touch_detected else (0, 255, 255)

        cv2.putText(frame, "FINGER-FLOOR TEST", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Hip Angle: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Floor Touch:   {status_str}", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness, cv2.LINE_AA)
        # اضافه شدن سطر شمارنده به کادر بالا
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    def analyze_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        floor_touched = False

        # متغیرهای جدید برای شمارش تکرار حرکت خم شدن
        rep_count = 0
        direction = 0  # 0 = در حال خم شدن به پایین، 1 = در حال برگشت به بالا
        prev_angle = None
        
        print(f"[INFO] Running Finger-to-Floor Pipeline...")
        window_name = "Finger-to-Floor Biomechanical Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # به دلیل نمای جانبی، سمتی که به دوربین نزدیک‌تر است یا میانگین نقاط چپ و راست را بررسی می‌کنیم.
                # در اینجا از سمت چپ به عنوان معیار پیش‌فرض استفاده می‌کنیم.
                shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

                h, w, _ = frame.shape
                s_pt = (int(shoulder.x * w), int(shoulder.y * h))
                h_pt = (int(hip.x * w), int(hip.y * h))
                a_pt = (int(ankle.x * w), int(ankle.y * h))
                w_pt = (int(wrist.x * w), int(wrist.y * h))

                # محاسبه زاویه تاشدگی ران/کمر
                hip_joint_angle = self.calculate_2d_angle(shoulder, hip, ankle)
                # تبدیل به زاویه بیومکانیکال (خم شدن نسبت به خط صاف بدن)
                live_angle = max(0.0, 180.0 - hip_joint_angle)
                captured_angles.append(live_angle)

                # منطق تشخیص لمس زمین: اگر مچ دست هم‌تراز یا پایین‌تر از مچ پا در راستای Y قرار گیرد
                if wrist.y >= (ankle.y - 0.05): # حاشیه خطای کوچک بالینی
                    floor_touched = True

                # --- الگوریتم موجی شمارش تکرار خم شدن از کمر ---
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.3 and direction == 0:
                        if live_angle > 40:  # حد آستانه شروع خم شدن عمیق
                            direction = 1
                    elif live_angle < prev_angle - 0.3 and direction == 1:
                        if live_angle < 15:  # برگشت به حالت تقریبا ایستاده
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle
                
                # --- پیام فیدبک گرافیکی پایین صفحه ---
                if floor_touched:
                    feedback_msg = "Excellent Mobility! Floor reached successfully."
                    feedback_color = (0, 255, 0)
                else:
                    feedback_msg = "Bend forward slowly keeping your knees straight."
                    feedback_color = (255, 255, 255)

                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # رسم خطوط حرکتی (ستون فقرات، ران و دست)
                cv2.line(frame, s_pt, h_pt, (0, 255, 0), int(w * 0.003) + 1)
                cv2.line(frame, h_pt, a_pt, (0, 255, 255), int(w * 0.003) + 1)
                cv2.line(frame, s_pt, w_pt, (255, 0, 255), int(w * 0.002) + 1) # خط دست

                for pt in [s_pt, h_pt, a_pt, w_pt]:
                    cv2.circle(frame, pt, int(w*0.005)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, live_angle, max(captured_angles), floor_touched, rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        return {
            "max_hip_flexion": max(captured_angles),
            "floor_reached": floor_touched,
            "repetitions": rep_count
        }

    def print_report(self, res_vid1, res_vid2):
        print("\n" + "="*60)
        print("          FINGER-TO-FLOOR (FFD) CLINICAL REPORT        ")
        print("="*60)
        print(f"Video 1 - Max Hip Flexion: {res_vid1['max_hip_flexion']:.1f}° | Touched Floor: {'YES' if res_vid1['floor_reached'] else 'NO'}")
        print(f"Video 2 - Max Hip Flexion: {res_vid2['max_hip_flexion']:.1f}° | Touched Floor: {'YES' if res_vid2['floor_reached'] else 'NO'}")
        print("-"*60)
        if res_vid1['floor_reached'] or res_vid2['floor_reached']:
            print("Clinical Assessment: NORMAL posterior chain flexibility.")
        else:
            print("Clinical Assessment: SHORTENED (Hamstring/Spine rigidity suspected).")
        print("="*60 + "\n")
class SpineRotationAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Spine Axial Rotation Assessment (Transverse Plane from Frontal View).
    Measures rotation angle based on shoulder width foreshortening and tracks hip stability.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.initial_shoulder_w = None
        self.initial_hip_w = None
        self.CLINICAL_NORM_ROTATION = 45.0 # Normal spine rotation is around 40-45 degrees

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps, hip_error):
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.16) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 51, 153), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"SPINE ROTATION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 51, 153), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Rotation: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Rotation:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        rep_count = 0
        direction = 0  # 0 = Rotating in, 1 = Returning to center
        prev_angle = None

        print(f"[INFO] Running Spine Axial Rotation Pipeline for {side.upper()} side...")
        window_name = f"Spine Rotation Suite - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

                # محاسبه فواصل افقی فعلی شانه‌ها و لگن
                current_shoulder_w = abs(r_shoulder.x - l_shoulder.x)
                current_hip_w = abs(r_hip.x - l_hip.x)

                # ذخیره ابعاد اولیه (فریم‌های ابتدایی که فرد صاف ایستاده است)
                if self.initial_shoulder_w is None or self.initial_shoulder_w == 0:
                    self.initial_shoulder_w = current_shoulder_w
                if self.initial_hip_w is None or self.initial_hip_w == 0:
                    self.initial_hip_w = current_hip_w

                # محاسبه زاویه چرخش ستون فقرات با فرمول آرک‌کسینوس نسبت به ابعاد اولیه
                ratio_s = np.clip(current_shoulder_w / self.initial_shoulder_w, 0.0, 1.0)
                live_angle = math.degrees(math.acos(ratio_s))
                captured_angles.append(live_angle)

                # محاسبه میزان خطای ناخواسته چرخش لگن
                ratio_h = np.clip(current_hip_w / self.initial_hip_w, 0.0, 1.0)
                hip_rotation_error = math.degrees(math.acos(ratio_h))

                # --- مدیریت تکرارها ---
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.3 and direction == 0:
                        if live_angle > 15: # شروع چرخش تایید شده
                            direction = 1
                    elif live_angle < prev_angle - 0.3 and direction == 1:
                        if live_angle < 7:  # برگشت به نقطه صفر مرکزی
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # --- پیام‌های فیدبک بالینی و کنترلی ---
                feedback_msg = f"Rotate your upper body towards the {side.upper()} side."
                feedback_color = (255, 255, 255)

                if hip_rotation_error > 12.0: # خطای چرخش ناخواسته لگن
                    feedback_msg = "WARNING: Keep your Hips facing FORWARD! Lock your pelvis."
                    feedback_color = (0, 0, 255)
                elif live_angle >= self.CLINICAL_NORM_ROTATION:
                    feedback_msg = "Excellent spinal rotation angle achieved! 🌟"
                    feedback_color = (0, 255, 0)

                h, w, _ = frame.shape
                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # گرافیک خط شانه برای نشان دادن تغییرات طول
                ls_p = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                rs_p = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                cv2.line(frame, ls_p, rs_p, (255, 0, 255), int(w * 0.003) + 1)
                cv2.circle(frame, ls_p, int(w*0.005)+1, (0, 0, 255), -1)
                cv2.circle(frame, rs_p, int(w*0.005)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max(captured_angles), rep_count, hip_rotation_error)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_rot = max(captured_angles)
        return {
            "max_rotation": max_rot,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_rot >= self.CLINICAL_NORM_ROTATION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_rotation", 
            title="SPINE ROTATION SYMMETRY REPORT"
        )

class AnkleInversionAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Ankle Inversion/Eversion Assessment (Frontal View).
    Measures the lateral deviation angle of the ankle joint using Knee, Ankle, and Foot Index.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.CLINICAL_NORM_INVERSION = 20.0 # Normal ankle inversion is around 20-30 degrees
        self.angle_history = [] # For smoothing jittery foot landmarks

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.16) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"ANKLE INVERSION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle:   {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Inversion:{int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        rep_count = 0
        direction = 0  # 0 = Inverting outward, 1 = Returning to neutral
        prev_angle = None
        self.angle_history = []

        print(f"[INFO] Running Ankle Inversion Pipeline for {side.upper()} side...")
        window_name = f"Ankle Inversion Suite - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # انتخاب پویا مفاصل بر اساس پای چپ یا راست
                if side.lower() == "left":
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    foot = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                else:
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                    foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                # محاسبه زاویه لولای مچ پا در کادر دو بعدی روبرو
                raw_joint_angle = self.calculate_2d_angle(knee, ankle, foot)
                
                # کالیبره کردن زاویه بیومکانیکی (انحراف از خط ۹۰ درجه عمود)
                raw_live_angle = abs(90.0 - raw_joint_angle)
                
                # فیلتر میانگین متحرک برای حذف پرش و لرزش مارکرهای پا
                self.angle_history.append(raw_live_angle)
                if len(self.angle_history) > 5:
                    self.angle_history.pop(0)
                live_angle = sum(self.angle_history) / len(self.angle_history)
                captured_angles.append(live_angle)

                # --- شمارش تکرار حرکت مچ پا ---
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.2 and direction == 0:
                        if live_angle > 8: # آستانه شروع انحراف مچ پا
                            direction = 1
                    elif live_angle < prev_angle - 0.2 and direction == 1:
                        if live_angle < 4: # بازگشت به حالت صاف
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # --- فیدبک بالینی پویا ---
                feedback_msg = "Roll your foot onto its outer edge slowly."
                feedback_color = (255, 255, 255)

                if live_angle >= self.CLINICAL_NORM_INVERSION:
                    feedback_msg = "Target Inversion Reached! Good flexibility."
                    feedback_color = (0, 255, 0)

                h, w, _ = frame.shape
                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # رسم زنجیره حرکتی ساق پا و کارهای مچ پا
                k_pt = (int(knee.x * w), int(knee.y * h))
                a_pt = (int(ankle.x * w), int(ankle.y * h))
                f_pt = (int(foot.x * w), int(foot.y * h))
                
                cv2.line(frame, k_pt, a_pt, (0, 255, 0), int(w * 0.003) + 1)
                cv2.line(frame, a_pt, f_pt, (255, 100, 0), int(w * 0.003) + 1)
                for pt in [k_pt, a_pt, f_pt]:
                    cv2.circle(frame, pt, int(w*0.005)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max(captured_angles), rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_inversion = max(captured_angles)
        return {
            "max_inversion": max_inversion,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_inversion >= self.CLINICAL_NORM_INVERSION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_inversion", 
            title="ANKLE INVERSION SYMMETRY REPORT"
        )
    
class PassiveHipFlexionAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Assisted/Passive Hip Flexion Assessment (Abb. 1b).
    Measures maximum hip flexion ROM when hands assist the movement up to ~130 Deg.
    Includes a moving average filter to handle potential landmark jitter from hand occlusion.
    """
    def __init__(self):
        super().__init__(model_complexity=1, detection_confidence=0.6, tracking_confidence=0.6)
        self.CLINICAL_NORM_FLEXION = 130.0  # نورم بالینی برای خمش با کمک دست (Abb. 1b)
        self.angle_history = []

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.16) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"PASSIVE HIP FLEX: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 165, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle:   {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Flexion:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def analyze_video(self, video_path, side="left"):
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        captured_angles = []
        rep_count = 0
        direction = 0  # 0 = Pulling leg with hands, 1 = Releasing leg
        prev_angle = None
        self.angle_history = []

        print(f"[INFO] Running Passive Hip Flexion Pipeline (Hands Assisted) for {side.upper()} side...")

        window_name = f"Passive Hip Flexion (Abb. 1b) - {side.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                if side.lower() == "left":
                    shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                else:
                    shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

                h, w, _ = frame.shape
                s_pt = (int(shoulder.x * w), int(shoulder.y * h))
                h_pt = (int(hip.x * w), int(hip.y * h))
                k_pt = (int(knee.x * w), int(knee.y * h))

                # محاسبه زاویه لولا
                angle_s_h_k = self.calculate_2d_angle(shoulder, hip, knee)
                raw_live_angle = max(0.0, 180.0 - angle_s_h_k)

                # اعمال فیلتر میانگین متحرک برای حذف نویز قرارگیری دست‌ها روی زانو
                self.angle_history.append(raw_live_angle)
                if len(self.angle_history) > 5:
                    self.angle_history.pop(0)
                live_angle = sum(self.angle_history) / len(self.angle_history)
                captured_angles.append(live_angle)

                # منطق شمارش تکرار
                if prev_angle is not None:
                    if live_angle > prev_angle + 0.3 and direction == 0:
                        if live_angle > 35:
                            direction = 1
                    elif live_angle < prev_angle - 0.3 and direction == 1:
                        if live_angle < 15:
                            rep_count += 1
                            direction = 0
                prev_angle = live_angle

                # پیام‌های راهنما
                feedback_msg = "Pull your knee toward your chest using your hands."
                feedback_color = (255, 255, 255)

                if 50.0 < live_angle < self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Good pull! Keep pulling smoothly toward maximum range."
                    feedback_color = (0, 165, 255)
                elif live_angle >= self.CLINICAL_NORM_FLEXION:
                    feedback_msg = "Maximum Passive Range Achieved (~130 Deg)! Excellent! 🌟"
                    feedback_color = (0, 255, 0)

                font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
                thickness = 2 if w < 1000 else int(w * 0.0015) + 1

                cv2.rectangle(frame, (20, h - int(h*0.08)), (w - 20, h - int(h*0.02)), (0, 0, 0), -1)
                cv2.putText(frame, feedback_msg, (35, h - int(h*0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, thickness, cv2.LINE_AA)

                # رسم زنجیره حرکتی
                cv2.line(frame, s_pt, h_pt, (0, 255, 0), int(w * 0.003) + 1)
                cv2.line(frame, h_pt, k_pt, (255, 165, 0), int(w * 0.003) + 1)
                for pt in [s_pt, h_pt, k_pt]:
                    cv2.circle(frame, pt, int(w*0.005)+1, (0, 0, 255), -1)

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.draw_dashboard(frame, side, live_angle, max(captured_angles), rep_count)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)

        if not captured_angles:
            return None

        max_flexion = max(captured_angles)
        return {
            "max_flexion": max_flexion,
            "repetitions": rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        self.print_symmetry_report(
            left_metrics=left_res, 
            right_metrics=right_res, 
            metric_key="max_flexion", 
            title="PASSIVE HIP FLEXION SYMMETRY REPORT"
        )