import sys
from pose_detector import (
    AnkleAssessment, 
    HipAssessment, 
    SpineAssessment, 
    HipFlexionAssessment, 
    FingerFloorAssessment, 
    SpineRotationAssessment, 
    AnkleInversionAssessment,
    PassiveHipFlexionAssessment
)

def main():
    while True:
        print("==========================================================")
        print("    BIOMECHANICAL CLINICAL ASSESSMENT LIBRARY (Neutral-0) ")
        print("==========================================================")
        print("Please select the assessment module you want to execute:")
        print(" 1. Ankle Range of Motion")
        print(" 2. Hip Abduction & Adduction")
        print(" 3. Spine Lateral Flexion")
        print(" 4. Active Hip Flexion Sagittal")
        print(" 5. Passive Hip Flexion Sagittal (Hands Assisted - Abb. 1b)")
        print(" 6. Finger-to-Floor Test")
        print(" 7. Spine Rotation Assessment")
        print(" 8. Ankle Inversion Assessment")
        print(" 9. Exit Library")
        print("----------------------------------------------------------")
        
        choice = input("Enter your choice (1-9): ").strip()

        if choice == "1":
            assessment = AnkleAssessment()
            left_video = "data/videos/UmfangmaßeSenkenLeft01.mp4"
            right_video = "data/videos/UmfangmaßeSenkenRight01.mp4"
            
            left_results = assessment.analyze_video(video_path=left_video, side="left")
            right_results = assessment.analyze_video(video_path=right_video, side="right")
            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "2":
            assessment = HipAssessment()
            left_video = "data/videos/HüftgelenkeAnfuhrenLedft01.mp4"
            right_video = "data/videos/HüftgelenkeAnfuhrenRight01.mp4"
            
            left_results = assessment.analyze_video(video_path=left_video, side="left")
            right_results = assessment.analyze_video(video_path=right_video, side="right")
            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "3":
            assessment = SpineAssessment()
            left_video = "data/videos/BWS-LWSLeft01.mp4"
            right_video = "data/videos/BWS-LWSRight01.mp4"
            
            left_results = assessment.analyze_video(video_path=left_video, side="left")
            right_results = assessment.analyze_video(video_path=right_video, side="right")
            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "4":
            assessment = HipFlexionAssessment()
            left_video = "data/videos/HüftgelenkeBeuggLeft01.mp4"
            right_video = "data/videos/HüftgelenkeBeuggRight01.mp4"
            
            print("\n[INFO] Initializing Active Hip Flexion Engine...")
            print("[STEP 1/2] Processing Left Active Hip Flexion...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Active Hip Flexion...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "5":
            # ----------------- ماژول خمش غیرفعال ران با دست -----------------
            assessment = PassiveHipFlexionAssessment()
            left_video = "data/videos/HüftgelenkeStreckLeft01.mp4"
            right_video = "data/videos/HüftgelenkeStreckRight01.mp4"
            
            print("\n[INFO] Initializing Passive Hip Flexion Engine (Hands Assisted)...")
            print("[STEP 1/2] Processing Left Passive Hip Flexion...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Passive Hip Flexion...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "6":
            assessment = FingerFloorAssessment()
            video1 = "data/videos/Finger-Boden01.mp4"
            video2 = "data/videos/Finger-Boden03.mp4"
            
            print("\n[INFO] Initializing Finger-to-Floor Engine...")
            print("[STEP 1/2] Analyzing Trial 1 (Finger-Boden01)...")
            res1 = assessment.analyze_video(video_path=video1)

            print("\n[STEP 2/2] Analyzing Trial 2 (Finger-Boden03)...")
            res2 = assessment.analyze_video(video_path=video2)

            if res1 and res2:
                assessment.print_report(res1, res2)
            assessment.close()

        elif choice == "7":
            assessment = SpineRotationAssessment()
            left_video = "data/videos/SeitneigenLeft01.mp4"
            right_video = "data/videos/SeitneigenRight01.mp4"
            
            print("\n[INFO] Initializing Spine Axial Rotation Engine...")
            print("[STEP 1/2] Processing Left Spine Rotation...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Spine Rotation...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()
        
        elif choice == "8":
            assessment = AnkleInversionAssessment()
            left_video = "data/videos/BeinlängeAusenrandLeft01.mp4"
            right_video = "data/videos/BeinlängeAusenrandRight01.mp4"
            
            print("\n[INFO] Initializing Ankle Inversion Assessment Engine...")
            print("[STEP 1/2] Processing Left Ankle...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Ankle...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "9":
            print("\nExiting library. Thank you for using Neutral-0 Biomechanical Suite!")
            sys.exit()
            
        else:
            print("\n[ERROR] Invalid input. Please enter a valid option (1-9).\n")

if __name__ == "__main__":
    main()