import sys
from src.pose_detector import AnkleAssessment, HipAssessment, SpineAssessment, HipFlexionAssessment, FingerFloorAssessment, SpineRotationAssessment, AnkleInversionAssessment

def main():
    while True:
        print("==========================================================")
        print("    BIOMECHANICAL CLINICAL ASSESSMENT LIBRARY (Neutral-0) ")
        print("==========================================================")
        print("Please select the assessment module you want to execute:")
        print(" 1. Ankle Range of Motion")
        print(" 2. Hip Abduction & Adduction")
        print(" 3. Spine Lateral Flexion")
        print(" 4. Hip Flexion Sagittal")
        print(" 5. Finger-to-Floor Test")
        print(" 6. Spine Rotation Assessment")
        print(" 7. Ankle Inversion Assessment")
        print(" 8. Exit Library")
        print("----------------------------------------------------------")
        
        choice = input("Enter your choice (1-8): ").strip()

        if choice == "1":
            assessment = AnkleAssessment()
            left_video = "data/videos/UmfangmaßeSenkenLeft01.mp4"
            right_video = "data/videos/UmfangmaßeSenkenRight01.mp4"
            
            left_results = assessment.analyze_video(video_path=left_video, side="left")
            right_results = assessment.analyze_video(video_path=right_video, side="right")
            if left_results and right_results:
                left_results['right_rom'] = right_results['rom']
                left_results['right_repetitions'] = right_results['repetitions']
                assessment.print_symmetry_report(left_metrics=left_results, right_metrics=left_results)
            assessment.close()

        elif choice == "2":
            assessment = HipAssessment()
            left_video = "data/videos/HüftgelenkeAnfuhrenLeft01.mp4"
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
            # ----------------- ماژول جدید خم کردن ران -----------------
            assessment = HipFlexionAssessment()
            # نام فایل‌های ارسالی جدید خود را در این بخش جایگزین کنید
            left_video = "data/videos/HüftgelenkeBeuggLeft01.mp4"  # ویدیویی که پای چپ بالا می‌آید
            right_video = "data/videos/HüftgelenkeBeuggRight01.mp4" # ویدیویی که پای راست بالا می‌آید
            
            print("\n[INFO] Initializing Hip Flexion Engine...")
            print("[STEP 1/2] Processing Left Hip Flexion...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Hip Flexion...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()

        elif choice == "5":
            # ----------------- ماژول جدید تست انگشت به زمین -----------------
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

        
        elif choice == "6":
            # ----------------- ماژول جدید چرخش محوری ستون فقرات -----------------
            assessment = SpineRotationAssessment()
            left_video = "data/videos/SeitneigenLeft01.mp4"   # فیلم چرخش به چپ
            right_video = "data/videos/SeitneigenRight01.mp4" # فیلم چرخش به راست
            
            print("\n[INFO] Initializing Spine Axial Rotation Engine...")
            print("[STEP 1/2] Processing Left Spine Rotation...")
            left_results = assessment.analyze_video(video_path=left_video, side="left")

            print("\n[STEP 2/2] Processing Right Spine Rotation...")
            right_results = assessment.analyze_video(video_path=right_video, side="right")

            if left_results and right_results:
                assessment.print_report(left_results, right_results)
            assessment.close()
        
        elif choice == "7":
            # ----------------- ماژول جدید خم کردن مچ پا به لبه بیرونی -----------------
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

        elif choice == "8":
            print("\nExiting library. Thank you for using Neutral-0 Biomechanical Suite!")
            sys.exit()
            
        else:
            print("\n[ERROR] Invalid input. Please enter a valid option (1-8).\n")

if __name__ == "__main__":
    main()