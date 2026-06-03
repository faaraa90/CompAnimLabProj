import os
from pose_detector import AnkleClinicalAnalyzer

def main():
    # 1. Initialize the specialized clinical ankle module
    analyzer = AnkleClinicalAnalyzer(model_complexity=1)

    # 2. Target ankle video (Abb. 6 - Side Profile view)
    video_file = "data/videos/UmfangmaßeSenkenLeft01.mp4"
    file_name = os.path.basename(video_file)

    # 3. Process video and extract range of motion angles
    print("[INFO] Starting ankle evaluation loop...")
    results = analyzer.analyze_ankle_video(video_path=video_file, window_name="Ankle - Abb. 6 Analysis")

    # 4. Run clinical validation and generate the medical screening report
    analyzer.generate_clinical_report(file_name=file_name, results=results)

    # Clean up memory
    analyzer.close()

if __name__ == "__main__":
    main()