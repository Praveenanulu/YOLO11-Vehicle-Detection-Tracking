from pathlib import Path

import cv2
from ultralytics import YOLO

# Project directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"

# Create output directory if necessary
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO model
MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Vehicle classes in the COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck

# Video input
VIDEO_PATH = INPUT_DIR / "traffic.mp4"

# Video output
OUTPUT_PATH = OUTPUT_DIR / "tracked_traffic.mp4"


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        print("Place a video named traffic.mp4 inside input_videos/")
        return

    print("Loading YOLO11n model...")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Error: Could not open the input video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Error: Could not create the output video.")
        cap.release()
        return

    print("Starting vehicle detection and tracking...")
    print("Press Q in the video window to stop early.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=0.25,
                device="cpu",
                verbose=False,
            )

            annotated_frame = results[0].plot()

            writer.write(annotated_frame)

            cv2.imshow("YOLO11 Vehicle Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Tracking finished. Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
