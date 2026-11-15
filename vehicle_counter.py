from pathlib import Path
import csv
from collections import Counter, defaultdict

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "input_videos" / "traffic_25s.mp4"
OUTPUT_DIR = BASE_DIR / "output_videos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "crossing_count_traffic_25s.mp4"
OUTPUT_CSV = OUTPUT_DIR / "crossing_counts_25s.csv"
DETAILS_CSV = OUTPUT_DIR / "crossing_details_25s.csv"

VEHICLES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

LINE_POSITION_RATIO = 0.60
CONFIDENCE = 0.25
MIN_OBSERVATIONS = 5
CLASS_HISTORY_SIZE = 15


def main():
    if not VIDEO_PATH.exists():
        print(f"Input video not found: {VIDEO_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Error: Cannot create output video.")
        cap.release()
        return

    line_y = int(height * LINE_POSITION_RATIO)

    previous_centers = {}
    class_histories = defaultdict(list)
    crossed_ids = set()
    counted_ids = set()
    crossing_directions = {}
    count_details = []

    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0,
    }

    frame_number = 0

    print("Starting line-crossing vehicle counting...")
    print(f"Counting line: y = {line_y}")
    print("Press Q in the video window to stop early.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLES.keys()),
                conf=CONFIDENCE,
                device="cpu",
                verbose=False,
            )[0]

            annotated = result.plot()

            cv2.line(
                annotated,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                3,
            )

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(
                    boxes, track_ids, class_ids
                ):
                    if class_id not in VEHICLES:
                        continue

                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    history = class_histories[track_id]
                    history.append(class_id)

                    if len(history) > CLASS_HISTORY_SIZE:
                        history.pop(0)

                    previous_y = previous_centers.get(track_id)

                    if previous_y is not None:
                        # Detect a crossing in either direction.
                        if (
                            previous_y < line_y <= center_y
                            or previous_y > line_y >= center_y
                        ):
                            crossed_ids.add(track_id)

                            if previous_y < line_y <= center_y:
                                crossing_directions[track_id] = (
                                    "Top to bottom"
                                )
                            else:
                                crossing_directions[track_id] = (
                                    "Bottom to top"
                                )

                    previous_centers[track_id] = center_y

                    # Count only after the track crosses the line.
                    # Count each tracking ID no more than once.
                    if (
                        track_id in crossed_ids
                        and track_id not in counted_ids
                        and len(history) >= MIN_OBSERVATIONS
                    ):
                        stable_class_id = Counter(history).most_common(1)[0][0]
                        vehicle_name = VEHICLES[stable_class_id]

                        counts[vehicle_name] += 1
                        counted_ids.add(track_id)

                        count_details.append({
                            "Track ID": track_id,
                            "Vehicle Type": vehicle_name,
                            "Crossing Frame": frame_number,
                            "Direction": crossing_directions.get(
                                track_id, "Unknown"
                            ),
                            "Observations At Count": len(history),
                        })

                    cv2.circle(
                        annotated,
                        (center_x, center_y),
                        4,
                        (255, 255, 0),
                        -1,
                    )

            panel_lines = [
                f"Cars: {counts['Car']}",
                f"Motorcycles: {counts['Motorcycle']}",
                f"Buses: {counts['Bus']}",
                f"Trucks: {counts['Truck']}",
                f"Total: {sum(counts.values())}",
            ]

            for index, label in enumerate(panel_lines):
                cv2.putText(
                    annotated,
                    label,
                    (20, 40 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            writer.write(annotated)
            cv2.imshow("YOLO11 Line-Crossing Counter", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Vehicle Type", "Count"])

        for name, count in counts.items():
            csv_writer.writerow([name, count])

        csv_writer.writerow(["Total", sum(counts.values())])

    with open(DETAILS_CSV, "w", newline="") as f:
        columns = [
            "Track ID",
            "Vehicle Type",
            "Crossing Frame",
            "Direction",
            "Observations At Count",
        ]
        csv_writer = csv.DictWriter(f, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(count_details)

    print("\nLine-crossing count finished.")
    print("\nFinal counts:")

    for name, count in counts.items():
        print(f"{name}: {count}")

    print(f"Total: {sum(counts.values())}")
    print(f"\nOutput video: {OUTPUT_VIDEO}")
    print(f"Count summary: {OUTPUT_CSV}")
    print(f"Crossing details: {DETAILS_CSV}")
    print(
        "\nImportant: These are tracking-based estimates. "
        "ID switches and missed detections can still affect accuracy."
    )


if __name__ == "__main__":
    main()
