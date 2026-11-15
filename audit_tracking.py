from pathlib import Path
import csv
from collections import Counter, defaultdict

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "input_videos" / "traffic_25s.mp4"
MODEL_PATH = BASE_DIR / "yolo11n.pt"
OUTPUT_CSV = BASE_DIR / "output_videos" / "track_audit_25s.csv"

VEHICLES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

if not VIDEO_PATH.exists():
    raise SystemExit(f"Video not found: {VIDEO_PATH}")

model = YOLO(str(MODEL_PATH))

tracks = defaultdict(lambda: {
    "first_frame": None,
    "last_frame": None,
    "observations": 0,
    "classes": Counter(),
})

frame_number = 0

print("Auditing tracking IDs. This may take a few minutes...")

results = model.track(
    source=str(VIDEO_PATH),
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    classes=list(VEHICLES.keys()),
    conf=0.25,
    device="cpu",
    verbose=False,
)

for result in results:
    frame_number += 1

    if result.boxes is None or result.boxes.id is None:
        continue

    ids = result.boxes.id.int().cpu().tolist()
    classes = result.boxes.cls.int().cpu().tolist()

    for track_id, class_id in zip(ids, classes):
        if class_id not in VEHICLES:
            continue

        track = tracks[track_id]

        if track["first_frame"] is None:
            track["first_frame"] = frame_number

        track["last_frame"] = frame_number
        track["observations"] += 1
        track["classes"][VEHICLES[class_id]] += 1

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Track ID",
        "First Frame",
        "Last Frame",
        "Observations",
        "Most Common Class",
        "Class Disagreement",
        "Class History",
    ])

    for track_id, track in sorted(tracks.items()):
        history = track["classes"]
        stable_class, stable_count = history.most_common(1)[0]
        disagreement = stable_count < track["observations"]

        writer.writerow([
            track_id,
            track["first_frame"],
            track["last_frame"],
            track["observations"],
            stable_class,
            "Yes" if disagreement else "No",
            "; ".join(f"{name}: {count}" for name, count in history.items()),
        ])

print(f"\nFrames processed: {frame_number}")
print(f"Unique tracking IDs observed: {len(tracks)}")
print("\nIDs by most common class:")

class_totals = Counter()
for track in tracks.values():
    if track["classes"]:
        class_totals[track["classes"].most_common(1)[0][0]] += 1

for name in VEHICLES.values():
    print(f"{name}: {class_totals[name]}")

print(f"\nAudit CSV saved to: {OUTPUT_CSV}")
print("Note: unique IDs are not guaranteed to equal unique physical vehicles.")
