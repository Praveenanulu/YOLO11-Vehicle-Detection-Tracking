# YOLO11 Vehicle Detection, Tracking and Counting

A computer vision project using YOLO11, OpenCV, and ByteTrack to detect, track, and count vehicles crossing a designated line in traffic video.

## Features

- YOLO11 vehicle detection
- Multi-object tracking with ByteTrack
- Detection of cars, motorcycles, buses, and trucks
- Bidirectional line-crossing counting
- Annotated output video
- CSV reports of vehicle counts and crossing details

## Technologies

- Python
- Ultralytics YOLO11
- OpenCV
- NumPy
- ByteTrack

## Project Structure

vehicle_tracking_project/
- vehicle_counter.py
- vehicle_tracker.py
- audit_tracking.py
- requirements.txt
- README.md
- .gitignore

## Installation

Clone this repository from GitHub, then enter the project directory.

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

Download the YOLO11 nano model:

    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

Place your traffic video at:

    input_videos/traffic_25s.mp4

## Run

    python vehicle_counter.py

Ensure the input video path in the script matches your video.

## Sample Results

Test video: 25 seconds, 1920 x 1080 resolution, 30 FPS.

| Vehicle Type | Recorded Crossings |
|---|---:|
| Car | 32 |
| Motorcycle | 11 |
| Bus | 1 |
| Truck | 8 |
| Total | 52 |

These are sample tracking-based estimates, not independently verified vehicle counts.

## Output Files

- crossing_count_traffic_25s.mp4 — annotated output video
- crossing_counts_25s.csv — count summary
- crossing_details_25s.csv — crossing details and tracking IDs

## Limitations

- Vehicles can be missed or misclassified.
- Tracking IDs may change for the same physical vehicle.
- Occlusion and tracking errors can affect the results.
- Counting accuracy has not been independently verified.

## Future Improvements

- Evaluate accuracy using manually verified data.
- Improve tracking consistency.
- Support real-time camera input.
- Add traffic analytics and visualizations.

## Author

Muhammed Shamshad V

Interests: Robotics, Artificial Intelligence, Computer Vision, and Autonomous Systems.
