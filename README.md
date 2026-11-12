# Real-Time Vision & Distance Assistant for Visually Impaired Users

An AI-powered real-time assistant built with Python, OpenCV, and YOLO that helps visually impaired users detect nearby objects, track their spatial position (left, center, right), estimate their distance in feet, and announce alerts asynchronously using text-to-speech.

## Features
- **Real-Time Object Detection**: Powered by Ultralytics YOLO (`yolo11n.pt` or custom weights).
- **Spatial Awareness**: Divides the camera viewport to categorize items as "on your left", "straight ahead", or "on your right".
- **Monocular Distance Estimation**: Calculates the approximate distance of detected items in feet based on bounding box pixel height and real-world dimensions.
- **Asynchronous Text-to-Speech**: Runs voice synthesis in a separate background thread so the video stream and detection loop never lag or freeze.
- **Priority Hazard Alerting**: Automatically prioritizes objects directly blocking the path ahead.


## Project Structure

```text
sdd-key-tool/
├── sdd_key.py          # Main CLI seed/key generator script
├── requirements.txt    # Python dependencies (if applicable)
├── .gitignore          # Files and directories for Git to ignore
└── README.md           # Project documentation and usage guide