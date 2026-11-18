import queue
import threading
import time
import cv2
import numpy as np
import pyttsx3
from PIL import Image
from ultralytics import YOLO

try:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

FOCAL_LENGTH = 600

AVERAGE_HEIGHTS = {
    # Default fallback heights for general COCO classes + your custom classes
    "person": 1.70, "bicycle": 1.00, "car": 1.50, "motorcycle": 1.00,
    "bus": 3.00, "truck": 2.50, "chair": 0.90, "couch": 0.90,
    "bed": 0.60, "table": 0.75, "tv": 0.60, "laptop": 0.25,
    "cell phone": 0.15, "bottle": 0.20, "cup": 0.10, "book": 0.20,
    "backpack": 0.45, "umbrella": 0.90, "handbag": 0.30,
    "drone": 0.30, "phone": 0.15, "ruler": 0.05
}
DEFAULT_HEIGHT = 0.5


class SpeechWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    break
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")

    def speak(self, text):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()


def estimate_distance(class_name, pixel_height):
    if pixel_height <= 0:
        return 0.0
    real_height = AVERAGE_HEIGHTS.get(class_name, DEFAULT_HEIGHT)
    distance_meters = (real_height * FOCAL_LENGTH) / pixel_height
    distance_feet = distance_meters * 3.28084
    return round(distance_feet, 1)


def main():
    print("Loading models...")
    # 1. Load your custom specialized model
    try:
        custom_model = YOLO("best.pt")
    except Exception as e:
        print(f"Could not load best.pt: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    speech = SpeechWorker()
    speech.speak("Great Vision universal tracker active.")

    last_spoken_time = 0
    speech_interval = 4.0
    last_announcement = "System initialized."

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            # Run inference on BOTH models
            custom_results = custom_model(frame, conf=0.4, verbose=False)
            general_results = general_model(frame, conf=0.4, verbose=False)

            detected_objects = []

            # Process custom model results
            for r in custom_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = custom_model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                    dist = estimate_distance(label, y2 - y1)
                    x_center = (x1 + x2) / 2
                    pos = "on your left" if x_center < width * \
                        0.35 else ("on your right" if x_center > width * 0.65 else "straight ahead")

                    detected_objects.append((label, pos, dist))
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {dist}ft", (int(x1), max(
                        int(y1)-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Process general model results (skipping duplicates if already caught)
            for r in general_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = general_model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                    dist = estimate_distance(label, y2 - y1)
                    x_center = (x1 + x2) / 2
                    pos = "on your left" if x_center < width * \
                        0.35 else ("on your right" if x_center > width * 0.65 else "straight ahead")

                    detected_objects.append((label, pos, dist))
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {dist}ft", (int(x1), max(
                        int(y1)-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            # Announce objects
            current_time = time.time()
            if detected_objects and (current_time - last_spoken_time > speech_interval):
                detected_objects.sort(key=lambda x: x[2])
                phrases = [f"{lbl}, {d} feet {p}" for lbl,
                           p, d in detected_objects[:3]]
                announcement = ". ".join(phrases)
                speech.speak(announcement)
                last_announcement = announcement
                last_spoken_time = current_time

            cv2.imshow("Great-Vision Universal", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()


if __name__ == "__main__":
    main()
