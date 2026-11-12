import queue
import threading
import time
import cv2
import numpy as np
import pyttsx3
from PIL import Image
from ultralytics import YOLO

# Optional imports for BLIP scene description (Press 'g')
try:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

# --- CALIBRATION & CONSTANTS ---
FOCAL_LENGTH = 600  # Estimated webcam focal length in pixels

AVERAGE_HEIGHTS = {
    "person": 1.70,
    "chair": 0.90,
    "bottle": 0.20,
    "laptop": 0.25,
    "cell phone": 0.15,
    "backpack": 0.45,
    "tv": 0.60,
    "cup": 0.10,
}
DEFAULT_HEIGHT = 0.5


class SpeechWorker:
    """Manages background text-to-speech so the video feed never freezes."""

    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)

        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")

    def speak(self, text):
        # Flush older backlog for immediate alert priority
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
    """Calculates approximate distance in feet."""
    if pixel_height <= 0:
        return 0.0
    real_height = AVERAGE_HEIGHTS.get(class_name, DEFAULT_HEIGHT)
    distance_meters = (real_height * FOCAL_LENGTH) / pixel_height
    distance_feet = distance_meters * 3.28084
    return round(distance_feet, 1)


def main():
    print("Loading YOLO model...")
    model = YOLO("yolo11n.pt")

    # Load BLIP scene captioning model if available
    blip_processor, blip_model = None, None
    if BLIP_AVAILABLE:
        print(
            "Loading BLIP scene captioning model (this may take a moment on first"
            " run)..."
        )
        try:
            blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            print("BLIP model loaded successfully.")
        except Exception as e:
            print(f"Could not load BLIP model: {e}")
    else:
        print(
            "Notice: Transformers/Torch not fully installed. Pressing 'g' for scene"
            " description will be disabled."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    speech = SpeechWorker()
    speech.speak("Great Vision assistant active. All systems online.")

    print("\n--- Great-Vision Assistant Running ---")
    print("Controls:")
    print("  [q] - Quit application")
    print("  [r] - Repeat last announcement")
    print("  [g] - Generate full scene description (BLIP AI)")

    last_spoken_time = 0
    speech_interval = 3.5  # Seconds between automated object alerts
    last_announcement = "System initialized."

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            height, width, _ = frame.shape
            results = model(frame, verbose=False)

            detected_objects = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        label = model.names[cls_id]

                        # Bounding box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                        box_height_pixels = y2 - y1
                        x_center = (x1 + x2) / 2

                        # Calculations
                        distance_ft = estimate_distance(
                            label, box_height_pixels)

                        if x_center < width * 0.35:
                            position = "on your left"
                        elif x_center > width * 0.65:
                            position = "on your right"
                        else:
                            position = "straight ahead"

                        detected_objects.append((label, position, distance_ft))

                        # Visual overlay
                        display_text = f"{label} ~{distance_ft}ft ({position})"
                        cv2.rectangle(
                            frame, (int(x1), int(y1)), (int(
                                x2), int(y2)), (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame,
                            display_text,
                            (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            # Throttled real-time speech logic
            current_time = time.time()
            if detected_objects and (
                current_time - last_spoken_time > speech_interval
            ):
                detected_objects.sort(key=lambda x: x[2])  # Sort by distance
                front_hazards = [
                    obj
                    for obj in detected_objects
                    if obj[1] == "straight ahead" and obj[2] < 8.0
                ]

                target = front_hazards[0] if front_hazards else detected_objects[0]
                label, position, dist = target

                announcement = f"{label}, about {dist} feet {position}"
                speech.speak(announcement)
                last_announcement = announcement
                last_spoken_time = current_time

            cv2.imshow("Great-Vision Ultimate Assistant", frame)

            # Keyboard Interactivity
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                speech.speak(f"Repeating: {last_announcement}")
            elif key == ord("g"):
                if blip_model and blip_processor:
                    speech.speak("Analyzing scene description...")
                    # Convert OpenCV BGR frame to RGB PIL Image
                    pil_img = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    inputs = blip_processor(
                        images=pil_img, return_tensors="pt")
                    output = blip_model.generate(**inputs, max_new_tokens=50)
                    caption = blip_processor.decode(
                        output[0], skip_special_tokens=True)

                    scene_text = f"Scene overview: {caption}"
                    print(f"\n[BLIP AI] {scene_text}")
                    speech.speak(scene_text)
                    last_announcement = scene_text
                else:
                    speech.speak("Scene captioning model is not available.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()
        print("Great-Vision assistant shut down safely.")


if __name__ == "__main__":
    main()
