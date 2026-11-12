import queue
import threading
import time
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# --- CALIBRATION & CONSTANTS ---
# Estimated focal length of standard webcam in pixels (fine-tune as needed)
FOCAL_LENGTH = 600

# Average real-world heights of common objects in meters
AVERAGE_HEIGHTS = {
    "person": 1.70,  # ~5 feet 7 inches
    "chair": 0.90,  # ~3 feet
    "bottle": 0.20,  # ~8 inches
    "laptop": 0.25,  # ~10 inches
    "cell phone": 0.15,  # ~6 inches
    "backpack": 0.45,  # ~1.5 feet
    "tv": 0.60,
    "cup": 0.10,
}
DEFAULT_HEIGHT = 0.5  # Fallback height if object isn't in dictionary (meters)


class SpeechWorker:
  """Manages background text-to-speech to prevent freezing the camera feed."""

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
    # Flush older backlog so the user always hears immediate info
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
  """Calculates distance in feet using bounding box pixel height."""
  if pixel_height <= 0:
    return 0.0

  real_height = AVERAGE_HEIGHTS.get(class_name, DEFAULT_HEIGHT)
  distance_meters = (real_height * FOCAL_LENGTH) / pixel_height
  distance_feet = distance_meters * 3.28084
  return round(distance_feet, 1)


def main():
  print("Loading YOLO model...")
  model = YOLO("yolo11n.pt")

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Error: Could not open camera.")
    return

  speech = SpeechWorker()
  speech.speak("Vision assistant with distance and spatial tracking active.")

  print("\n--- Assistant Running ---")
  print("Press 'q' to quit.")
  print("Press 'r' to repeat the last announcement.")

  last_spoken_time = 0
  speech_interval = 3.5  # Time buffer between spoken alerts
  last_announcement = "Nothing detected yet."

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

            # Extract bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

            box_height_pixels = y2 - y1
            x_center = (x1 + x2) / 2

            # Calculate distance in feet
            distance_ft = estimate_distance(label, box_height_pixels)

            # Spatial orientation (Left, Center, Right)
            if x_center < width * 0.35:
              position = "on your left"
            elif x_center > width * 0.65:
              position = "on your right"
            else:
              position = "straight ahead"

            detected_objects.append((label, position, distance_ft))

            # Draw visual bounding box and info on feed
            display_text = f"{label} ~{distance_ft}ft ({position})"
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
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

      # Throttled speech decision making
      current_time = time.time()
      if detected_objects and (
          current_time - last_spoken_time > speech_interval
      ):
        # Sort objects by distance (closest first)
        detected_objects.sort(key=lambda x: x[2])

        # Prioritize hazards directly in front if within 8 feet
        front_hazards = [
            obj
            for obj in detected_objects
            if obj[1] == "straight ahead" and obj[2] < 8.0
        ]

        if front_hazards:
          target = front_hazards[0]  # Closest front hazard
        else:
          target = detected_objects[0]  # Closest overall object

        label, position, dist = target
        announcement = f"{label}, about {dist} feet {position}"

        speech.speak(announcement)
        last_announcement = announcement
        last_spoken_time = current_time

      cv2.imshow("Complete Assistive View", frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      elif key == ord("r"):
        speech.speak(f"Repeating: {last_announcement}")

  finally:
    cap.release()
    cv2.destroyAllWindows()
    speech.stop()
    print("Assistant shut down safely.")


if __name__ == "__main__":
  main()