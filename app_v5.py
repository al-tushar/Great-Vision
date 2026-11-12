import subprocess
import sys
import time

import cv2
import easyocr
from ultralytics import YOLO

print("RUNNING VERSION 5 (background speech)", flush=True)

last_message = ""
speech_process = None


def stop_speaking():
    """Stop any speech that is currently playing."""
    global speech_process
    if speech_process is not None and speech_process.poll() is None:
        speech_process.terminate()
    speech_process = None


def speak(text):
    """Speak in the background. The camera keeps running."""
    global speech_process
    stop_speaking()
    print("Assistant:", text, flush=True)
    speech_process = subprocess.Popen(
        [sys.executable, "speak_worker.py", text])


def announce(text):
    """Speak a result AND remember it so H can repeat it."""
    global last_message
    last_message = text
    speak(text)


def position_of(x_center, image_width):
    if x_center < image_width / 3:
        return "on the left"
    elif x_center > 2 * image_width / 3:
        return "on the right"
    else:
        return "in the center"


print("Loading YOLO...", flush=True)
model = YOLO("yolo11n.pt")

print("Loading EasyOCR...", flush=True)
reader = easyocr.Reader(["en"], gpu=False)
print("Models ready!", flush=True)

camera = cv2.VideoCapture(0)
speak("Ready. S for objects, R to read text, H to repeat, X to stop talking, Q to quit.")

boxes_to_show = []
show_until = 0

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera error", flush=True)
        break

    preview = frame.copy()

    if time.time() < show_until:
        for x1, y1, x2, y2, label in boxes_to_show:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(preview, "S: Objects | R: Read | H: Repeat | X: Stop | Q: Quit",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow("Vision Assistant", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q")):
        break

    # --- X: stop talking ---
    if key in (ord("x"), ord("X")):
        stop_speaking()
        print("Speech stopped.", flush=True)

    # --- H: repeat last message ---
    if key in (ord("h"), ord("H")):
        if last_message:
            speak(last_message)
        else:
            speak("There is no message to repeat yet.")

    # --- S: objects ---
    if key in (ord("s"), ord("S")):
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        width = frame.shape[1]

        found = []
        boxes_to_show = []
        for box in results.boxes:
            name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            where = position_of((x1 + x2) / 2, width)
            found.append((area, name, where))
            boxes_to_show.append((int(x1), int(y1), int(x2), int(y2), name))

        found.sort(reverse=True)

        if not found:
            announce(
                "I don't see anything I recognize. This does not mean it is clear.")
        else:
            parts = [f"{name} {where}" for _, name, where in found[:4]]
            announce("I see " + ", ".join(parts))

        show_until = time.time() + 5

    # --- R: read text ---
    if key in (ord("r"), ord("R")):
        speak("Reading text, hold steady.")
        results = reader.readtext(frame)

        words = [text.strip() for (_, text, conf) in results
                 if conf >= 0.4 and text.strip()]

        if not words:
            announce(
                "I could not clearly see any text. Try holding it closer with good light.")
        else:
            announce("It says: " + " ".join(words))

stop_speaking()
camera.release()
cv2.destroyAllWindows()
