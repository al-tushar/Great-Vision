import time

import cv2
import easyocr
import pyttsx3
from ultralytics import YOLO

print("RUNNING VERSION 4 (boxes + repeat)", flush=True)

last_message = ""


def speak(text):
    """Speak text. Used for short status messages."""
    print("Assistant:", text, flush=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


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
speak("Ready. S for objects, R to read text, H to hear the last message again, Q to quit.")

boxes_to_show = []     # list of (x1, y1, x2, y2, label)
show_until = 0         # time when the boxes disappear

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera error", flush=True)
        break

    preview = frame.copy()

    # Draw boxes from the last detection (only on the preview, not on `frame`)
    if time.time() < show_until:
        for x1, y1, x2, y2, label in boxes_to_show:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(preview, "S: Objects | R: Read | H: Repeat | Q: Quit", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Vision Assistant", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q")):
        break

    # --- H: repeat the last message ---
    if key in (ord("h"), ord("H")):
        if last_message:
            speak(last_message)
        else:
            speak("There is no message to repeat yet.")

    # --- S: objects ---
    if key in (ord("s"), ord("S")):
        speak("Checking objects.")
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

        show_until = time.time() + 5   # start the 5 seconds after speaking ends

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

camera.release()
cv2.destroyAllWindows()
