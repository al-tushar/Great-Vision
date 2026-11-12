import cv2
import easyocr
import pyttsx3
from ultralytics import YOLO

print("RUNNING VERSION 3 (with positions)", flush=True)


def speak(text):
    print("Assistant:", text, flush=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


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
speak("Ready. Press S for objects, R to read text, or Q to quit.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera error", flush=True)
        break

    preview = frame.copy()
    cv2.putText(preview, "S: Objects | R: Read Text | Q: Quit", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Vision Assistant", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q")):
        break

    if key in (ord("s"), ord("S")):
        speak("Checking objects.")
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        height, width = frame.shape[:2]

        found = []
        for box in results.boxes:
            name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            where = position_of((x1 + x2) / 2, width)
            found.append((area, name, where))

        found.sort(reverse=True)

        if not found:
            speak("I don't see anything I recognize. This does not mean it is clear.")
        else:
            parts = [f"{name} {where}" for _, name, where in found[:4]]
            speak("I see " + ", ".join(parts))

    if key in (ord("r"), ord("R")):
        speak("Reading text, hold steady.")
        results = reader.readtext(frame)

        words = [text.strip() for (_, text, conf) in results
                 if conf >= 0.5 and text.strip()]

        if not words:
            speak(
                "I could not clearly see any text. Try holding it closer with good light.")
        else:
            speak("It says: " + " ".join(words))

camera.release()
cv2.destroyAllWindows()
