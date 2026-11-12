import cv2
import easyocr
import pyttsx3
from ultralytics import YOLO


def speak(text):
    print("Assistant:", text, flush=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


print("Loading object model...", flush=True)
model = YOLO("yolo11n.pt")

print("Loading text reader...", flush=True)
reader = easyocr.Reader(["en"], gpu=False)
print("Both models ready!", flush=True)

camera = cv2.VideoCapture(0)

speak("Ready. Press S for objects, R to read text, or Q to quit.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera error", flush=True)
        break

    preview = frame.copy()
    cv2.putText(
        preview,
        "S: Objects | R: Read Text | Q: Quit",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Vision Assistant", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q")):
        break

    if key in (ord("s"), ord("S")):
        speak("Checking objects.")
        results = model.predict(frame, conf=0.5, verbose=False)[0]

        annotated = frame.copy()
        labels = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                name = model.names[class_id]
                labels.append(name)

                # Draw a green box and the object name.
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    name,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Vision Assistant", annotated)
        cv2.waitKey(1)

        if not labels:
            speak("I don't see anything I recognize.")
        else:
            unique_things = list(set(labels))[:3]
            sentence = "I see " + " and ".join(unique_things)
            speak(sentence)

    if key in (ord("r"), ord("R")):
        speak("Reading text. Hold steady.")
        results = reader.readtext(frame)

        words = [
            text.strip()
            for (_, text, conf) in results
            if conf >= 0.5 and text.strip()
        ]

        if not words:
            speak("I could not clearly see any text.")
        else:
            speak("It says: " + " ".join(words))

camera.release()
cv2.destroyAllWindows()
