import subprocess
import sys
import time
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

print("RUNNING VERSION 6 FREE (YOLO + OCR + BLIP describe)", flush=True)

WORKER_PATH = Path(__file__).parent / "speak_worker.py"

last_message = ""
speech_process = None
caption_processor = None
caption_model = None


def stop_speaking():
    global speech_process
    try:
        if speech_process is not None and speech_process.poll() is None:
            speech_process.terminate()
    except Exception as e:
        print("Stop speech error:", e, flush=True)
    speech_process = None


def speak(text):
    global speech_process
    print("Assistant:", text, flush=True)
    try:
        stop_speaking()
        speech_process = subprocess.Popen(
            [sys.executable, str(WORKER_PATH), text])
    except Exception as e:
        print("Speech error:", e, flush=True)


def announce(text):
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


def get_caption_model():
    """Load BLIP only when first needed, to keep startup fast."""
    global caption_processor, caption_model
    if caption_model is not None:
        return True
    try:
        print("Loading BLIP... first time downloads ~1GB", flush=True)
        from transformers import BlipForConditionalGeneration, BlipProcessor
        model_id = "Salesforce/blip-image-captioning-base"
        caption_processor = BlipProcessor.from_pretrained(model_id)
        caption_model = BlipForConditionalGeneration.from_pretrained(model_id)
        print("BLIP ready!", flush=True)
        return True
    except Exception as e:
        print("BLIP load error:", e, flush=True)
        return False


def describe_scene(frame):
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = caption_processor(pil_img, return_tensors="pt")
    out = caption_model.generate(**inputs, max_length=40)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


# --- Load YOLO + OCR at start ---
print("Loading YOLO...", flush=True)
model = YOLO("yolo11n.pt")

print("Loading EasyOCR...", flush=True)
reader = easyocr.Reader(["en"], gpu=False)
print("Models ready! BLIP will load on first G press.", flush=True)

camera = cv2.VideoCapture(0)
speak("Ready. S for objects, R to read, G to describe, H to repeat, X to stop, Q to quit.")

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

    cv2.putText(preview, "S: Objects | R: Read | G: Describe", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, "H: Repeat | X: Stop | Q: Quit", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Vision Assistant", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q")):
        break

    if key in (ord("x"), ord("X")):
        stop_speaking()
        print("Speech stopped.", flush=True)

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
        if results.boxes is not None:
            for box in results.boxes:
                name = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                where = position_of((x1 + x2) / 2, width)
                found.append((area, name, where))
                boxes_to_show.append(
                    (int(x1), int(y1), int(x2), int(y2), name))

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

    # --- G: smart describe ---
    if key in (ord("g"), ord("G")):
        speak("Describing scene, please wait and hold steady.")
        if not get_caption_model():
            announce(
                "Description model could not load. Check internet and free disk space.")
        else:
            try:
                caption = describe_scene(frame)
                if caption:
                    announce("I think I see: " + caption)
                else:
                    announce("I could not describe the scene.")
            except Exception as e:
                print("Caption error:", e, flush=True)
                announce("I could not describe the scene.")

stop_speaking()
camera.release()
cv2.destroyAllWindows()
