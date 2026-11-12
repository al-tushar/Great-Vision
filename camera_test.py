import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not found. Check its connection and permissions.")

try:
    while True:
        success, image = camera.read()

        if not success:
            print("Could not get an image from the camera.")
            break

        cv2.imshow("Camera test", image)

        # Press Q while the camera window is focused to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()