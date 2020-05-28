from vidgear.gears import VideoGear
import cv2

stream = VideoGear(source=0, stabilize = True).start()

while True:

    frame = stream.read()

    if frame is None:
        break

    cv2.imshow("Stabilized Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

stream.stop()