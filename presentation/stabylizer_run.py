from presentation.stabylizer import Stabilizer
import cv2
import time

imageCapture = cv2.VideoCapture(0)
imageCapture.open(0)
time.sleep(2.0)
frame=0
counter=0

stabilizer=Stabilizer()

while True:
    image=imageCapture.read()
    frame, result=stabilizer.stabilize(image, frame)

    cv2.imshow("Result", result)
    cv2.imshow("Image", image[1])
    key = cv2.waitKey(1) & 0xFF

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
imageCapture.release()