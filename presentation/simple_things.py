import cv2
import keyboard
import numpy as np


def gray_image():
    cam = cv2.VideoCapture(0)

    while True:
        _, img = cam.read()
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


def reverse(mirror=True):
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Frame', img)

        if cv2.waitKey(33) == 27:  # Esc key to stop
            break

    cv2.destroyAllWindows()


def canny():
    camera = cv2.VideoCapture(0)
    while True:
        _, img = camera.read()
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        gradient_img = cv2.Canny(gray, 100, 150)

        cv2.imshow("Frame", gradient_img)

        if cv2.waitKey(33) == 27:
            break

        cv2.destroyAllWindows()


def gradient_sobel():
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()

        image = np.float32(frame.copy()) / 255.0

        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        output_img = mag

        cv2.imshow("Frame", output_img)

        if cv2.waitKey(33) == 27:
            break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        if keyboard.is_pressed('1'):
            gray_image()

        if keyboard.is_pressed('2'):
            reverse()

        if keyboard.is_pressed('3'):
            canny()

        if keyboard.is_pressed('4'):
            gradient_sobel()



            

