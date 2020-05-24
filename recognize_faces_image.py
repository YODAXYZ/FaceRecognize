# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/ex_1.png

import face_recognition
import argparse
import pickle
import cv2


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model to use: either `hog` or `cnn`")
    arg_parser = vars(ap.parse_args())
    return arg_parser


def square_in_face(boxes, names, image):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


if __name__ == "__main__":

    args = argument_parser()

    print("[INFO] loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())

    image = cv2.imread(args["image"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    square_in_face(boxes, names, image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
