import cv2
import sys
import numpy
import os

(img_width, img_height) = (112, 92)

name = sys.argv[1]
path = os.path.join('att_faces', name)

if not os.path.isdir(path):
    os.mkdir(path)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
count = 1
key = 0

while count <= 20:
    retval, image = camera.read()

    # # optional
    # image = cv2.flip(image, 1, 0)

    # detectFace(image, count)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, (gray.shape[1] / 4, gray.shape[0] / 4))
    faces = classifier.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])

    if faces:
        i = faces[0]
        (x, y, w, h) = [v * 4 for v in i]
        face = gray[y:y + h, x:x + w]

        resize = cv2.resize(face, (img_width, img_height))

        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
               if n[0]!='.' ]+[0])[-1] + 1

        cv2.imwrite('%s/%s.png' % (path, pin), resize)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(image, name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.putText(image, name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        count += 1

    cv2.imshow('Video', image)

    key = cv2.waitKey(1000)

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
