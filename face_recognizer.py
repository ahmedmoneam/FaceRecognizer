import cv2
import sys
import numpy
import os


(img_width, img_height) = (112, 92)

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk('att_faces'):

    for subdir in dirs:

        names[id] = subdir
        subpath = os.path.join('att_faces', subdir)
        for filename in os.listdir(subpath):
            path = subpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))

        id += 1


model = cv2.cv.createFisherFaceRecognizer()
# model = cv2.cv.createEigenFaceRecognizer()
# model = cv2.cv.createLBPHFaceRecognizer()

model.load('face_recognizer_model.xml')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

while True:

    retval, image = camera.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resize = cv2.resize(image, (image.shape[1] / 4, image.shape[0] / 4))

    faces = classifier.detectMultiScale(resize)

    # for (x, y, w, h) in faces:
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * 4 for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))

        prediction = model.predict(face_resize)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)

        if prediction[1] < 500:
            cv2.putText(image, "%s - %.0f" % (names[prediction[0]], prediction[1]),
                (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unknown",
                (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0))

    cv2.imshow('Camera', image)

    quit = cv2.waitKey(10)
    if quit is 27:
        break

camera.release()
cv2.destroyAllWindows()
