import cv2
import sys
import numpy
import os


# Training recognizer
print "Training..."

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

print names

(img_width, img_height) = (112, 92)

(images, labels) = [numpy.array(l) for l in [images, labels]]


model = cv2.createFisherFaceRecognizer()
# model = cv2.createEigenFaceRecognizer()
# model = cv2.createLBPHFaceRecognizer()


model.train(images, labels)

model.save('face_recognizer_model.xml')
