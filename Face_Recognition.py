
# to imort the required libraries
import cv2
import numpy
import os


# loading the face recognition algorithm
alg = 'haar cascade files\\haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(alg)
dataSet = 'FaceSet'
print("Training Phase of Algorithm...")

# loading the data of 'first' ('0')
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dataSet):
    for subdir in dirs:
        names[id] = subdir
        subjectPath = os.path.join(dataSet, subdir)
        for filename in os.listdir(subjectPath):
            path = subjectPath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            # print(labels)
        id += 1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
(width, height) = (130, 100)

# to load the classfier
model = cv2.face.LBPHFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

camera = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, img) = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        if prediction[1] < 800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]),
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, "Unknown", (x-10, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0))
            if(cnt > 100):
                print("Unknown Person Detected")
                cv2.imwrite("unknown.jpg", img)
                cnt = 0
    cv2.imshow("FaceRecognition", img)
    if ord('q') == cv2.waitKey(10):
        break
camera.release()
cv2.destroyAllWindows()
