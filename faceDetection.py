# face detection using opencv and haarcascade alogorithm and to capture the face frame
import cv2

# to read the file in a variable
alg = "D:\\Study\\AI internship\\Day_5\\haar cascade files\\haarcascade_frontalface_alt.xml"
# to load the the '.xml' file in cv2 detection
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    textDisplay = "No Person Detected"
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.2,4)
    for (x,y,w,h)in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,225),5)
        print(face.shape)
        textDisplay = "Person Detected Clearly"
        print(textDisplay)
        ImgNew = cv2.putText(img,textDisplay,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,225,225),5)
        # to display the last frame when face was captured
        cv2.imshow("Face Detection",img)

    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key=='q':
        break
cam.release()
cv2.destroyAllWindows()
