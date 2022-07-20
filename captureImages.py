# face detection using opencv and haarcascade alogorithm and to save the face data in folder 'FaceSet'
import cv2
import os 


# to load the folder to store the face data 
dataset = "FaceSet"
# to set a folder for a paticular type of face data 
name = "first"
# to create a copy of name folder variable inside the dataset folder variable
path = os.path.join(dataset,name)
# to ckeck for a folder exits, if exits then 'donothing', if not then create the folder
if not os.path.isdir(path):
    os.mkdir(path)
# to save the image frame in particular size 
(width,height)=(130,100)


# to read the file in a variable
alg = "haar cascade files\\haarcascade_frontalface_alt.xml"
# to load the the '.xml' file in cv2 detection
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)


count = 1
while count<61:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.2,4)
    for (x,y,w,h)in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,225),5)
        # to crop the image to get the details of face only 
        faceOnly = grayImg[y:y+h,x:x+w]
        # to resize the image with specific height and width
        resizeImg = cv2.resize(faceOnly,(width,height))
        # to write each and every image with different image
        cv2.imwrite("%s/%s.jpg"%(path,count),resizeImg)
    print(count)
    count+=1

    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key==27:
        break
print("Image captured sucessfully")
cam.release()
cv2.destroyAllWindows()
