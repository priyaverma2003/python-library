# import cv2
# import cv2.data
# model="/haarcascade_frontalface_default.xml"
# classifier=cv2.CascadeClassifier(cv2.data.harrcascades+model)
# image = cv2.imread("./celeb.jpeg")
# gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# faces=classifier.detect


# image=cv2.imread("./h2.jpeg")
# image=cv2.rectangle(image,(100,100),(250,250))
# while True: 
#     cv2.imshow("My Image",image)
#     if cv2.waitKey(1)==ord("q"):
#         break
    
import cv2
import cv2.data
import os
print(os.listdir(cv2.data.haarcascades))

model = "/haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + model)
image = cv2.imread("./modi.jpeg")
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(image,1.3,6)
print(faces)

for face in faces:
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
while True:
    cv2.imshow("My Image",image)
    if cv2.waitKey(1) == ord("q"):
        break

    

