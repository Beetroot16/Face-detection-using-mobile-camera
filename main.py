import requests
import cv2
from numpy import array,uint8

url = r"http://192.168.168.241:8080/shot.jpg"

face_cascade = cv2.CascadeClassifier('FINAL_CODES\without_serial\haarcascade_frontalface_default.xml')

while True:
    online_vid = requests.get(url)
    online_vid_arr = array(bytearray(online_vid.content),dtype = uint8)
    online_img = cv2.imdecode(online_vid_arr, -1)

    img = online_img
    image = img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 2)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+y, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('image_window',image)

    if cv2.waitKey(1) and 0xFF == ('q'):
        break
online_vid.release()
cv2.destroyAllWindows()