import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('eye.xml')
i=0

kisi_id=input('Enter id')
while True:
    _, img =cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataEye/face-" + kisi_id + '.' + str(i) + ".jpg", gray[y:y + h , x :x + w])
        cv2.rectangle(img, (x , y), (x + w, y + h), (225, 0, 0), 2)
        cv2.imshow('image', img[y :y + h, x :x + w])
        cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break

