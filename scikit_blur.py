import cv2
import pyvirtualcam
from skimage.filters import gaussian_filter

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
virtual_cam = pyvirtualcam.Camera()
cap = cv2.VideoCapture(0)
virtual_cam.start()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = face_region
    frame = gaussian_filter(frame, sigma=23)
    virtual_cam.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

virtual_cam.stop()
virtual_cam.release()
cap.release()
cv2.destroyAllWindows()
