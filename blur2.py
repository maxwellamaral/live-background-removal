import cv2
import pyvirtualcam


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
neck_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')

cap = cv2.VideoCapture(0)

# Criar uma câmera virtual
virtual_cam = pyvirtualcam.Camera(width=640, height=480, fps=20)

# Iniciar o fluxo de vídeo da câmera virtual
virtual_cam.start()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    necks = neck_cascade.detectMultiScale(gray, 1.3, 5)
    # contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        face_region_copy = face_region.copy()
        frame[:,:] = cv2.GaussianBlur(frame, (23, 23), 0)
        frame[y:y+h, x:x+w] = face_region_copy
        
        # face_region = cv2.GaussianBlur(face_region, (23, 23), 0)
        # frame[y:y+h, x:x+w] = face_region
    for (x, y, w, h) in necks:
        neck_region = frame[y:y+h, x:x+w]
        neck_region_copy = neck_region.copy()
        frame[:,:] = cv2.GaussianBlur(frame, (23, 23), 0)
        frame[y:y+h, x:x+w] = neck_region_copy
        
        # neck_region = cv2.GaussianBlur(neck_region, (23, 23), 0)
        # frame[y:y+h, x:x+w] = neck_region
    cv2.imshow('Frame', frame)
    virtual_cam.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Parar o fluxo de vídeo da câmera virtual
virtual_cam.stop()

# Liberar a câmera virtual
virtual_cam.release()