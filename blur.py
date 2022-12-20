import cv2

# Carregar o classificador de faces Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar quadro a quadro
    ret, frame = cap.read()

    # Converter o quadro para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no quadro
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenhar um retângulo em torno das faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Achar os contornos da face
        contours, _ = cv2.findContours(gray[y:y+h, x:x+w], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar o contorno da face
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

        # Copiar a região da face para uma nova imagem
        face_region = frame[y:y+h, x:x+w]
        face_region_copy = face_region.copy()

        # Aplicar o efeito de desfoque gaussiano no resto da imagem, exceto nas faces
        frame[:,:] = cv2.GaussianBlur(frame, (23, 23), 0)

        # Colar a região da face sem desfoque de volta na imagem
        frame[y:y+h, x:x+w] = face_region_copy

    # Exibir o quadro resultante
    cv2.imshow('Frame', frame)

    # Interromper o loop com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e destruir as janelas
cap.release()
cv2.destroyAllWindows()