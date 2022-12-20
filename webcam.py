"""
Instale as bibliotecas necessárias, como a OpenCV ou a scikit-image.
Abra o vídeo da transmissão ao vivo usando uma dessas bibliotecas.
Faça o pré-processamento do vídeo, como convertê-lo para uma escala de cinza ou aplicar algum filtro.
Execute a remoção de fundo usando uma das funções disponíveis nas bibliotecas mencionadas.
Exiba o vídeo resultante com o fundo removido.
"""

# fazer um fundo borrado fora do rosto durante uma transmissão ao vivo usando 
# a webcam e o algoritmo de detecção de faces Haar cascades da OpenCV juntamente 
# com o filtro de desfoque gaussiano da OpenCV

import cv2
import numpy as np

# Carregue o classificador de faces Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abra a webcam
video = cv2.VideoCapture(0)

# Verifique se a webcam foi aberta com sucesso
if not video.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Execute um loop infinito para exibir os quadros da webcam
while True:
    # Obtenha o próximo quadro da webcam
    ret, frame = video.read()

    # Verifique se o quadro foi obtido com sucesso
    if not ret:
        break

    # Converta o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecte as faces no quadro
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Crie uma máscara preta com as dimensões da imagem
    mask = np.zeros_like(frame)

    # Desenhe os retângulos das faces detectadas na máscara
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)

    # Aplique o desfoque gaussiano na imagem inteira
    blur = cv2.GaussianBlur(frame, (15, 15), 0)

    blur = cv2.bitwise_and(blur, mask)

    # Exiba o quadro
    cv2.imshow("Fundo borrado", blur)

    # Verifique se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libere os recursos da webcam
video.release()

# Feche todas as janelas
cv2.destroyAllWindows()
