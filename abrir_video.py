import cv2

# Abra o arquivo de vídeo
video = cv2.VideoCapture("video.mp4")

# Verifique se o vídeo foi aberto com sucesso
if not video.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Execute um loop infinito para exibir os quadros do vídeo
while True:
    # Obtenha o próximo quadro do vídeo
    ret, frame = video.read()

    # Verifique se o quadro foi obtido com sucesso
    if not ret:
        break

    # Exiba o quadro
    cv2.imshow("Quadro", frame)

    # Verifique se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libere os recursos do vídeo
video.release()

# Feche todas as janelas
cv2.destroyAllWindows()
