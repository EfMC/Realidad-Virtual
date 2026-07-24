import cv2
import os
import urllib.request


cascade_filename = 'haarcascade_frontalface_default.xml'
cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)

# Si la instalación de OpenCV no incluye el archivo o la ruta falla
if not os.path.exists(cascade_path):
    print("El archivo XML no se encontró en la instalación de OpenCV.")
    # Usaremos una copia local en la carpeta del proyecto
    cascade_path = cascade_filename
    
    # Si no existe localmente, lo descargamos automáticamente
    if not os.path.exists(cascade_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        print(f"Descargando {cascade_filename} desde el repositorio oficial de OpenCV...")
        urllib.request.urlretrieve(url, cascade_path)
        print("Descarga completada con éxito.")

# Cargar el clasificador
face_cascade = cv2.CascadeClassifier(cascade_path)

# Verificación de seguridad
if face_cascade.empty():
    raise RuntimeError(f"Error grave: No se pudo cargar el modelo desde '{cascade_path}'")



cap = cv2.VideoCapture(0) #Inicialización de la cámara

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    exit()

print("Face Tracking iniciado con éxito. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el fotograma de la cámara.")
        break

    # Voltear horizontalmente la imagen (efecto espejo)
    frame = cv2.flip(frame, 1)
    
    height, width, _ = frame.shape
    center_screen_x, center_screen_y = width // 2, height // 2


    # Escala de grises para mejorar rendimiento de detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=10, 
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        # Centro del rostro detectado
        face_center_x = x + (w // 2)
        face_center_y = y + (h // 2)

        # Desplazamiento respecto al centro del encuadre
        offset_x = face_center_x - center_screen_x
        offset_y = face_center_y - center_screen_y
        
        # Z estimado (inversamente proporcional al tamaño del rostro)
        estimated_z = int(10000 / w) if w > 0 else 0

        # Dibujar elementos gráficos
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

        # Dibujar líneas guía desde el centro hacia el rostro
        cv2.line(frame, (center_screen_x, center_screen_y), (face_center_x, face_center_y), (255, 0, 0), 1)

        # Mostrar coordenadas en pantalla
        text = f"X: {offset_x} | Y: {offset_y} | Z: {estimated_z}"
        cv2.putText(
            frame, text, (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Dibujar la cruz del centro de la pantalla
    cv2.drawMarker(frame, (center_screen_x, center_screen_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

    # Mostrar ventana
    cv2.imshow('Face Tracking - Realidad Virtual', frame)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
