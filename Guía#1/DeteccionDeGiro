import cv2
import numpy as np
import os
import urllib.request
import time

model_name = "face_detection_yunet_2023mar.onnx"
model_url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{model_name}"

if not os.path.exists(model_name):
    urllib.request.urlretrieve(model_url, model_name)

cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = cv2.FaceDetectorYN.create(
    model=model_name,
    config="",
    input_size=(width, height),
    score_threshold=0.5,
    nms_threshold=0.3
)

prev_angles = np.zeros(2)
alpha = 0.25  # Suavizado de movimiento (EMA)

# Variables de Calibración
calibrated = False
calibration_samples = []
pitch_bias = 0.0
yaw_bias = 0.0

print("Head Tracking YuNet con Auto-Calibracion.")
print("Paso 1: Mira de frente a la camara para calibrar el punto 0.0 deg.")
print("Controles: Presiona 'c' para recalibrar | Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)
    status_text = "Buscando rostro..."

    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

            landmarks = face[4:14].reshape((5, 2))
            eye_r, eye_l = landmarks[0], landmarks[1]
            nose = landmarks[2]
            mouth_r, mouth_l = landmarks[3], landmarks[4]

            # 1. CÁLCULO DE YAW (Giro Lateral)
            dist_eye_r = np.linalg.norm(nose - eye_r)
            dist_eye_l = np.linalg.norm(nose - eye_l)
            total_eye_dist = dist_eye_r + dist_eye_l
            
            if total_eye_dist > 0:
                raw_yaw = ((dist_eye_r - dist_eye_l) / total_eye_dist) * 90.0
            else:
                raw_yaw = 0.0

            # 2. CÁLCULO DE PITCH (Cabeceo Vertical)
            eyes_center = (eye_r + eye_l) / 2.0
            mouth_center = (mouth_r + mouth_l) / 2.0
            face_height = np.linalg.norm(eyes_center - mouth_center)

            if face_height > 0:
                nose_position = (nose[1] - eyes_center[1]) / face_height
                # Invertimos el signo para que subir la cara incremente el valor (+)
                raw_pitch = - (nose_position * 180.0)
            else:
                raw_pitch = 0.0

            # 3. PROCESO DE CALIBRACIÓN INICIAL
            if not calibrated:
                calibration_samples.append((raw_pitch, raw_yaw))
                status_text = f"Calibrando... Manten la cara de frente ({len(calibration_samples)}/20)"
                
                if len(calibration_samples) >= 20:
                    samples_arr = np.array(calibration_samples)
                    pitch_bias = np.mean(samples_arr[:, 0])
                    yaw_bias = np.mean(samples_arr[:, 1])
                    calibrated = True
                    print(f"-> Calibracion Completada! Offsets -> Pitch: {pitch_bias:.1f}, Yaw: {yaw_bias:.1f}")
            
            else:
                # Restamos el bias para garantizar 0.0° mirando al frente
                pitch_centered = (raw_pitch - pitch_bias) * 1.8
                yaw_centered = (raw_yaw - yaw_bias)

                # Suavizado de señal
                curr_angles = np.array([pitch_centered, yaw_centered])
                smooth_angles = prev_angles * (1 - alpha) + curr_angles * alpha
                prev_angles = smooth_angles

                pitch, yaw = smooth_angles[0], smooth_angles[1]

                # Clasificación de la postura
                if yaw > 8:
                    status_text = "Giro a la Derecha"
                elif yaw < -8:
                    status_text = "Giro a la Izquierda"
                elif pitch > 8:
                    status_text = "Mirando Arriba"
                elif pitch < -5:
                    status_text = "Mirando Abajo"
                else:
                    status_text = "Mirando de Frente"

                # Mostrar lecturas en pantalla
                cv2.putText(frame, f"Pitch (Arriba/Abajo): {pitch:.1f} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw   (Der/Izq)     : {yaw:.1f} deg", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Dibujar puntos en el rostro
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

            cv2.line(frame, (int(eyes_center[0]), int(eyes_center[1])), 
                            (int(mouth_center[0]), int(mouth_center[1])), (255, 255, 0), 2)

    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imshow("YuNet Head Tracking Autocalibrado", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Reset para recalibrar
        calibrated = False
        calibration_samples = []
        print("Recalibrando posicion neutra...")

cap.release()
cv2.destroyAllWindows()
