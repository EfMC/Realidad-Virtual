import cv2
import numpy as np
import pyvista as pv
import threading
import time
import os
import sys
import urllib.request
from tkinter import Tk, filedialog
import vtk

# ------------------------------------------------------------------
# 0. CONFIGURACIÓN INICIAL DE VTK
# ------------------------------------------------------------------
vtk.vtkObject.GlobalWarningDisplayOff()

# ------------------------------------------------------------------
# 1. DETECCIÓN FACIAL (HAAR CASCADE ROBUSTO)
# ------------------------------------------------------------------
cascade_filename = 'haarcascade_frontalface_default.xml'
cascade_path = None

if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
    try:
        possible_path = os.path.join(cv2.data.haarcascades, cascade_filename)
        if os.path.exists(possible_path):
            cascade_path = possible_path
    except Exception:
        pass

if not cascade_path or not os.path.exists(cascade_path):
    cascade_path = os.path.join(os.getcwd(), cascade_filename)
    if not os.path.exists(cascade_path):
        print("[INFO] Descargando haarcascade_frontalface_default.xml...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error fatal: No se pudo cargar el clasificador desde {cascade_path}")
    sys.exit()

# Variables globales compartidas
tracking_data = {
    "x": 0.0,
    "y": 0.0,
    "zoom": 1.0,  # Multiplicador de distancia basado en el acercamiento de la cara
    "frame": None,
    "running": True
}

# Ancho promedio de cara en píxeles a distancia neutra (~50-60 cm)
BASE_FACE_WIDTH = 160.0

# ------------------------------------------------------------------
# 2. HILO DE CAPTURA Y TRACKING EN 3D (X, Y, Z / ZOOM)
# ------------------------------------------------------------------
def face_tracking_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara web.")
        tracking_data["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Filtro de suavizado exponencial para evitar saltos en el zoom
    smooth_zoom = 1.0

    while tracking_data["running"]:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_center_x = x + (fw // 2)
            face_center_y = y + (fh // 2)

            # Normalización X e Y (-1.0 a 1.0)
            tracking_data["x"] = (face_center_x - center_x) / (w / 2.0)
            tracking_data["y"] = (face_center_y - center_y) / (h / 2.0)

            # ESTIMACIÓN DE Z (ZOOM):
            # A mayor ancho de la cara (fw), estás MÁS CERCA de la cámara -> Zoom In (multiplicador < 1.0)
            # A menor ancho de la cara (fw), estás MÁS LEJOS -> Zoom Out (multiplicador > 1.0)
            target_zoom = BASE_FACE_WIDTH / float(fw)
            
            # Limitar rango de zoom (Evita giros raros si te pegas demasiado o te alejas mucho)
            target_zoom = np.clip(target_zoom, 0.35, 2.2)

            # Filtro pasa-bajas para un movimiento ultra-fluido (Suavizado al 20%)
            smooth_zoom = (smooth_zoom * 0.8) + (target_zoom * 0.2)
            tracking_data["zoom"] = smooth_zoom

            # Dibujar elementos visuales de retroalimentación
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            
            # Mostrar nivel de Zoom en pantalla de cámara
            dist_text = f"Zoom: {1.0 / smooth_zoom:.2f}x"
            cv2.putText(frame, dist_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracking_data["frame"] = frame_rgb

        time.sleep(0.01)

    cap.release()
    print("[INFO] Cámara liberada correctamente.")

# ------------------------------------------------------------------
# 3. SELECTOR DE ARCHIVOS CAD
# ------------------------------------------------------------------
def load_cad_dialog():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo CAD 3D",
        filetypes=[("Archivos CAD 3D", "*.stl *.obj *.ply *.vtk *.gcode"), ("Todos los archivos", "*.*")]
    )
    root.destroy()
    if file_path and os.path.exists(file_path):
        try:
            print(f"Cargando CAD: {file_path}")
            return pv.read(file_path)
        except Exception as e:
            print(f"Error al cargar el archivo CAD: {e}")
    return None

# ------------------------------------------------------------------
# 4. APLICACIÓN PRINCIPAL CON PERSPECTIVA DINÁMICA
# ------------------------------------------------------------------
def main():
    current_mesh = load_cad_dialog()
    if current_mesh is None:
        print("No se seleccionó ningún archivo CAD. Cancelando ejecución.")
        sys.exit()

    t = threading.Thread(target=face_tracking_thread)
    t.daemon = True
    t.start()

    timeout = 50
    while tracking_data["frame"] is None and tracking_data["running"] and timeout > 0:
        time.sleep(0.1)
        timeout -= 1

    if tracking_data["frame"] is None:
        print("Error: Tiempo de espera agotado para recibir señal de la cámara.")
        tracking_data["running"] = False
        sys.exit()

    plotter = pv.Plotter(shape=(1, 2), title="Realidad Virtual: Tracking Perspectiva + Zoom Facial")

    def on_window_close(obj, event):
        tracking_data["running"] = False
        if plotter.render_window is not None:
            plotter.render_window.Finalize()

    if plotter.iren:
        plotter.iren.add_observer("ExitEvent", on_window_close)

    # Lado Izquierdo: Cámara Web
    plotter.subplot(0, 0)
    plotter.add_text("Camara Web (Tracking X, Y, Z)", font_size=11, color="white")
    
    h_img, w_img, _ = tracking_data["frame"].shape
    cam_plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=w_img, j_size=h_img)
    
    texture_data = pv.numpy_to_texture(tracking_data["frame"])
    cam_actor = plotter.add_mesh(cam_plane, texture=texture_data)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    # Lado Derecho: Render Modelo CAD 3D
    plotter.subplot(0, 1)
    plotter.set_background("slategray")
    plotter.add_axes()

    cad_actor = plotter.add_mesh(current_mesh, color="gainsboro", show_edges=True, smooth_shading=True, specular=0.5)

    plotter.reset_camera()
    initial_cam_pos, initial_focal_point, _ = plotter.camera_position
    base_x, base_y, base_z = initial_cam_pos
    focal_x, focal_y, focal_z = initial_focal_point

    dist_base = np.sqrt((base_x - focal_x)**2 + (base_y - focal_y)**2 + (base_z - focal_z)**2)

    MAX_ANGULO_X = np.radians(90)
    MAX_ANGULO_Y = np.radians(90)
    FACTOR_DISTANCIA = 2
    R_base = dist_base * FACTOR_DISTANCIA

    def callback_cambiar_cad():
        new_mesh = load_cad_dialog()
        if new_mesh is not None:
            nonlocal cad_actor, initial_focal_point, R_base, focal_x, focal_y, focal_z
            plotter.subplot(0, 1)
            plotter.remove_actor(cad_actor)
            cad_actor = plotter.add_mesh(new_mesh, color="gainsboro", show_edges=True, smooth_shading=True, specular=0.5)
            plotter.reset_camera()
            
            c_pos, f_pt, _ = plotter.camera_position
            initial_focal_point = f_pt
            focal_x, focal_y, focal_z = f_pt
            d = np.sqrt((c_pos[0]-f_pt[0])**2 + (c_pos[1]-f_pt[1])**2 + (c_pos[2]-f_pt[2])**2)
            R_base = d * FACTOR_DISTANCIA

    plotter.add_checkbox_button_widget(
        lambda state: callback_cambiar_cad(),
        value=False,
        color_on="dodgerblue",
        color_off="dodgerblue",
        size=30,
        position=(10, 10)
    )
    plotter.add_text("   Cambiar CAD", position=(45, 15), font_size=10, color="white")

    plotter.show(interactive_update=True)

    # --------------------------------------------------------------
    # BUCLE PRINCIPAL DE RENDERIZADO
    # --------------------------------------------------------------
    try:
        while tracking_data["running"]:
            if plotter.render_window is None or plotter.render_window.GetNeverRendered():
                break

            # 1. Actualizar fotograma de cámara
            if tracking_data["frame"] is not None:
                plotter.subplot(0, 0)
                cam_actor.texture = pv.numpy_to_texture(tracking_data["frame"])

            # 2. Actualizar posición de la cámara 3D con tracking X, Y, Z (Zoom)
            plotter.subplot(0, 1)
            tx = tracking_data["x"]
            ty = tracking_data["y"]
            z_factor = tracking_data["zoom"]

            # Radio dinámico: varia según si te acercas o alejas de la webcam
            R_dinamico = R_base * z_factor

            theta = -tx * MAX_ANGULO_X
            phi = ty * MAX_ANGULO_Y
            phi = np.clip(phi, np.radians(-80), np.radians(80))

            # Coordenadas esféricas usando el radio de distancia ajustado
            cam_x = (focal_x + R_dinamico * np.sin(theta) * np.cos(phi)) * -1
            cam_y = (focal_y + R_dinamico * np.sin(phi)) * -1
            cam_z = focal_z + R_dinamico * np.cos(theta) * np.cos(phi)

            up_y = np.cos(phi)
            up_z = -np.sin(phi) if phi != 0 else 0.0

            plotter.camera_position = [
                (cam_x, cam_y, cam_z),
                initial_focal_point,
                (0, up_y, up_z)
            ]

            plotter.update()
            time.sleep(0.016)

    except Exception as e:
        print(f"[INFO] Finalizando loop principal: {e}")

    tracking_data["running"] = False
    if plotter.render_window is not None:
        plotter.render_window.Finalize()
    plotter.close()
    plotter.deep_clean()
    print("[INFO] Aplicación finalizada limpiamente.")

if __name__ == "__main__":
    main()