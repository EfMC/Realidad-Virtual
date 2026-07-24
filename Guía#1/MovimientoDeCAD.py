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
# 0. CONFIGURACIÓN INICIAL DE VTK (PREVENCIÓN DE ERRORES DE SHADERS)
# ------------------------------------------------------------------
# Desactiva los mensajes de advertencia de VTK para evitar que ensucien la consola
# al liberar recursos de OpenGL en el recolector de basura.
vtk.vtkObject.GlobalWarningDisplayOff()

# ------------------------------------------------------------------
# 1. DETECCIÓN FACIAL (HAAR CASCADE)
# ------------------------------------------------------------------
cascade_filename = 'haarcascade_frontalface_default.xml'
cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)

if not os.path.exists(cascade_path):
    cascade_path = cascade_filename
    if not os.path.exists(cascade_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Variables globales compartidas
tracking_data = {
    "x": 0.0,
    "y": 0.0,
    "frame": None,
    "running": True
}

# ------------------------------------------------------------------
# 2. HILO DE CAPTURA DE CÁMARA (CON FIX MSMF -> DSHOW)
# ------------------------------------------------------------------
def face_tracking_thread():
    # Usar DirectShow para evitar errores de Media Foundation en Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        # Fallback si DSHOW falla
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara web.")
        tracking_data["running"] = False
        return

    # Ajustar resolución básica para asegurar estabilidad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

            # Dibujar recuadro de seguimiento
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

        # Convertir BGR a RGB para PyVista
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracking_data["frame"] = frame_rgb

        time.sleep(0.01)

    cap.release()
    print("[INFO] Cámara liberada correctamente.")

# ------------------------------------------------------------------
# 3. SELECTOR DE ARCHIVOS CAD (TKINTER)
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
# 4. APLICACIÓN PRINCIPAL
# ------------------------------------------------------------------
def main():
    # 1. Solicitar el archivo CAD
    current_mesh = load_cad_dialog()
    if current_mesh is None:
        print("No se seleccionó ningún archivo CAD. Cancelando ejecución.")
        sys.exit()

    # 2. Iniciar hilo de seguimiento facial
    t = threading.Thread(target=face_tracking_thread)
    t.daemon = True
    t.start()

    # Esperar a que la cámara capture el primer fotograma
    timeout = 50  # 5 segundos máximo de espera
    while tracking_data["frame"] is None and tracking_data["running"] and timeout > 0:
        time.sleep(0.1)
        timeout -= 1

    if tracking_data["frame"] is None:
        print("Error: Tiempo de espera agotado para recibir señal de la cámara.")
        tracking_data["running"] = False
        sys.exit()

    # 3. Configurar Plotter con 2 Sub-ventanas (1 fila, 2 columnas)
    plotter = pv.Plotter(shape=(1, 2), title="Sistema CAD + Tracking Facial Unificado")

    # --------------------------------------------------------------
    # MANEJO DE CIERRE LIMPIO DE OPENGL
    # --------------------------------------------------------------
    def on_window_close(obj, event):
        """ Callback que destruye el contexto de OpenGL en el orden correcto. """
        tracking_data["running"] = False
        if plotter.render_window is not None:
            plotter.render_window.Finalize()

    # Escuchar cuando el usuario cierra la ventana (Evento de VTK)
    if plotter.iren:
        plotter.iren.add_observer("ExitEvent", on_window_close)

    # --------------------------------------------------------------
    # SUB-VENTANA IZQUIERDA (CÁMARA WEB)
    # --------------------------------------------------------------
    plotter.subplot(0, 0)
    plotter.add_text("Camara Web (Tracking)", font_size=11, color="white")
    
    h_img, w_img, _ = tracking_data["frame"].shape
    cam_plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=w_img, j_size=h_img)
    
    texture_data = pv.numpy_to_texture(tracking_data["frame"])
    cam_actor = plotter.add_mesh(cam_plane, texture=texture_data)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    # --------------------------------------------------------------
    # SUB-VENTANA DERECHA (MODELO CAD 3D)
    # --------------------------------------------------------------
    plotter.subplot(0, 1)
    plotter.set_background("slategray")
    plotter.add_axes()

    cad_actor = plotter.add_mesh(current_mesh, color="gainsboro", show_edges=True, smooth_shading=True, specular=0.5)

    plotter.reset_camera()
    initial_cam_pos, initial_focal_point, initial_view_up = plotter.camera_position
    base_x, base_y, base_z = initial_cam_pos
    focal_x, focal_y, focal_z = initial_focal_point

    dist_base = np.sqrt((base_x - focal_x)**2 + (base_y - focal_y)**2 + (base_z - focal_z)**2)

    MAX_ANGULO_X = np.radians(90)
    MAX_ANGULO_Y = np.radians(90)
    FACTOR_DISTANCIA = 2
    R = dist_base * FACTOR_DISTANCIA

    # Callback para cambiar de archivo CAD
    def callback_cambiar_cad():
        new_mesh = load_cad_dialog()
        if new_mesh is not None:
            nonlocal cad_actor, initial_focal_point, R, focal_x, focal_y, focal_z
            plotter.subplot(0, 1)
            plotter.remove_actor(cad_actor)
            cad_actor = plotter.add_mesh(new_mesh, color="gainsboro", show_edges=True, smooth_shading=True, specular=0.5)
            plotter.reset_camera()
            
            c_pos, f_pt, _ = plotter.camera_position
            initial_focal_point = f_pt
            focal_x, focal_y, focal_z = f_pt
            d = np.sqrt((c_pos[0]-f_pt[0])**2 + (c_pos[1]-f_pt[1])**2 + (c_pos[2]-f_pt[2])**2)
            R = d * FACTOR_DISTANCIA

    # Botón flotante "Cambiar CAD"
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

            # 1. Actualizar fotograma de la cámara web (Lado Izquierdo)
            if tracking_data["frame"] is not None:
                plotter.subplot(0, 0)
                cam_actor.texture = pv.numpy_to_texture(tracking_data["frame"])

            # 2. Actualizar la perspectiva esférica del CAD (Lado Derecho)
            plotter.subplot(0, 1)
            tx = tracking_data["x"]
            ty = tracking_data["y"]

            theta = -tx * MAX_ANGULO_X
            phi = ty * MAX_ANGULO_Y
            phi = np.clip(phi, np.radians(-80), np.radians(80))

            # Coordenadas esféricas puras (Distancia constante R)
            cam_x = (focal_x + R * np.sin(theta) * np.cos(phi)) * -1
            cam_y = (focal_y + R * np.sin(phi)) * -1
            cam_z = focal_z + R * np.cos(theta) * np.cos(phi)

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

    # --------------------------------------------------------------
    # LIMPIEZA FINAL Y DESTRUCCIÓN ORDENADA
    # --------------------------------------------------------------
    tracking_data["running"] = False
    
    # 1. Liberar la ventana gráfica explícitamente
    if plotter.render_window is not None:
        plotter.render_window.Finalize()
    
    # 2. Cerrar el plotter y limpiar memoria
    plotter.close()
    plotter.deep_clean()
    
    print("[INFO] Aplicación finalizada limpiamente.")

if __name__ == "__main__":
    main()