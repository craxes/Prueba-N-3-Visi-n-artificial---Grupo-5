import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# ==========================================
# CONFIGURACIÓN DEL SISTEMA
# ==========================================
VIDEO_PATH = "AutoPersona.mp4"  # Archivo de video a procesar
MODEL_NAME = 'yolov8m.pt'       # Modelo YOLO a utilizar (n, s, m, l, x)
LINE_X_POS = 0.38               # Posición de la línea vertical (0.0 a 1.0)
FRAME_SKIP = 2                  # Procesar 1 de cada X frames para mejorar FPS

# Definición de clases (Basado en COCO Dataset)
# 0: Persona, 2: Auto, 5: Bus, 7: Camión
VEHICULOS = [2, 5, 7] 

# ==========================================
# INICIALIZACIÓN
# ==========================================
print("Cargando modelo YOLOv8...")
model = YOLO(MODEL_NAME)

# Inicializar captura de video
cap = cv2.VideoCapture(VIDEO_PATH)

# Historial de trayectorias: Almacena los últimos puntos de cada ID
track_history = defaultdict(lambda: [])

# Variables de Conteo
personas_in = 0   # Dirección: Izquierda -> Derecha
personas_out = 0  # Dirección: Derecha -> Izquierda
autos_in = 0      # Dirección: Izquierda -> Derecha (Solo autos)

frame_id = 0      # Contador interno para frame skipping

# ==========================================
# BUCLE PRINCIPAL DE PROCESAMIENTO
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fin del video o error de lectura.")
        break

    # 1. Optimización: Frame Skipping
    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    # 2. Preprocesamiento
    # Redimensionamos para estandarizar el área de detección
    frame = cv2.resize(frame, (720, 800))
    height, width, _ = frame.shape
    line_x = int(width * LINE_X_POS) # Calcular posición en píxeles de la línea

    # 3. Detección y Tracking con YOLO
    # persist=True asegura que el ID del objeto se mantenga entre cuadros
    results = model.track(
        frame,
        persist=True,
        conf=0.40,  # Umbral de confianza
        iou=0.50,   # Umbral de intersección
        verbose=False
    )

    # 4. Lógica de Negocio
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clases = results[0].boxes.cls.cpu().tolist()

        # Dibujar Línea de Referencia (Color Verde Inicial)
        color_linea = (0, 255, 0)
        
        # Procesar cada objeto detectado
        for box, track_id, clase in zip(boxes, track_ids, clases):
            x, y, w, h = box

            # Filtro de ruido: ignorar detecciones muy pequeñas
            if w < 30 or h < 30:
                continue

            cx, cy = int(x), int(y)

            # Actualizar historial de movimiento
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:  # Limitar historial a 30 cuadros
                track.pop(0)

            # Verificar Cruce de Línea
            if len(track) >= 2:
                prev_cx, _ = track[-2]
                curr_cx, _ = track[-1]

                # --> Cruce IZQ a DER (Entrada)
                if prev_cx < line_x and curr_cx >= line_x:
                    if clase in VEHICULOS:
                        autos_in += 1
                        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 4) # Flash Rojo
                    elif clase == 0: # Persona
                        personas_in += 1
                        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 4) # Flash Rojo

                # <-- Cruce DER a IZQ (Salida)
                elif prev_cx > line_x and curr_cx <= line_x:
                    if clase == 0: # Persona
                        personas_out += 1
                        cv2.line(frame, (line_x, 0), (line_x, height), (255, 0, 0), 4) # Flash Azul

            # Dibujar Objeto (Punto central y Caja)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            cv2.rectangle(
                frame,
                (int(x - w / 2), int(y - h / 2)),
                (int(x + w / 2), int(y + h / 2)),
                (255, 255, 0),
                2
            )

        # Dibujar la línea fija después de los flashes para que persista
        cv2.line(frame, (line_x, 0), (line_x, height), color_linea, 2)
        cv2.putText(frame, "Linea de Conteo", (line_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_linea, 2)

    # 5. Interfaz de Usuario (HUD)
    # Mostramos los contadores en la esquina superior izquierda
    cv2.putText(frame, f"Autos In: {autos_in}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Amarillo

    cv2.putText(frame, f"Personas In: {personas_in}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   # Verde

    cv2.putText(frame, f"Personas Out: {personas_out}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   # Rojo

    # Mostrar frame final
    cv2.imshow("Proyecto Final: Contador YOLOv8", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
