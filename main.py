import cv2
import json
import os
import numpy as np
import threading          
import winsound  
import serial                    
import serial.tools.list_ports
from datetime import datetime

# ── Detectores ───────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ── Configuración ─────────────────────────────────────────────────────────────
DB_FILE     = "empleados.json"
ROSTROS_DIR = "rostros"
LOG_FILE    = "accesos_log.txt"
os.makedirs(ROSTROS_DIR, exist_ok=True)
# sonido 
_sonido_activo = False   
# ── Arduino (variables compartidas) ──────────────────────────────────────────
pir_movimiento  = False   # True si PIR detectó movimiento
arduino_ok      = False   # True si Arduino está conectado
acceso_pin_ok   = False   # True si ingresó PIN correcto
acceso_pin_fail = False   # True si ingresó PIN incorrecto
arduino_serial  = None    # objeto serial                 

def emitir_sonido(tipo):
    global _sonido_activo
    if _sonido_activo:
        return
    def _play():
        global _sonido_activo
        _sonido_activo = True
        if tipo == 'rojo':
            for _ in range(4):
                winsound.Beep(1200, 180)
                winsound.Beep(800,  120)
        elif tipo == 'verde':
            winsound.Beep(880,  120)
            winsound.Beep(1100, 180)
        _sonido_activo = False
    threading.Thread(target=_play, daemon=True).start()

# ── Arduino Serial ────────────────────────────────────────────────────────────
def conectar_arduino():
    """Busca el Arduino por su Vendor ID oficial (2341 = Arduino)."""
    global arduino_serial, arduino_ok
    
    ARDUINO_VIDS = ["VID:PID=2341", "VID:PID=2A03", "VID:PID=1A86", "VID:PID=0403"]
    
    print("🔍 Puertos detectados:")
    for port in serial.tools.list_ports.comports():
        print(f"   {port.device:8s} | {port.description} | {port.hwid}")
        if any(vid in port.hwid for vid in ARDUINO_VIDS):
            try:
                arduino_serial = serial.Serial(port.device, 9600, timeout=1)
                arduino_ok     = True
                print(f"✅ Arduino detectado en {port.device} (VID oficial)")
                return
            except Exception as e:
                print(f"   ⚠️  Error abriendo {port.device}: {e}")

    print("⚠️  Arduino no encontrado — sistema funciona sin él")

def hilo_arduino():
    """Lee mensajes de Arduino en background sin bloquear la cámara."""
    global pir_movimiento, acceso_pin_ok, acceso_pin_fail
    while True:
        if not arduino_ok or arduino_serial is None:
            break
        try:
            if arduino_serial.in_waiting > 0:
                linea = arduino_serial.readline().decode("utf-8").strip()

                if linea == "PIR:1":
                    pir_movimiento = True
                    print("🔴 PIR: movimiento detectado")

                elif linea == "PIR:0":
                    pir_movimiento = False

                elif linea == "ACCESO_OK":
                    acceso_pin_ok   = True
                    acceso_pin_fail = False
                    print("✅ PIN correcto")

                elif linea == "ACCESO_DENEGADO":
                    acceso_pin_fail = True
                    acceso_pin_ok   = False
                    print("❌ PIN incorrecto")

        except:
            break

def enviar_arduino(mensaje):
    """Manda un mensaje al Arduino de forma segura."""
    if arduino_ok and arduino_serial:
        try:
            arduino_serial.write((mensaje + "\n").encode("utf-8"))
        except:
            pass

# ── Reconocedor LBPH ─────────────────────────────────────────────────────────
reconocedor = cv2.face.LBPHFaceRecognizer_create()
labels      = {}
entrenado   = False

# ── Historial en pantalla (últimos 4) ─────────────────────────────────────────
historial = []   # lista de dicts: {nombre, hora, confianza}

def registrar_log(nombre, confianza):
    """Guarda cada acceso en el archivo de auditoría."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {nombre:<20} | confianza: {confianza}%\n")

def cargar_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {"empleados": []}

def guardar_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def entrenar_reconocedor():
    global entrenado
    db = cargar_db()
    imagenes, ids = [], []
    for emp in db["empleados"]:
        foto = emp.get("foto", "")
        if not os.path.exists(foto):
            continue
        img = cv2.imread(foto, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        imagenes.append(img)
        ids.append(emp["id"])
        labels[emp["id"]] = emp["nombre"]
    if imagenes:
        reconocedor.train(imagenes, np.array(ids))
        entrenado = True
        print(f"✅ Reconocedor entrenado con {len(imagenes)} persona(s).")
    else:
        print("⚠️  No hay fotos para entrenar aún.")

def registrar_empleado(frame, x, y, w, h, lentes):
    db     = cargar_db()
    emp_id = len(db["empleados"]) + 1
    nombre = input(f"📝 Nombre del empleado #{emp_id}: ").strip()
    if not nombre:
        nombre = f"Empleado_{emp_id}"
    rostro    = frame[y:y+h, x:x+w]
    foto_path = os.path.join(ROSTROS_DIR, f"{nombre.lower().replace(' ','_')}.jpg")
    cv2.imwrite(foto_path, rostro)
    nuevo = {
        "id":             emp_id,
        "nombre":         nombre,
        "foto":           foto_path,
        "fecha_registro": datetime.now().strftime("%Y-%m-%d"),
        "ultima_entrada": datetime.now().strftime("%H:%M:%S"),
        "lentes":         lentes
    }
    db["empleados"].append(nuevo)
    guardar_db(db)
    print(f"✅ {nombre} registrado!")
    entrenar_reconocedor()
    return nombre

def actualizar_entrada(nombre):
    db = cargar_db()
    for emp in db["empleados"]:
        if emp["nombre"] == nombre:
            emp["ultima_entrada"] = datetime.now().strftime("%H:%M:%S")
    guardar_db(db)

def get_emp_info(nombre):
    """Devuelve el dict completo del empleado desde la DB."""
    db = cargar_db()
    for emp in db["empleados"]:
        if emp["nombre"] == nombre:
            return emp
    return None

# ── Helpers visuales ─────────────────────────────────────────────────────────
COLOR_CONOCIDO    = (0,   220, 100)
COLOR_DESCONOCIDO = (80,  80,  80)
COLOR_LENTES      = (255, 180,   0)
COLOR_PANEL       = (20,  20,  20)

# ── Alerta intruso ────────────────────────────────────────────────────────────
FRAMES_ALERTA      = 90         # ~3 seg a 30fps antes de activar alerta
desconocido_frames = 0          # contador acumulado
alerta_activa      = False
parpadeo_contador  = 0          # controla el ON/OFF visual
parpadeo_visible   = True

def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, fh), _ = cv2.getTextSize(text, font, 0.65, 2)
    cv2.rectangle(frame, (x, y - fh - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, 0.65, (255, 255, 255), 2)

def draw_barra_confianza(frame, x, y, w, confianza, color):
    """Dibuja una pequeña barra de confianza debajo del recuadro facial."""
    bw   = w
    fill = int(bw * confianza / 100)
    cv2.rectangle(frame, (x, y + 4), (x + bw, y + 12), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y + 4), (x + fill, y + 12), color, -1)
    cv2.putText(frame, f"{confianza}%", (x + bw + 6, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def draw_panel_empleado(frame, emp_info, confianza, lentes):
    """Panel lateral derecho con datos del empleado reconocido."""
    ph, pw = frame.shape[:2]
    px = pw - 260
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 10, 0), (pw, 210), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    ahora = datetime.now()
    font  = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "ACCESO AUTORIZADO", (px - 5, 22),
                font, 0.5, COLOR_CONOCIDO, 1)
    cv2.line(frame, (px - 10, 28), (pw, 28), (40, 40, 40), 1)

    nombre_corto = emp_info["nombre"][:22]
    cv2.putText(frame, nombre_corto, (px - 5, 52),
                font, 0.65, (255, 255, 255), 2)

    datos = [
        ("ID",        f"#{emp_info['id']:03d}"),
        ("Hora",      ahora.strftime("%H:%M:%S")),
        ("Fecha",     ahora.strftime("%d/%m/%Y")),
        ("Confianza", f"{confianza}%"),
        ("Lentes",    "Si" if lentes else "No"),
        ("Registro",  emp_info.get("fecha_registro", "—")),
    ]
    for i, (clave, valor) in enumerate(datos):
        yy = 78 + i * 22
        cv2.putText(frame, f"{clave}:", (px - 5, yy),
                    font, 0.42, (130, 130, 130), 1)
        cv2.putText(frame, valor, (px + 80, yy),
                    font, 0.42, (220, 220, 220), 1)

def draw_historial(frame, historial):
    """Muestra los últimos accesos en la esquina inferior derecha."""
    ph, pw = frame.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    base_y = ph - 20
    cv2.putText(frame, "Ultimos accesos:", (pw - 260, base_y - len(historial) * 20 - 10),
                font, 0.42, (100, 100, 100), 1)
    for i, reg in enumerate(reversed(historial[-4:])):
        texto = f"{reg['hora']}  {reg['nombre'][:14]:<14}  {reg['confianza']}%"
        yy    = base_y - (len(historial[-4:]) - 1 - i) * 20
        cv2.putText(frame, texto, (pw - 260, yy),
                    font, 0.4, (160, 160, 160), 1)
        

def draw_semaforo(frame, estado):
    """
    estado: 'rojo' | 'amarillo' | 'verde'
    Dibuja un semáforo en la esquina superior izquierda.
    """
    ph, pw = frame.shape[:2]
    # Fondo del semáforo
    sx, sy = 20, 110   # posición (debajo del contador de rostros y fecha)
    cv2.rectangle(frame, (sx, sy), (sx + 36, sy + 100), (30, 30, 30), -1)
    cv2.rectangle(frame, (sx, sy), (sx + 36, sy + 100), (80, 80, 80), 1)

    # Colores apagados (oscuro) y encendidos
    c_rojo     = (0,   0,   200) if estado == 'rojo'     else (0,  0,  50)
    c_amarillo = (0,  200,  220) if estado == 'amarillo' else (0, 50,  60)
    c_verde    = (0,  200,   80) if estado == 'verde'    else (0, 50,  20)

    cv2.circle(frame, (sx + 18, sy + 20),  12, c_rojo,     -1)
    cv2.circle(frame, (sx + 18, sy + 50),  12, c_amarillo, -1)
    cv2.circle(frame, (sx + 18, sy + 80),  12, c_verde,    -1)


def draw_alerta_intruso(frame, parpadeo_visible):
    """Borde rojo parpadeante + banner de advertencia."""
    h, w = frame.shape[:2]
    if parpadeo_visible:
        # Borde rojo grueso alrededor de toda la pantalla
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
        cv2.rectangle(frame, (6, 6), (w - 7, h - 7), (0, 0, 180), 2)

        # Banner superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        cv2.putText(frame, "! ACCESO NO AUTORIZADO — INTRUSO DETECTADO !",
                    (w // 2 - 310, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

def tiene_lentes(frame, x, y, w, h):
    zona = frame[y + int(h*0.20) : y + int(h*0.55), x : x + w]
    if zona.size == 0:
        return False
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    ojos = eye_cascade.detectMultiScale(gray, 1.1, 8, minSize=(20, 20))
    return len(ojos) >= 2

entrenar_reconocedor()
# ── Iniciar Arduino ───────────────────────────────────────────────────────────
conectar_arduino()
if arduino_ok:
    threading.Thread(target=hilo_arduino, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

print("🎥 Cámara activa.")
print("   R = Registrar nuevo empleado")
print("   Q = Salir")

mensaje       = ""
mensaje_timer = 0
ya_saludados  = set()
panel_info    = None   # {emp_info, confianza, lentes, timer}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    hay_desconocido = False

    for (x, y, w, h) in faces:
        nombre    = "Desconocido"
        color     = COLOR_DESCONOCIDO
        confianza = 0
        lentes    = tiene_lentes(frame, x, y, w, h)

        if entrenado:
            rostro_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            emp_id, dist = reconocedor.predict(rostro_gray)
            confianza = round(100 - dist, 1)
            if dist < 80 and emp_id in labels:
                nombre = labels[emp_id]
                color  = COLOR_CONOCIDO
                if nombre not in ya_saludados:
                    ya_saludados.add(nombre)
                    actualizar_entrada(nombre)
                    registrar_log(nombre, confianza)
                    emp_info = get_emp_info(nombre)
                    panel_info = {"emp": emp_info, "confianza": confianza,
                                  "lentes": lentes, "timer": 180}
                    historial.append({
                        "nombre":    nombre,
                        "hora":      datetime.now().strftime("%H:%M"),
                        "confianza": confianza,
                    })
                    mensaje       = f"Bienvenido {nombre}!  {datetime.now().strftime('%H:%M:%S')}  ({confianza}%)"
                    mensaje_timer = 150
                    print(f"👤 {nombre} reconocido — {confianza}% — {datetime.now().strftime('%H:%M:%S')}")
            else:
                # Sigue siendo desconocido
                hay_desconocido = True

        elif len(faces) > 0:
            # Aún no hay modelo entrenado pero hay cara
            hay_desconocido = True

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        draw_label(frame, nombre, x, y, color)
        draw_barra_confianza(frame, x, y + h, w, int(confianza), color)

        if lentes:
            ey1 = y + int(h * 0.20)
            ey2 = y + int(h * 0.55)
            cv2.rectangle(frame, (x, ey1), (x+w, ey2), COLOR_LENTES, 1)
            draw_label(frame, "Lentes", x, ey2 + 20, COLOR_LENTES)

    # ── Lógica contador intruso ───────────────────────────────────────────────
    if hay_desconocido:
        desconocido_frames += 1
        if desconocido_frames >= FRAMES_ALERTA and not alerta_activa:
            alerta_activa = True
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{ts} | {'⚠ INTRUSO DETECTADO':<20} | confianza: N/A\n")
            print(f"🚨 ALERTA: Intruso detectado a las {datetime.now().strftime('%H:%M:%S')}")
    else:
        # Resetear cuando no hay desconocido en pantalla
        desconocido_frames = 0
        alerta_activa      = False

    # ── Parpadeo ──────────────────────────────────────────────────────────────
    if alerta_activa:
        parpadeo_contador += 1
        if parpadeo_contador >= 15:
            parpadeo_visible  = not parpadeo_visible
            parpadeo_contador = 0
        draw_alerta_intruso(frame, parpadeo_visible)

    # ── Semáforo ──────────────────────────────────────────────────────────────
    # ── Semáforo ──────────────────────────────────────────────────────────────
    if alerta_activa:
        estado_semaforo = 'rojo'

    elif panel_info and panel_info["timer"] > 0:
    # ✅ Empleado reconocido → VERDE  (esta condición debe ir ANTES que amarillo)
        estado_semaforo = 'verde'

    elif entrenado and len(faces) > 0:
    # 🔄 Hay cara pero aún procesando → AMARILLO (transitorio)
        estado_semaforo = 'amarillo'

    else:
    # 🔴 Sin nadie, o cara no reconocida → ROJO
        estado_semaforo = 'rojo'

    draw_semaforo(frame, estado_semaforo)

    if alerta_activa and parpadeo_contador == 0:          # ← AGREGAR
        emitir_sonido('rojo')                             # ← AGREGAR
    elif estado_semaforo == 'verde' and panel_info and panel_info["timer"] == 179:
        emitir_sonido('verde')  

    # ── Panel empleado reconocido ─────────────────────────────────────────────
    if panel_info and panel_info["timer"] > 0:
        draw_panel_empleado(frame, panel_info["emp"],
                            panel_info["confianza"], panel_info["lentes"])
        panel_info["timer"] -= 1

    # ── Historial ─────────────────────────────────────────────────────────────
    if historial:
        draw_historial(frame, historial)

    # ── HUD general ──────────────────────────────────────────────────────────
    cv2.putText(frame, f"Rostros: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%d/%m/%Y  %H:%M:%S"),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "R=Registrar  Q=Salir", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    if mensaje and mensaje_timer > 0:
        cv2.putText(frame, mensaje, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 180), 2)
        mensaje_timer -= 1

    cv2.imshow("EMPRESARIOTEC S.A. — Control de Acceso", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        print("📸 Coloca al nuevo empleado frente a la cámara...")
        print("   Presiona ESPACIO para capturar su foto")
        while True:
            ret2, frame2 = cap.read()
            if not ret2:
                break
            gray2  = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            faces2 = face_cascade.detectMultiScale(gray2, 1.1, 5, minSize=(80, 80))
            cv2.putText(frame2, "Posicionate y presiona ESPACIO", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            for (x2, y2, w2, h2) in faces2:
                cv2.rectangle(frame2, (x2, y2), (x2+w2, y2+h2), (0, 255, 255), 2)
            cv2.imshow("EMPRESARIOTEC S.A. — Control de Acceso", frame2)
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord(" ") and len(faces2) > 0:
                nombre = registrar_empleado(frame2, *faces2[0],
                                            tiene_lentes(frame2, *faces2[0]))
                mensaje       = f"✓ {nombre} registrado!"
                mensaje_timer = 120
                break
            elif key2 == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()