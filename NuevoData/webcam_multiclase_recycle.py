# webcam_multiclase_recycle.py
# Requisitos: pip install opencv-python tensorflow numpy

import os
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG
# =========================
MODEL_DIR = Path("outputs_multiclase_mnv2")  # ajusta si usaste otra carpeta
MODEL_PATH = MODEL_DIR / "model_final.keras" # o "best_finetune.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

CAM_INDEX = 0             # índice de la webcam
IMG_SIZE = (224, 224)     # usa el mismo tamaño que entrenaste
SMOOTH_N = 8              # frames para promedio móvil de probabilidades
SHOW_TOPK = 3             # muestra top-k en panel
MIN_CONF_TO_DRAW = 0.30   # umbral mínimo para mostrar etiqueta principal

# Colores (BGR)
GREEN = (0, 200, 0)
RED   = (0, 0, 220)
YELLOW= (0, 215, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (40, 40, 40)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================
# MAPEO: clase -> categoría de reciclaje
# =========================
REUSABLE = {
    "aluminum_food_cans",
    "aluminum_soda_cans",
    "steel_food_cans",
    "glass_beverage_bottles",
    "glass_cosmetic_containers",
    "glass_food_jars",
    "cardboard_boxes",
    "cardboard_packaging",
    "newspaper",
    "office_paper",
    "magazines",
    "clothing",
    "shoes",
    "plastic_detergent_bottles",
    "plastic_food_containers",
    "plastic_soda_bottles",
    "plastic_water_bottles",
    # condicional se maneja aparte
}

NON_REUSABLE = {
    "paper_cups",
    "tea_bags",
    "disposable_plastic_cutlery",
    "plastic_shopping_bags",
    "plastic_straws",
    "plastic_trash_bags",
    "styrofoam_cups",
    "styrofoam_food_containers",
    "coffee_grounds",
    "eggshells",
    "food_waste",
    "aerosol_cans",
}

CONDITIONAL = {"plastic_cup_lids"}  # reciclable condicional (PET/PP)

# =========================
# UTILIDADES
# =========================
def load_class_names(path: Path):
    names = path.read_text(encoding="utf-8").splitlines()
    # Quitar líneas vacías y espacios
    names = [n.strip() for n in names if n.strip()]
    return names

def draw_filled_rect(img, pt1, pt2, color, alpha=0.6):
    """Rectángulo sólido translúcido."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def label_for_class(cls: str):
    """Devuelve (texto, color) para leyenda de reciclaje según clase."""
    if cls in REUSABLE:
        return "♻️ Reutilizable", GREEN
    if cls in NON_REUSABLE:
        return "🚯 No reutilizable", RED
    if cls in CONDITIONAL:
        return "♻️ Reutilizable (condicional)", YELLOW
    # fallback si apareciera una clase no mapeada
    return "¿Sin categoría?", GRAY

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)

# =========================
# CARGA MODELO
# =========================
def load_keras_model(model_path: Path):
    model = keras.models.load_model(model_path)
    return model

# --- Si prefieres TFLite, descomenta este bloque y usa predict_tflite() abajo ---
# def load_tflite_model(tflite_path: Path):
#    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
#    interpreter.allocate_tensors()
#    input_details = interpreter.get_input_details()
#    output_details = interpreter.get_output_details()
#    return interpreter, input_details, output_details

# def predict_tflite(interpreter, input_details, output_details, x):
#    interpreter.set_tensor(input_details[0]['index'], x.astype(np.float32))
#    interpreter.invoke()
#    out = interpreter.get_tensor(output_details[0]['index'])
#    return out  # (1, C)

# =========================
# MAIN
# =========================
def main():
    # Cargar clases
    class_names = load_class_names(CLASS_NAMES_PATH)
    num_classes = len(class_names)
    print("[INFO] Clases:", class_names)

    # Cargar Keras
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    model = load_keras_model(MODEL_PATH)
    print("[OK] Modelo cargado:", MODEL_PATH.name)

    # # Opción TFLite:
    # tflite_path = MODEL_DIR / "model_float16.tflite"
    # interpreter, in_details, out_details = None, None, None
    # if tflite_path.exists():
    #     interpreter, in_details, out_details = load_tflite_model(tflite_path)
    #     print("[OK] TFLite cargado:", tflite_path.name)

    # Buffer para suavizar (promedio móvil de probabilidades)
    prob_buf = deque(maxlen=SMOOTH_N)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    print("[INFO] Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # (Opcional) espejo horizontal para UX
        frame = cv2.flip(frame, 1)

        # Preprocess
        img = cv2.resize(frame, IMG_SIZE)
        x = img[..., ::-1]  # BGR->RGB
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)  # (1, H, W, 3)

        # Predicción
        # if interpreter is not None:
        #     probs = predict_tflite(interpreter, in_details, out_details, x)[0]
        # else:
        probs = model.predict(x, verbose=0)[0]  # (C,)
        # Garantiza que es distribución (por si el modelo ya devuelve softmax)
        if probs.ndim == 1:
            s = probs.sum()
            probs = probs if 0.99 < s < 1.01 else softmax(probs)

        # Suavizado
        prob_buf.append(probs)
        probs_smooth = np.mean(prob_buf, axis=0)

        # Top-k
        top_idx = np.argsort(probs_smooth)[::-1][:max(1, SHOW_TOPK)]
        top_probs = probs_smooth[top_idx]

        # Etiqueta principal
        cls_idx = int(top_idx[0])
        cls_name = class_names[cls_idx]
        conf = float(top_probs[0])
        label_recycle, color_recycle = label_for_class(cls_name)

        # --- Dibujo overlay ---
        h, w = frame.shape[:2]
        # Panel de texto
        panel_w = 360
        panel_h = 120 + 26 * (SHOW_TOPK - 1)
        x0, y0 = 12, 12
        frame = draw_filled_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), alpha=0.55)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 2)

        # Línea 1: Clase y confianza
        main_text = f"{cls_name}  |  {conf*100:.1f}%"
        cv2.putText(frame, main_text, (x0 + 16, y0 + 38), FONT, 0.8, WHITE if conf >= MIN_CONF_TO_DRAW else (180,180,180), 2)

        # Línea 2: Leyenda de reciclaje
        cv2.putText(frame, label_recycle, (x0 + 16, y0 + 72), FONT, 0.8, color_recycle, 2)

        # Top-k adicional
        if SHOW_TOPK > 1:
            cv2.putText(frame, "Top-k:", (x0 + 16, y0 + 100), FONT, 0.6, (200,200,200), 1)
            for i in range(1, SHOW_TOPK):
                if i >= len(top_idx): break
                k_cls = class_names[int(top_idx[i])]
                k_p = float(top_probs[i])
                cv2.putText(frame, f"{i+1}. {k_cls}  {k_p*100:.1f}%",
                            (x0 + 16, y0 + 100 + 26*i), FONT, 0.55, (210,210,210), 1)

        # Barra de confianza principal
        bar_x1, bar_y1 = x0 + 16, y0 + panel_h - 22
        bar_x2 = x0 + panel_w - 16
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + 12), (80, 80, 80), 1)
        fill_w = int((bar_x2 - bar_x1 - 2) * max(0.0, min(1.0, conf)))
        cv2.rectangle(frame, (bar_x1 + 1, bar_y1 + 1), (bar_x1 + 1 + fill_w, bar_y1 + 11),
                      color_recycle, -1)

        # Mostrar
        cv2.imshow("Clasificador de Residuos (multiclase)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
