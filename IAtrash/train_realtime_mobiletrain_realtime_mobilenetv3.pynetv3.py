import os, sys, time, math, random, shutil
from pathlib import Path
import numpy as np

# ---------------------------------------------
# 0) CONFIGURACIÓN
# ---------------------------------------------
DATA_DIR = r"DatasetResiduos_flat"   # raíz de tu dataset (dos clases o con splits)
OUTPUT_DIR = r"outputs_residuos_cls" # carpeta de resultados
IMG_SIZE = (224, 224)
BATCH = 32
INIT_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
MODEL_NAME = "mobilenetv3_small"  # 'mobilenetv3_small' o 'mobilenetv3_large'

# ---------------------------------------------
# 1) TENSORFLOW + GPU
# ---------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# GPU y mixed precision
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print(f"[INFO] GPU detectada: {gpus}. Mixed precision ACTIVADA.")
    else:
        print("[WARN] No se detectó GPU, se usará CPU.")
except Exception as e:
    print("[WARN] Config GPU:", e)

# ---------------------------------------------
# 2) UTILIDADES
# ---------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import csv

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def count_files_per_class(root):
    root = Path(root)
    counts = {}
    for c in sorted([d for d in root.iterdir() if d.is_dir()]):
        n = sum(1 for p in c.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        counts[c.name] = n
    return counts

def has_splits(root):
    root = Path(root)
    return (root/"train").exists() and (root/"val").exists()

def make_splits_if_needed(root, val_split=0.15, test_split=0.15):
    root = Path(root)
    if has_splits(root):
        print("[INFO] Se detectaron splits existentes (train/val[/test]).")
        return

    classes = [d for d in root.iterdir() if d.is_dir()]
    assert len(classes) == 2, "Se esperaban exactamente 2 carpetas de clase en la raíz."
    print(f"[INFO] Creando splits train/val/test desde {root} ...")

    tmp = root / "_tmp_flat"
    tmp.mkdir(exist_ok=True)

    # copiar a tmp por clase
    for cdir in classes:
        (tmp/cdir.name).mkdir(parents=True, exist_ok=True)
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                dst = tmp/cdir.name/f"{p.stem}{p.suffix.lower()}"
                shutil.copy2(p, dst)

    # crear carpetas split
    for split in ["train","val","test"]:
        for cdir in classes:
            (root/split/cdir.name).mkdir(parents=True, exist_ok=True)

    # repartir
    rng = random.Random(SEED)
    for cdir in classes:
        files = [p for p in (tmp/cdir.name).glob("*") if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        n = len(files)
        nv = int(n*val_split)
        nt = int(n*test_split)
        train_files = files[nv+nt:]
        val_files = files[:nv]
        test_files = files[nv:nv+nt]

        for lst, split in [(train_files,"train"),(val_files,"val"),(test_files,"test")]:
            for src in lst:
                dst = root/split/cdir.name/src.name
                shutil.copy2(src, dst)

    shutil.rmtree(tmp, ignore_errors=True)
    print("[OK] Splits creados.")

def class_weights_from_counts(counts_dict):
    classes = sorted(list(counts_dict.keys()))
    counts = np.array([counts_dict[c] for c in classes], dtype=np.float32)
    total = counts.sum()
    weights = total / (len(classes) * counts)
    return {i: float(w) for i, w in enumerate(weights)}, classes

def plot_and_save_history(hist_head, hist_ft, outdir):
    """
    Une historias (head + finetune) y guarda gráficas de loss/acc/auc.
    """
    def _concat_metrics(h1, h2, key):
        a = h1.history.get(key, [])
        b = h2.history.get(key, [])
        return a + b

    metrics = ["loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"]
    hist = {m: _concat_metrics(hist_head, hist_ft, m) for m in metrics}

    # Guardar CSV de historia
    csv_path = Path(outdir) / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metrics)
        for i in range(len(hist["loss"])):
            row = [i+1] + [hist[m][i] if i < len(hist[m]) else "" for m in metrics]
            writer.writerow(row)
    print(f"[OK] Historial guardado en {csv_path}")

    # Gráfica: Loss
    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir)/"plot_loss.png", dpi=200)
    plt.close()

    # Gráfica: Accuracy
    if len(hist["accuracy"]) > 0:
        plt.figure()
        plt.plot(hist["accuracy"], label="train_acc")
        plt.plot(hist["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training & Validation Accuracy")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(outdir)/"plot_accuracy.png", dpi=200)
        plt.close()

    # Gráfica: AUC
    if len(hist["auc"]) > 0:
        plt.figure()
        plt.plot(hist["auc"], label="train_auc")
        plt.plot(hist["val_auc"], label="val_auc")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("Training & Validation AUC")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(outdir)/"plot_auc.png", dpi=200)
        plt.close()

def evaluate_and_plots(model, test_ds, class_names, outdir):
    """
    Evalúa y guarda: matriz de confusión, ROC y reporte de clasificación.
    """
    if test_ds is None:
        print("[INFO] No hay test/ No se generarán gráficas de evaluación.")
        return

    # Recolectar y_true y y_score
    y_true = []
    y_score = []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0).ravel()
        y_score.extend(probs.tolist())
        y_true.extend(yb.numpy().tolist())

    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(Path(outdir)/"confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir)/"roc_curve.png", dpi=200)
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (Path(outdir)/"classification_report.txt").write_text(report, encoding="utf-8")
    print("[OK] Reporte de clasificación guardado.")
    print(report)

def build_datasets(data_dir, img_size, batch, seed):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from pathlib import Path

    data_dir = Path(data_dir)
    make_splits_if_needed(data_dir)  # crea train/val/test si no existen

    # Augmentación
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(factor=0.1),
    ], name="augmentation")

    # --- Helper: crea ds y devuelve también class_names antes de map/prefetch ---
    def make_ds(sub):
        raw_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir/sub,
            labels="inferred",
            label_mode="int",
            class_names=None,         # detecta segun carpetas
            image_size=img_size,
            batch_size=batch,
            shuffle=True,
            seed=seed
        )
        class_names = raw_ds.class_names[:]  # <-- ¡guardar ANTES de map()!
        ds = raw_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
        if sub == "train":
            ds = ds.map(lambda x, y: (aug(x, training=True), y))
        ds = ds.cache().prefetch(tf.data.AUTOTUNE)
        return ds, class_names

    # Construir datasets y capturar class_names del TRAIN
    train_ds, class_names = make_ds("train")
    val_ds, _ = make_ds("val")
    test_ds = None
    if (data_dir / "test").exists():
        test_ds, _ = make_ds("test")

    # --- Pesos de clase en el orden EXACTO de class_names ---
    counts = count_files_per_class(data_dir / "train")  # {'No_reutilizables': N, 'Reutilizables': M}
    total = sum(counts.get(c, 0) for c in class_names)
    num_classes = len(class_names)
    # Formula clásica: w_c = total / (num_classes * count_c)
    class_weights = {}
    for i, cname in enumerate(class_names):
        n = counts.get(cname, 1)  # evitar div/0 si alguna clase está vacía por error
        class_weights[i] = float(total / (num_classes * n))

    print("[INFO] Clases detectadas:", class_names)
    print("[INFO] Pesos de clase:", class_weights)
    return train_ds, val_ds, test_ds, class_names, class_weights


# ---------------------------------------------
# 4) MODELO
# ---------------------------------------------
def build_model(model_name, img_size, num_classes=2):
    inputs = layers.Input(shape=img_size+(3,))
    x = inputs  # ya normalizamos a [0,1] antes

    if model_name == "mobilenetv3_small":
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet", input_shape=img_size+(3,))
    else:
        base = tf.keras.applications.MobileNetV3Large(
            include_top=False, weights="imagenet", input_shape=img_size+(3,))

    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)  # binary

    model = keras.Model(inputs, outputs, name=model_name+"_reutilizable_cls")
    return model, base

# ---------------------------------------------
# 5) ENTRENAMIENTO
# ---------------------------------------------
def train():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, class_weights = build_datasets(
        DATA_DIR, IMG_SIZE, BATCH, SEED
    )

    model, base = build_model(MODEL_NAME, IMG_SIZE, num_classes=2)
    model.compile(
        optimizer=keras.optimizers.Adam(INIT_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(OUTPUT_DIR)/"best_head.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                       patience=3, restore_best_weights=True)
    rlrop = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1)

    print("[INFO] Entrenando cabeza (backbone congelado)...")
    hist_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=[ckpt, es, rlrop],
        verbose=1
    )

    # -------- Fine-tuning ----------
    print("[INFO] Fine-tuning: descongelando bloques finales...")
    base.trainable = True
    for layer in base.layers[:int(0.7*len(base.layers))]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    ckpt_ft = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(OUTPUT_DIR)/"best_finetune.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1)

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=[ckpt_ft, es, rlrop],
        verbose=1
    )

    # Guardar gráficos de entrenamiento
    plot_and_save_history(hist_head, hist_ft, OUTPUT_DIR)

    # Evaluación
    best_path = Path(OUTPUT_DIR)/"best_finetune.keras"
    if not best_path.exists():
        best_path = Path(OUTPUT_DIR)/"best_head.keras"
    best_model = keras.models.load_model(best_path)
    if test_ds is not None:
        print("[INFO] Evaluando en test...")
        eval_res = best_model.evaluate(test_ds, verbose=1)
        print("[TEST] loss, acc, auc =", eval_res)
        # Guardar métricas test en txt
        with open(Path(OUTPUT_DIR)/"test_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"loss={eval_res[0]:.6f}\naccuracy={eval_res[1]:.6f}\nauc={eval_res[2]:.6f}\n")

        # Plots de evaluación
        evaluate_and_plots(best_model, test_ds, class_names, OUTPUT_DIR)

    # Guardar modelos
    best_model.save(Path(OUTPUT_DIR)/"model_final.keras")
    best_model.save(Path(OUTPUT_DIR)/"model_final.h5")
    print("[OK] Modelo guardado.")

    # Exportar TFLite (float16 para velocidad)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    (Path(OUTPUT_DIR)/"model_float16.tflite").write_bytes(tflite_model)
    print("[OK] Exportado TFLite:", Path(OUTPUT_DIR)/"model_float16.tflite")

    # Guardar nombres de clase
    (Path(OUTPUT_DIR)/"class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print("[INFO] Clases:", class_names)
    print("[DONE] Entrenamiento completo.")

# ---------------------------------------------
# 6) INFERENCIA EN WEBCAM
# ---------------------------------------------
def webcam_demo():
    """
    Usa el mejor modelo guardado. Muestra etiqueta y probabilidad.
    Filtro de estabilidad: promedio móvil de N predicciones.
    """
    import cv2
    model_path = Path(OUTPUT_DIR)/"model_final.keras"
    if not model_path.exists():
        model_path = Path(OUTPUT_DIR)/"best_finetune.keras"
    if not model_path.exists():
        model_path = Path(OUTPUT_DIR)/"best_head.keras"
    if not model_path.exists():
        raise FileNotFoundError("No encontré un modelo entrenado en OUTPUT_DIR.")

    model = keras.models.load_model(model_path)
    class_names = (Path(OUTPUT_DIR)/"class_names.txt").read_text(encoding="utf-8").splitlines()
    class_names = sorted(class_names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    from collections import deque
    N = 8
    buf = deque(maxlen=N)

    print("[INFO] Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = tf.image.resize(frame, IMG_SIZE).numpy()
        x = img[..., ::-1]  # BGR->RGB
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)

        prob = float(model.predict(x, verbose=0)[0][0])
        buf.append(prob)
        smoothed = float(np.mean(buf))

        cls_idx = 1 if smoothed >= 0.5 else 0
        label = class_names[cls_idx] if len(class_names) == 2 else ("Reutilizables" if cls_idx==1 else "No_reutilizables")
        p = smoothed if cls_idx==1 else (1.0-smoothed)
        text = f"{label}: {p*100:.1f}%"

        color = (0,255,0) if cls_idx==1 else (0,0,255)
        import cv2
        cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.rectangle(frame, (15,55), (int(15+300*p), 80), color, -1)
        cv2.rectangle(frame, (15,55), (315,80), (255,255,255), 2)

        cv2.imshow("Clasificacion Reutilizable/No", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------
# 7) MAIN
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","webcam"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    #else:
        #webcam_demo()
