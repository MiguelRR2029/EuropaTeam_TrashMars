import os, random, shutil, csv, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR     = r"DatasetMulticlase"     # Raíz del dataset (una carpeta por clase)
OUTPUT_DIR   = r"outputs_multiclase_mnv2"
IMG_SIZE     = (224, 224)
BATCH        = 32
LR_HEAD      = 1e-3
LR_FINE      = 3e-4
EPOCHS_HEAD  = 10
EPOCHS_FINE  = 10
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
SEED         = 42
LABEL_SMOOTH = 0.05
UNFREEZE_FRAC = 0.4           # % superior a descongelar en fine-tune
USE_MIXED_PRECISION = False   # pon True si todo corre estable en tu equipo

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

# =========================
# TF + GPU
# =========================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Evita “atascos” con asignación dinámica de VRAM
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gups:
            pass
except:
    pass
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if USE_MIXED_PRECISION:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        print(f"[INFO] GPU detectada: {gpus}. Mixed precision: {USE_MIXED_PRECISION}")
    else:
        print("[WARN] No se detectó GPU; usando CPU.")
except Exception as e:
    print("[WARN] Config GPU:", e)

# =========================
# UTILS
# =========================
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

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
    assert len(classes) >= 2, "Se esperaban >= 2 carpetas de clase en la raíz."
    print(f"[INFO] Creando splits train/val/test desde {root} ...")

    tmp = root / "_tmp_flat"
    ensure_dir(tmp)

    # Copiar a tmp por clase (aplana)
    for cdir in classes:
        ensure_dir(tmp/cdir.name)
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                dst = tmp/cdir.name/f"{p.stem}{p.suffix.lower()}"
                shutil.copy2(p, dst)

    # Crear carpetas split
    for split in ["train","val","test"]:
        for cdir in classes:
            ensure_dir(root/split/cdir.name)

    # Repartir estratificado simple por clase
    rng = random.Random(SEED)
    for cdir in classes:
        files = [p for p in (tmp/cdir.name).glob("*") if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        n = len(files)
        nv = int(n*val_split)
        nt = int(n*test_split)
        train_files = files[nv+nt:]
        val_files   = files[:nv]
        test_files  = files[nv:nv+nt]

        for lst, split in [(train_files,"train"),(val_files,"val"),(test_files,"test")]:
            for src in lst:
                dst = root/split/cdir.name/src.name
                shutil.copy2(src, dst)

    shutil.rmtree(tmp, ignore_errors=True)
    print("[OK] Splits creados.")

def class_weights_from_counts(counts_dict, class_names):
    total = sum(counts_dict.get(c, 0) for c in class_names)
    ncls  = len(class_names)
    weights = {}
    for i, cname in enumerate(class_names):
        cnt = counts_dict.get(cname, 1)
        weights[i] = float(total / (ncls * cnt))
    return weights

def save_history_plots(hist_head, hist_ft, outdir):
    def cat(key):
        a = hist_head.history.get(key, [])
        b = hist_ft.history.get(key, [])
        return a + b

    metrics = ["loss","accuracy","val_loss","val_accuracy"]
    hist = {m: cat(m) for m in metrics}

    # CSV
    csv_path = Path(outdir)/"training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch"] + metrics)
        for i in range(len(hist["loss"])):
            row = [i+1] + [hist[m][i] if i < len(hist[m]) else "" for m in metrics]
            w.writerow(row)
    print(f"[OK] Historial -> {csv_path}")

    # Loss
    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(outdir)/"plot_loss.png", dpi=200); plt.close()

    # Acc
    if len(hist["accuracy"])>0:
        plt.figure()
        plt.plot(hist["accuracy"], label="train_acc")
        plt.plot(hist["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training & Validation Acc")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(Path(outdir)/"plot_accuracy.png", dpi=200); plt.close()

def evaluate_multiclass(model, ds, class_names, outdir):
    if ds is None:
        print("[INFO] No hay test: omito evaluación.")
        return

    # Recolectar logits
    y_true, y_prob = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        y_prob.append(p)          # (B, C)
        y_true.append(yb.numpy()) # (B, C)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # Predicciones finales
    y_pred_idx = np.argmax(y_prob, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)

    # CM
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plt.figure(figsize=(8,6))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=60, ha="right", fontsize=9)
    plt.yticks(ticks, class_names, fontsize=9)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j],'d'),
                     ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black", fontsize=8)
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(Path(outdir)/"confusion_matrix.png", dpi=220); plt.close()

    # Reporte
    report = classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=4)
    (Path(outdir)/"classification_report.txt").write_text(report, encoding="utf-8")
    print("[OK] classification_report.txt\n", report)

    # AUC-OVR macro (si hay >2 clases)
    if len(class_names) > 2:
        try:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            with open(Path(outdir)/"auc_ovr.txt","w",encoding="utf-8") as f:
                f.write(f"AUC-OVR (macro) = {auc_macro:.6f}\n")
            print(f"[OK] AUC-OVR macro = {auc_macro:.4f}")
        except Exception as e:
            print("[WARN] No se pudo calcular AUC-OVR:", e)

    # Métricas Keras directas
    loss, acc = model.evaluate(ds, verbose=1)
    with open(Path(outdir)/"test_metrics.txt","w",encoding="utf-8") as f:
        f.write(f"loss={loss:.6f}\naccuracy={acc:.6f}\n")

# =========================
# DATASETS
# =========================
def build_datasets(data_dir, img_size, batch, seed):
    data_dir = Path(data_dir)
    make_splits_if_needed(data_dir)

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    # train
    raw_train = tf.keras.utils.image_dataset_from_directory(
        data_dir/"train",
        labels="inferred",
        label_mode="categorical",   # multiclase → one-hot
        class_names=None,           # detecta según carpetas
        image_size=img_size,
        batch_size=batch,
        shuffle=True,
        seed=seed
    )
    class_names = raw_train.class_names[:]

    train_ds = raw_train.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))
    train_ds = train_ds.map(lambda x,y: (aug(x, training=True), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # val
    raw_val = tf.keras.utils.image_dataset_from_directory(
        data_dir/"val",
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,    # fija orden
        image_size=img_size,
        batch_size=batch,
        shuffle=False
    )
    val_ds = raw_val.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

    # test (opcional)
    test_ds = None
    if (data_dir/"test").exists():
        raw_test = tf.keras.utils.image_dataset_from_directory(
            data_dir/"test",
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=img_size,
            batch_size=batch,
            shuffle=False
        )
        test_ds = raw_test.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

    # Pesos de clase (orden exacto)
    counts = count_files_per_class(data_dir/"train")
    class_weights = class_weights_from_counts(counts, class_names)

    print("[INFO] Clases:", class_names)
    print("[INFO] Pesos de clase:", class_weights)
    return train_ds, val_ds, test_ds, class_names, class_weights

# =========================
# MODELO (MobileNetV2)
# =========================
def build_model(img_size, num_classes):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=img_size+(3,)
    )
    base.trainable = False

    inputs = layers.Input(shape=img_size+(3,))
    x = inputs  # ya normalizado a [0,1] antes
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_multiclase")
    return model, base

# =========================
# TRAIN
# =========================
def train():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # log simple para ver si “se traba”
    t0 = time.time()

    train_ds, val_ds, test_ds, class_names, class_weights = build_datasets(
        DATA_DIR, IMG_SIZE, BATCH, SEED
    )
    num_classes = len(class_names)

    model, base = build_model(IMG_SIZE, num_classes)
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    # Etapa 1: cabeza
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(OUTPUT_DIR)/"best_head.keras"),
        monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    )
    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=5, restore_best_weights=True
    )
    rlrop = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )

    print("[INFO] Entrenando cabeza (backbone congelado)...")
    hist_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=[ckpt, es, rlrop],
        verbose=1
    )

    # Etapa 2: Fine-tuning
    print("[INFO] Fine-tuning: descongelando la parte superior del backbone...")
    base.trainable = True
    freeze_until = int((1.0 - UNFREEZE_FRAC) * len(base.layers))
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_until)

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    ckpt_ft = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(OUTPUT_DIR)/"best_finetune.keras"),
        monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    )

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=[ckpt_ft, es, rlrop],
        verbose=1
    )

    save_history_plots(hist_head, hist_ft, OUTPUT_DIR)

    # Elige mejor
    best_path = Path(OUTPUT_DIR)/"best_finetune.keras"
    if not best_path.exists():
        best_path = Path(OUTPUT_DIR)/"best_head.keras"
    best_model = keras.models.load_model(best_path)

    # Evaluación en test
    evaluate_multiclass(best_model, test_ds, class_names, OUTPUT_DIR)

    # Guardar modelos
    best_model.save(Path(OUTPUT_DIR)/"model_final.keras")
    best_model.save(Path(OUTPUT_DIR)/"model_final.h5")
    print("[OK] Modelos guardados.")

    # Export TFLite (float16)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        (Path(OUTPUT_DIR)/"model_float16.tflite").write_bytes(tflite_model)
        print("[OK] TFLite ->", Path(OUTPUT_DIR)/"model_float16.tflite")
    except Exception as e:
        print("[WARN] Export TFLite falló:", e)

    # Guardar nombres de clase
    (Path(OUTPUT_DIR)/"class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print("[INFO] Clases:", class_names)

    print(f"[DONE] Tiempo total: {time.time()-t0:.1f}s")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()

