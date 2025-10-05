import os, random, shutil, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR     = r"DatasetResiduos_flat"   # Raíz del dataset (si no hay split, se crea)
OUTPUT_DIR   = r"outputs_residuos_cls"   # Carpeta de resultados
IMG_SIZE     = (224, 224)
BATCH        = 32
INIT_LR      = 1e-4
FINE_LR      = 1e-5
EPOCHS_HEAD  = 10
EPOCHS_FINE  = 10
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
SEED         = 42
BACKBONE     = "mobilenetv3_small"  # 'mobilenetv3_small' | 'mobilenetv3_large'

# Opciones avanzadas
USE_BALANCED_TRAIN = True   # Oversampling por batch (mezcla 50/50)
USE_FOCAL_LOSS     = False  # Focal loss para desbalance (si False -> BCE)

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

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print(f"[INFO] GPU detectada: {gpus}. Mixed precision ACTIVADA.")
    else:
        print("[WARN] No se detectó GPU, usando CPU.")
except Exception as e:
    print("[WARN] Config GPU:", e)

# =========================
# UTILS
# =========================
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score

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
    assert len(classes) == 2, "Se esperaban exactamente 2 carpetas de clase en la raíz."
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

    # Repartir
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
    # w_c = total / (num_classes * count_c) en el orden de class_names
    total = sum(counts_dict.get(c, 0) for c in class_names)
    ncls = len(class_names)
    weights = {}
    for i, cname in enumerate(class_names):
        cnt = counts_dict.get(cname, 1)
        weights[i] = float(total / (ncls * cnt))
    return weights

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w  = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -w * tf.pow((1 - pt), gamma) * tf.math.log(pt)
    return loss

# =========================
# DATASETS
# =========================
def build_datasets(data_dir, img_size, batch, seed):
    data_dir = Path(data_dir)
    make_splits_if_needed(data_dir)

    # Augmentación
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(factor=0.1),
    ], name="augmentation")

    def make_base_ds(subdir):
        raw_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir/subdir,
            labels="inferred",
            label_mode="int",
            class_names=None,      # detecta según carpetas
            image_size=img_size,
            batch_size=batch,
            shuffle=True,
            seed=seed
        )
        class_names = raw_ds.class_names[:]  # guardar antes de map/prefetch
        ds = raw_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
        if subdir == "train" and not USE_BALANCED_TRAIN:
            ds = ds.map(lambda x, y: (aug(x, training=True), y))
        ds = ds.cache().prefetch(tf.data.AUTOTUNE)
        return ds, class_names

    # train/val/test
    train_ds, class_names = make_base_ds("train")
    val_ds, _   = make_base_ds("val")
    test_ds     = None
    if (data_dir/"test").exists():
        test_ds, _ = make_base_ds("test")

    # Pesos de clase (orden exacto de class_names)
    counts = count_files_per_class(data_dir/"train")
    class_weights = class_weights_from_counts(counts, class_names)

    # Balanced train (oversampling por batch) si se desea
    if USE_BALANCED_TRAIN:
        # crear un ds por clase a partir de las carpetas de train
        train_root = data_dir/"train"
        cls_dirs = [d for d in train_root.iterdir() if d.is_dir()]
        # Para orden consistente, reordenar según class_names
        name_to_dir = {d.name: d for d in cls_dirs}
        per_class_ds = []
        for cname in class_names:
            cdir = name_to_dir[cname]
            raw = tf.keras.utils.image_dataset_from_directory(
                cdir, labels="inferred", label_mode="int",
                class_names=[cname],  # etiqueta única
                image_size=img_size, batch_size=batch,
                shuffle=True, seed=seed
            )
            dsi = raw.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))
            # augment SOLO en train balanceado
            dsi = dsi.map(lambda x,y: (aug(x, training=True), y))
            per_class_ds.append(dsi)

        # Mezclar con pesos iguales
        train_ds = tf.data.Dataset.sample_from_datasets(
            per_class_ds, weights=[1.0/len(per_class_ds)]*len(per_class_ds), seed=seed
        ).prefetch(tf.data.AUTOTUNE)

    print("[INFO] Clases:", class_names)
    print("[INFO] Pesos de clase:", class_weights)
    return train_ds, val_ds, test_ds, class_names, class_weights

# =========================
# MODELO
# =========================
def build_model(backbone_name, img_size):
    inputs = layers.Input(shape=img_size+(3,))
    x = inputs  # ya normalizado en el pipeline

    if backbone_name == "mobilenetv3_small":
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
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name=backbone_name+"_reutilizable_cls")
    return model, base

# =========================
# PLOTS & EVAL
# =========================
def save_history_plots(hist_head, hist_ft, outdir):
    def cat(key):
        a = hist_head.history.get(key, [])
        b = hist_ft.history.get(key, [])
        return a + b

    metrics = ["loss","accuracy","auc","val_loss","val_accuracy","val_auc"]
    hist = {m: cat(m) for m in metrics}

    # CSV
    csv_path = Path(outdir)/"training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + metrics)
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

    # AUC
    if len(hist["auc"])>0:
        plt.figure()
        plt.plot(hist["auc"], label="train_auc")
        plt.plot(hist["val_auc"], label="val_auc")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("Training & Validation AUC")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(Path(outdir)/"plot_auc.png", dpi=200); plt.close()

def eval_with_threshold(model, ds, class_names, outdir, thr):
    if ds is None:
        print("[INFO] No hay test: omito evaluación y plots.")
        return
    # y_true / y_score
    y_true, y_score = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0).ravel()
        y_score.extend(p.tolist()); y_true.extend(yb.numpy().tolist())
    y_true  = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred  = (y_score >= thr).astype(int)

    # CM
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j],'d'),
                     ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(Path(outdir)/"confusion_matrix.png", dpi=200); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(outdir)/"roc_curve.png", dpi=200); plt.close()

    # Reporte
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (Path(outdir)/"classification_report.txt").write_text(rep, encoding="utf-8")
    print("[OK] classification_report.txt\n", rep)

    # Métricas simples
    loss, acc, auc_metric = model.evaluate(ds, verbose=1)
    with open(Path(outdir)/"test_metrics.txt","w",encoding="utf-8") as f:
        f.write(f"loss={loss:.6f}\naccuracy={acc:.6f}\nauc={auc_metric:.6f}\nthr={thr:.4f}\n")

def best_threshold_from_val(model, val_ds, outdir):
    y_true, y_score = [], []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0).ravel()
        y_score.extend(p.tolist()); y_true.extend(yb.numpy().tolist())
    y_true  = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    (Path(outdir)/"best_threshold.txt").write_text(f"{best_t:.4f}")
    print(f"[INFO] Umbral óptimo F1 (VAL) = {best_t:.4f}")
    return best_t

# =========================
# TRAIN
# =========================
def train():
    ensure_dir(OUTPUT_DIR)
    train_ds, val_ds, test_ds, class_names, class_weights = build_datasets(
        DATA_DIR, IMG_SIZE, BATCH, SEED
    )

    model, base = build_model(BACKBONE, IMG_SIZE)

    # Loss
    loss_fn = binary_focal_loss(gamma=2.0, alpha=0.25) if USE_FOCAL_LOSS else "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(INIT_LR),
        loss=loss_fn,
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
        class_weight=None if USE_BALANCED_TRAIN else class_weights,
        callbacks=[ckpt, es, rlrop],
        verbose=1
    )

    # Fine-tuning
    print("[INFO] Fine-tuning: descongelando bloques finales...")
    base.trainable = True
    for layer in base.layers[:int(0.7*len(base.layers))]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_LR),
        loss=loss_fn,
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    ckpt_ft = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(OUTPUT_DIR)/"best_finetune.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1)

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=None if USE_BALANCED_TRAIN else class_weights,
        callbacks=[ckpt_ft, es, rlrop],
        verbose=1
    )

    # Plots entrenamiento
    save_history_plots(hist_head, hist_ft, OUTPUT_DIR)

    # Modelo final (elige el mejor)
    best_path = Path(OUTPUT_DIR)/"best_finetune.keras"
    if not best_path.exists():
        best_path = Path(OUTPUT_DIR)/"best_head.keras"
    best_model = keras.models.load_model(best_path, compile=False)
    best_model.compile(
        optimizer=keras.optimizers.Adam(FINE_LR),
        loss=loss_fn,
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    # Umbral óptimo con VALIDACIÓN
    best_thr = best_threshold_from_val(best_model, val_ds, OUTPUT_DIR)

    # Evaluación en TEST con best_thr
    eval_with_threshold(best_model, test_ds, class_names, OUTPUT_DIR, best_thr)

    # Guardados
    best_model.save(Path(OUTPUT_DIR)/"model_final.keras")
    best_model.save(Path(OUTPUT_DIR)/"model_final.h5")
    print("[OK] Modelos guardados.")

    # Export TFLite (float16)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    (Path(OUTPUT_DIR)/"model_float16.tflite").write_bytes(tflite_model)
    print("[OK] TFLite ->", Path(OUTPUT_DIR)/"model_float16.tflite")

    (Path(OUTPUT_DIR)/"class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print("[INFO] Clases:", class_names)
    print("[DONE] Entrenamiento completo.")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()

