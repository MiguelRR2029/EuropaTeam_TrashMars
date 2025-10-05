import os
import shutil
import hashlib
import random
from pathlib import Path

# ========= CONFIG =========
SOURCE_DIR   = r"DatasetResiduos"        # <- tu dataset actual
TARGET_DIR   = r"DatasetResiduos_flat"   # <- salida con dos clases
MOVE_FILES   = False                     # True = mover; False = copiar
DO_SPLIT     = True                      # True = crea train/val/test; False = solo 2 carpetas
SPLIT_RATIOS = (0.7, 0.15, 0.15)         # train, val, test (solo si DO_SPLIT=True)
SEED         = 42
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# ==========================

random.seed(SEED)

def iter_image_files(root):
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p

def safe_copy_or_move(src, dst, move=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def unique_name(path: Path, prefix: str):
    """Genera nombres únicos usando hash del path completo para evitar colisiones."""
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}{path.suffix.lower()}"

def collect_files_by_class(source_dir):
    """
    Asume estructura:
    DatasetResiduos/
      Reutilizables/
        <muchas subcarpetas>/default | real_world | ...
      No_reutilizables/
        <muchas subcarpetas> ...
    """
    source_dir = Path(source_dir)
    classes = []
    for name in ["Reutilizables", "No_reutilizables"]:
        cdir = source_dir / name
        if cdir.exists():
            classes.append(name)

    if not classes:
        raise RuntimeError("No encontré carpetas 'Reutilizables' o 'No_reutilizables' en SOURCE_DIR.")

    files_by_class = {}
    for cname in classes:
        files = list(iter_image_files(source_dir / cname))
        files_by_class[cname] = files
        print(f"[INFO] {cname}: {len(files)} imágenes encontradas.")
    return files_by_class

def flatten_two_folders(source_dir, target_dir, move=False, do_split=False, split_ratios=(0.7,0.15,0.15)):
    files_by_class = collect_files_by_class(source_dir)
    target_dir = Path(target_dir)

    if do_split:
        tr, va, te = split_ratios
        assert abs((tr+va+te) - 1.0) < 1e-6, "Las proporciones de split deben sumar 1.0"

        for cname, files in files_by_class.items():
            random.shuffle(files)
            n = len(files)
            n_tr = int(n * tr)
            n_va = int(n * va)
            # el resto a test
            splits = {
                "train": files[:n_tr],
                "val": files[n_tr:n_tr+n_va],
                "test": files[n_tr+n_va:]
            }
            for split_name, flist in splits.items():
                for src in flist:
                    out_dir = target_dir / split_name / cname
                    out_name = unique_name(src, prefix=cname.lower())
                    dst = out_dir / out_name
                    safe_copy_or_move(src, dst, move=move)
            print(f"[OK] {cname}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        print(f"\n[HECHO] Dataset aplanado con splits en: {target_dir}")
        print("Estructura ejemplo:")
        print(str(target_dir / "train" / "Reutilizables") + "/<imgs>")
        print(str(target_dir / "val"   / "No_reutilizables") + "/<imgs>")
    else:
        # Solo dos carpetas: Reutilizables/ y No_reutilizables/
        for cname, files in files_by_class.items():
            for src in files:
                out_dir = target_dir / cname
                out_name = unique_name(src, prefix=cname.lower())
                dst = out_dir / out_name
                safe_copy_or_move(src, dst, move=move)
            print(f"[OK] {cname}: {len(files)} imágenes copiadas/movidas.")
        print(f"\n[HECHO] Dataset aplanado en: {target_dir}")
        print("Estructura:")
        print(str(target_dir / "Reutilizables") + "/<imgs>")
        print(str(target_dir / "No_reutilizables") + "/<imgs>")

if __name__ == "__main__":
    flatten_two_folders(
        SOURCE_DIR,
        TARGET_DIR,
        move=MOVE_FILES,
        do_split=DO_SPLIT,
        split_ratios=SPLIT_RATIOS
    )
