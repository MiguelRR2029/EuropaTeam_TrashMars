import os
import shutil
from pathlib import Path

# =======================
# CONFIG
# =======================
SOURCE_DIR = r"images"               # Carpeta base donde tienes todas las clases
TARGET_DIR = r"DatasetMulticlase"    # Carpeta destino (nueva)
MERGE_FOLDERS = ["default", "real_world"]  # subcarpetas a unir

# =======================
# FUNCIÓN PRINCIPAL
# =======================
def merge_subfolders(source_dir, target_dir, merge_folders):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"[INFO] Se encontraron {len(class_dirs)} clases.")

    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class_dir = target_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        total_imgs = 0
        for sub in merge_folders:
            subdir = class_dir / sub
            if not subdir.exists():
                continue
            for file in subdir.rglob("*"):
                if file.is_file():
                    dst = out_class_dir / file.name
                    # Evita sobrescribir: añade sufijo si ya existe
                    if dst.exists():
                        base, ext = os.path.splitext(file.name)
                        i = 1
                        while (out_class_dir / f"{base}_{i}{ext}").exists():
                            i += 1
                        dst = out_class_dir / f"{base}_{i}{ext}"
                    shutil.copy2(file, dst)
                    total_imgs += 1

        print(f"[OK] Clase '{class_name}': {total_imgs} imágenes combinadas.")

    print(f"\n[HECHO] Dataset multiclase creado en: {target_dir}")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    merge_subfolders(SOURCE_DIR, TARGET_DIR, MERGE_FOLDERS)
