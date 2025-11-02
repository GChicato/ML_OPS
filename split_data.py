import os, shutil, random
from pathlib import Path
from typing import List

# --- ADAPTE ce chemin à ton Mac (mets-le EXACT) ---
SRC = Path("/Users/guillaumeletosser/Desktop/Albert school/mlops/processed_data")
# ---------------------------------------------------

DST = Path("data")  # on va générer data/train, data/val, data/test
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
assert abs(sum(SPLITS.values()) - 1.0) < 1e-6

def main():
    assert SRC.exists(), f"Introuvable: {SRC}"
    classes = [d.name for d in SRC.iterdir() if d.is_dir()]
    print("Classes:", classes)

    for split in SPLITS:
        for cls in classes:
            (DST / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    for cls in classes:
        files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files += list((SRC/cls).glob(ext))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        parts = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:],
        }
        for split, items in parts.items():
            out = DST / split / cls
            for p in items:
                shutil.copy2(p, out / p.name)

        print(f"{cls}: {n} -> train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}")

if __name__ == "__main__":
    main()
