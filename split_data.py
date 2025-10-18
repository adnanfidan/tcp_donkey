import os
import random
import shutil
from pathlib import Path

# ========= AYARLAR =========
SRC_DIR = "/home/prac/Desktop/Thesis/training_beamng_50_data"   # <-- BURAYI KENDİNE GÖRE DOĞRULA
TRAIN_DIR = "/home/prac/Desktop/Thesis/training_beamng_50_data/train_data"
VAL_DIR   = "/home/prac/Desktop/Thesis/training_beamng_50_data/val_data"
VAL_RATIO = 0.2
ONLY_NONEMPTY = True      # True ise içinde dosya olan klasörleri alır
NAME_PATTERN = None       # Örn: "route" ile başlayanlar için: lambda n: n.lower().startswith("route")
DRY_RUN = False            # Önce True ile çalıştır, çıktıyı gör. Sonra False yapıp taşı.

# ========= YARDIMCI =========
def is_nonempty_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # içinde en az 1 dosya ya da alt klasör var mı?
    try:
        next(p.iterdir())
        return True
    except StopIteration:
        return False

# ========= BAŞLA =========
src = Path(SRC_DIR).expanduser().resolve()
train = Path(TRAIN_DIR).expanduser().resolve()
val = Path(VAL_DIR).expanduser().resolve()

print(f"[DEBUG] SRC_DIR:   {src}  (exists={src.exists()})")
print(f"[DEBUG] TRAIN_DIR: {train}")
print(f"[DEBUG] VAL_DIR:   {val}")

if not src.exists():
    raise SystemExit("[HATA] SRC_DIR mevcut değil. Path'i kontrol et.")

# src içeriğini listele
entries = list(src.iterdir())
print(f"[DEBUG] {src} içindeki öğe sayısı: {len(entries)}")
for e in entries[:30]:  # çok uzamasın
    print("   -", ("[DIR]" if e.is_dir() else "[FILE]"), e.name)

# route klasörlerini seç
candidates = [e for e in entries if e.is_dir()]
if NAME_PATTERN:
    candidates = [e for e in candidates if NAME_PATTERN(e.name)]
if ONLY_NONEMPTY:
    candidates = [e for e in candidates if is_nonempty_dir(e)]

print(f"[DEBUG] Bulunan klasör (kriter sonrası): {len(candidates)}")
for e in candidates[:30]:
    print("   -", e.name)

if len(candidates) == 0:
    print("[UYARI] Kriterlere uyan klasör bulunamadı.")
    print("        - SRC_DIR doğru mu?")
    print("        - Klasörlerin isimleri/boş olup olmadıkları?")
    print("        - Daha önce taşınıp SRC_DIR boş kalmış olabilir mi?")
    raise SystemExit(0)

# karıştır ve böl
random.shuffle(candidates)
val_size = max(1, int(len(candidates) * VAL_RATIO)) if len(candidates) > 1 else 1 if VAL_RATIO > 0 else 0
val_routes = candidates[:val_size]
train_routes = candidates[val_size:]

print(f"[INFO] Toplam {len(candidates)} route bulundu.")
print(f"[INFO] Train: {len(train_routes)}  | Val: {len(val_routes)}")

# hedefleri oluştur
train.mkdir(parents=True, exist_ok=True)
val.mkdir(parents=True, exist_ok=True)

# taşıma (veya dry-run)
def move_dir(src_path: Path, dst_root: Path):
    dst_path = dst_root / src_path.name
    if DRY_RUN:
        print(f"[DRY-RUN] move: {src_path} -> {dst_path}")
    else:
        if dst_path.exists():
            # var ise üzerine yazmasını istemiyorsan burada farklı isim türetebilirsin
            shutil.rmtree(dst_path)
        shutil.move(str(src_path), str(dst_path))
        print(f"[OK] {src_path.name} taşındı -> {dst_root}")

for d in train_routes:
    move_dir(d, train)
for d in val_routes:
    move_dir(d, val)

if DRY_RUN:
    print("\n[NOT] DRY_RUN=True idi. Her şey doğru görünüyorsa DRY_RUN=False yapıp tekrar çalıştır.")
else:
    print(f"\n[SONUÇ] Train: {len(train_routes)} klasör -> {train}")
    print(f"[SONUÇ] Val:   {len(val_routes)} klasör -> {val}")
