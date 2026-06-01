"""
UPDATE DATASET — Rename + Merge videos_update vào label_gold.csv

Cấu trúc videos_update:
  videos_update/
  ├── ăn uống/
  │   ├── NTTM_2081672_L.mp4   → rename: NTTM_ăn uống_L.mp4
  │   ├── NTTM_8400355_R.mp4   → rename: NTTM_ăn uống_R.mp4
  │   └── ABC_xxxx_C.mp4       → rename: ABC_ăn uống_C.mp4
  └── bạn/
      └── ...

Sau khi chạy:
  - File được rename đúng format: {PERSON}_{LABEL}_{ANGLE}.mp4
  - Video được copy vào Videos/
  - label_gold.csv được cập nhật thêm các dòng mới
  - Report thống kê in ra màn hình
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Cấu hình ──────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\MYPC\Downloads\Dataset"
VIDEOS_UPDATE   = os.path.join(BASE_DIR, "videos_update")
VIDEOS_DIR      = os.path.join(BASE_DIR, "Videos")
CSV_PATH        = os.path.join(BASE_DIR, "Labels", "label_gold.csv")
CSV_BACKUP      = os.path.join(BASE_DIR, "Labels", "label_gold_backup.csv")

ANGLE_MAP = {"L": "L", "R": "R", "C": "C"}

# ── Đọc CSV hiện tại ───────────────────────────────────────────────────────
df_old = pd.read_csv(CSV_PATH)
df_old.columns = [c.strip().upper() for c in df_old.columns]
shutil.copy(CSV_PATH, CSV_BACKUP)
print(f"✅ Backup: {CSV_BACKUP}")
print(f"📊 label_gold.csv hiện tại: {len(df_old)} dòng\n")

existing_videos = set(df_old["VIDEO"].str.strip().str.lower().tolist())

new_rows    = []
renamed     = []
skipped     = []
errors      = []
label_stats = defaultdict(int)

# ── Duyệt videos_update/{label_folder}/ ───────────────────────────────────
for label_folder in sorted(os.listdir(VIDEOS_UPDATE)):
    folder_path = os.path.join(VIDEOS_UPDATE, label_folder)
    if not os.path.isdir(folder_path):
        continue

    label = label_folder.strip()

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".mp4"):
            continue

        src_path = os.path.join(folder_path, fname)
        parts    = Path(fname).stem.split("_")

        # PERSON = phần đầu, ANGLE = phần cuối nếu là L/R/C
        person = parts[0].upper() if parts else "UNK"
        angle  = "C"
        if len(parts) >= 2 and parts[-1].upper() in ANGLE_MAP:
            angle = ANGLE_MAP[parts[-1].upper()]

        # Tên file mới chuẩn — thay dấu cách bằng _
        label_safe = re.sub(r'[\\/*?:"<>|]', "", label).strip()
        label_safe = label_safe.replace(" ", "_")
        new_fname  = f"{person}_{label_safe}_{angle}.mp4"
        dst_path   = os.path.join(VIDEOS_DIR, new_fname)

        # Check trùng trong CSV
        if new_fname.lower() in existing_videos:
            skipped.append(f"  ⚠ Trùng CSV, bỏ qua: {new_fname}")
            continue

        # Tránh ghi đè file trên disk
        counter = 1
        while os.path.exists(dst_path):
            new_fname = f"{person}_{label_safe}_{angle}_{counter}.mp4"
            dst_path  = os.path.join(VIDEOS_DIR, new_fname)
            counter  += 1

        try:
            shutil.copy2(src_path, dst_path)
            renamed.append(f"  ✅ {fname:45s} → {new_fname}")
            new_rows.append({
                "VIDEO" : new_fname,
                "LABEL" : label,
                "PERSON": person,
                "ANGLE" : angle,
                "SOURCE": "videos_update",
            })
            existing_videos.add(new_fname.lower())
            label_stats[label] += 1
        except Exception as e:
            errors.append(f"  ❌ {fname}: {e}")

# ── Merge & lưu CSV ────────────────────────────────────────────────────────
if new_rows:
    df_new = pd.DataFrame(new_rows)
    for col in ["PERSON", "ANGLE", "SOURCE"]:
        if col not in df_old.columns:
            df_old[col] = ""
    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    df_merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ CSV cập nhật: {len(df_old)} → {len(df_merged)} dòng (+{len(new_rows)})\n")
else:
    df_merged = df_old
    print("⚠ Không có video mới.\n")

# ── Report ─────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"📁 RENAMED & COPIED ({len(renamed)} file):")
for r in renamed:
    print(r)

if skipped:
    print(f"\n⚠ BỎ QUA ({len(skipped)}):")
    for s in skipped:
        print(s)

if errors:
    print(f"\n❌ LỖI ({len(errors)}):")
    for e in errors:
        print(e)

print("\n" + "=" * 60)
print("📊 VIDEO MỚI THEO NHÃN:")
for label, cnt in sorted(label_stats.items()):
    total = len(df_merged[df_merged["LABEL"] == label])
    print(f"  {label:25s}: +{cnt} mới  (tổng hiện tại: {total})")

print("\n" + "=" * 60)
print("📊 THỐNG KÊ DATASET SAU UPDATE:")
print(f"  Tổng nhãn  : {df_merged['LABEL'].nunique()}")
print(f"  Tổng video : {len(df_merged)}")

dist = df_merged.groupby("LABEL").size()
print(f"  Video/nhãn : min={dist.min()}  max={dist.max()}  TB={dist.mean():.1f}")

thin = dist[dist < 4].index.tolist()
if thin:
    print(f"\n  ⚠ Nhãn còn < 4 video ({len(thin)} nhãn) — nên quay thêm:")
    for t in thin:
        print(f"     {t:25s}: {dist[t]} video")

print("\n✅ HOÀN TẤT")