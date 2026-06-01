"""
BƯỚC 1 — ĐÁNH GIÁ MODEL & ABLATION STUDY
Chạy sau khi train_vsl_server.py hoàn tất.
Output:
  - confusion_matrix.png
  - per_class_accuracy.csv
  - ablation_results.csv
  - ablation_chart.png
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import mediapipe as mp
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\MYPC\Downloads\Dataset"
MODEL_DIR  = os.path.join(BASE_DIR, "Models")
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
CSV_PATH   = os.path.join(BASE_DIR, "Labels", "label_gold.csv")
EVAL_DIR   = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

N_FRAMES   = 30
N_FEATURES = 225
AUGMENT_COUNT = 30

# ── Load label map ──────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "label_map.json"), encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
label_to_id = {v: k for k, v in id_to_label.items()}
NUM_CLASSES = len(id_to_label)

# ── Helpers (copy từ train để tái tạo đúng dataset) ────────────────────────
mp_holistic = mp.solutions.holistic

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    frames = []
    with mp_holistic.Holistic(model_complexity=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            pose = np.zeros((33,3), np.float32)
            lh   = np.zeros((21,3), np.float32)
            rh   = np.zeros((21,3), np.float32)
            if res.pose_landmarks:
                for i,lm in enumerate(res.pose_landmarks.landmark): pose[i]=[lm.x,lm.y,lm.z]
            if res.left_hand_landmarks:
                for i,lm in enumerate(res.left_hand_landmarks.landmark): lh[i]=[lm.x,lm.y,lm.z]
            if res.right_hand_landmarks:
                for i,lm in enumerate(res.right_hand_landmarks.landmark): rh[i]=[lm.x,lm.y,lm.z]
            frames.append(np.concatenate([pose,lh,rh]).flatten())
    cap.release()
    return np.array(frames, np.float32) if len(frames)>=5 else None

def normalize_and_interpolate(seq):
    T = len(seq)
    if T != N_FRAMES:
        x_old = np.linspace(0,1,T); x_new = np.linspace(0,1,N_FRAMES)
        out = np.zeros((N_FRAMES, N_FEATURES), np.float32)
        for i in range(N_FEATURES):
            out[:,i] = interp1d(x_old, seq[:,i], kind="linear",
                                fill_value="extrapolate")(x_new)
        seq = out
    normed = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        kps = frame.reshape(-1,3)
        ls,rs = kps[11], kps[12]
        center = (ls+rs)/2; scale = np.linalg.norm(rs-ls)
        if scale<1e-6: scale=1.0
        normed[t] = ((kps-center)/scale).flatten()
    return normed.astype(np.float32)

def augment(seq):
    aug = seq.copy()
    if np.random.rand()<0.6: aug += np.random.normal(0,0.012,aug.shape).astype(np.float32)
    if np.random.rand()<0.5: aug *= np.random.uniform(0.88,1.12)
    if np.random.rand()<0.5:
        shift = np.random.randint(-3,4)
        if shift>0: aug = np.pad(aug,((shift,0),(0,0)),mode="edge")[:N_FRAMES]
        elif shift<0: aug = np.pad(aug,((0,-shift),(0,0)),mode="edge")[-shift:]
    if np.random.rand()<0.5:
        m = aug.reshape(N_FRAMES,75,3).copy(); m[:,:,0]*=-1
        lh_tmp=m[:,33:54].copy(); m[:,33:54]=m[:,54:75]; m[:,54:75]=lh_tmp
        aug=m.reshape(N_FRAMES,N_FEATURES)
    return aug.astype(np.float32)

# ── Build dataset ──────────────────────────────────────────────────────────
print("🔄 Tái tạo dataset để đánh giá...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().upper() for c in df.columns]
X_list, Y_list = [], []
for _, row in df.iterrows():
    vpath = os.path.join(VIDEOS_DIR, row["VIDEO"])
    if not os.path.exists(vpath): continue
    raw = extract_features(vpath)
    if raw is None: continue
    base = normalize_and_interpolate(raw)
    X_list.append(base); Y_list.append(label_to_id[row["LABEL"]])
    for _ in range(AUGMENT_COUNT):
        X_list.append(augment(base)); Y_list.append(label_to_id[row["LABEL"]])

X = np.array(X_list, np.float32)
Y = np.array(Y_list, np.int32)
_, X_val, _, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print(f"   Val set: {X_val.shape[0]} mẫu")

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN A — ĐÁNH GIÁ MODEL CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 PHẦN A: Đánh giá best_bilstm.keras")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_bilstm.keras"))

y_pred_prob = model.predict(X_val, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

acc_top1 = accuracy_score(y_val, y_pred)
acc_top5 = top_k_accuracy_score(y_val, y_pred_prob, k=5)
print(f"   Top-1 Accuracy: {acc_top1:.4f} ({acc_top1*100:.2f}%)")
print(f"   Top-5 Accuracy: {acc_top5:.4f} ({acc_top5*100:.2f}%)")

# Per-class accuracy
report = classification_report(y_val, y_pred,
                                target_names=[id_to_label[i] for i in range(NUM_CLASSES)],
                                output_dict=True)
per_class_df = pd.DataFrame(report).T
per_class_df.to_csv(os.path.join(EVAL_DIR, "per_class_accuracy.csv"))
print(f"   per_class_accuracy.csv đã lưu")

# Confusion matrix (full & top-20 worst classes)
cm = confusion_matrix(y_val, y_pred)
labels = [id_to_label[i] for i in range(NUM_CLASSES)]

fig, ax = plt.subplots(figsize=(max(16, NUM_CLASSES//3), max(14, NUM_CLASSES//3)))
sns.heatmap(cm, annot=(NUM_CLASSES<=30), fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            linewidths=0.3 if NUM_CLASSES<=30 else 0)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix — Val Acc: {acc_top1*100:.1f}%", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"   confusion_matrix.png đã lưu")

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN B — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔬 PHẦN B: Ablation Study — 6 cấu hình")

# Index keypoints trong vector 225 = (75 điểm × 3)
# Pose: 0..98  (33 điểm × 3 = 99)
# LH:  99..161 (21 điểm × 3 = 63)
# RH: 162..224 (21 điểm × 3 = 63)
POSE_IDX = list(range(0,   99))
LH_IDX   = list(range(99,  162))
RH_IDX   = list(range(162, 225))
HAND_IDX = LH_IDX + RH_IDX

def mask_features(X, keep_idx):
    out = np.zeros_like(X)
    out[:, :, keep_idx] = X[:, :, keep_idx]
    return out

def build_bilstm(n_features_eff, num_classes):
    inp = tf.keras.Input(shape=(N_FRAMES, n_features_eff))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def build_gru(n_features_eff, num_classes):
    inp = tf.keras.Input(shape=(N_FRAMES, n_features_eff))
    x = tf.keras.layers.GRU(64)(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def build_transformer(n_features_eff, num_classes):
    inp = tf.keras.Input(shape=(N_FRAMES, n_features_eff))
    # Positional encoding đơn giản bằng Dense
    x = tf.keras.layers.Dense(128)(inp)
    # Multi-head attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

ablation_configs = [
    # (tên, feature_indices, build_fn)
    ("Pose only — BiLSTM",       POSE_IDX,  build_bilstm),
    ("Hand only — BiLSTM",       HAND_IDX,  build_bilstm),
    ("Pose+Hand — GRU",          list(range(225)), build_gru),
    ("Pose+Hand — Transformer",  list(range(225)), build_transformer),
    ("Pose+Hand — BiLSTM",       list(range(225)), build_bilstm),   # model chính
]

ablation_results = []
cb_ablation = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                     restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=5, verbose=0),
]

_, X_tr, _, y_tr = train_test_split(X, Y, test_size=0.8, random_state=42, stratify=Y)  # 20% train để ablation nhanh
_, X_v,  _, y_v  = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

for name, feat_idx, build_fn in ablation_configs:
    print(f"\n  ▶ {name}")
    n_eff = len(feat_idx)

    # Slice đúng features
    X_tr_sub = X_tr[:, :, feat_idx]
    X_v_sub  = X_v[:,  :, feat_idx]

    m = build_fn(n_eff, NUM_CLASSES)
    m.fit(X_tr_sub, y_tr,
          validation_data=(X_v_sub, y_v),
          epochs=80, batch_size=32,
          callbacks=cb_ablation, verbose=0)

    _, val_acc = m.evaluate(X_v_sub, y_v, verbose=0)
    n_params = m.count_params()
    print(f"     val_acc={val_acc:.4f}  params={n_params:,}")
    ablation_results.append({"Config": name, "Features": n_eff,
                              "Val Accuracy": round(val_acc, 4),
                              "Params": n_params})

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(os.path.join(EVAL_DIR, "ablation_results.csv"), index=False)
print(f"\n   ablation_results.csv đã lưu")
print(abl_df.to_string(index=False))

# Biểu đồ ablation
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4C9BE8" if "BiLSTM" in c["Config"] and "Pose+Hand" in c["Config"]
          else "#A0C4FF" for c in ablation_results]
bars = ax.barh(abl_df["Config"], abl_df["Val Accuracy"]*100, color=colors)
ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9)
ax.set_xlabel("Validation Accuracy (%)")
ax.set_title("Ablation Study — Feature & Architecture Comparison")
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "ablation_chart.png"), dpi=150)
plt.close()
print("   ablation_chart.png đã lưu")

print(f"""
✅ BƯỚC 1 HOÀN TẤT
   Kết quả lưu tại: {EVAL_DIR}
   ├── confusion_matrix.png
   ├── per_class_accuracy.csv
   ├── ablation_results.csv
   └── ablation_chart.png
""")