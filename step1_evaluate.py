import sys
import io
import os
import json
import random
import traceback
import numpy as np
import pandas as pd

# Đồng bộ hiển thị tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Cấu hình GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Tìm thấy {len(gpus)} GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU] Lỗi cấu hình GPU: {e}")
else:
    print("[GPU] Không tìm thấy GPU — sử dụng CPU")

print(f"[TF] Phiên bản TensorFlow: {tf.__version__}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  accuracy_score, top_k_accuracy_score)
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from scipy.interpolate import interp1d
    import mediapipe as mp
    import cv2
    print("[OK] Đã nạp thành công toàn bộ thư viện")
except ImportError as e:
    print(f"[LỖI] Thiếu thư viện: {e}")
    sys.exit(1)

# ==========================================
# CẤU HÌNH
# ==========================================
SEED          = 42
BASE_DIR      = r"C:\Users\MYPC\Downloads\Dataset"
VIDEOS_UPDATE = os.path.join(BASE_DIR, "videos_update")
MODEL_DIR     = os.path.join(BASE_DIR, "Models")
EVAL_DIR      = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

N_FRAMES      = 30
N_FEATURES    = 345
AUGMENT_COUNT = 30

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Phân nhóm đặc trưng
IDX_POSE = list(range(0,   99))
IDX_LH   = list(range(99,  162))
IDX_RH   = list(range(162, 225))
IDX_LIPS = list(range(225, 345))

IDX_HAND           = IDX_LH + IDX_RH
IDX_HAND_POSE      = IDX_POSE + IDX_LH + IDX_RH
IDX_HAND_LIPS      = IDX_LH + IDX_RH + IDX_LIPS
IDX_HAND_POSE_LIPS = IDX_POSE + IDX_LH + IDX_RH + IDX_LIPS

FEATURE_GROUPS = {
    "Hand only"          : IDX_HAND,
    "Hand + Pose"        : IDX_HAND_POSE,
    "Hand + Lips"        : IDX_HAND_LIPS,
    "Hand + Pose + Lips" : IDX_HAND_POSE_LIPS,
}

LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95
]

# ==========================================
# 1. ĐỌC LABEL MAP
# ==========================================
print("\n[1/8] Đang nạp bản đồ nhãn...")
try:
    with open(os.path.join(MODEL_DIR, "label_map.json"), encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_id = {v: k for k, v in id_to_label.items()}
    NUM_CLASSES  = len(id_to_label)
    unique_labels = [id_to_label[i] for i in range(NUM_CLASSES)]
    print(f"     Tìm thấy {NUM_CLASSES} nhãn")
except Exception as e:
    print(f"[LỖI] Không đọc được label_map.json: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 2. HÀM TRÍCH XUẤT & XỬ LÝ
# ==========================================
mp_holistic = mp.solutions.holistic

def extract_features(video_path: str):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        frames = []
        with mp_holistic.Holistic(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as h:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = h.process(rgb)
                pose = np.zeros((33, 3), np.float32)
                lh   = np.zeros((21, 3), np.float32)
                rh   = np.zeros((21, 3), np.float32)
                lips = np.zeros((40, 3), np.float32)
                if res.pose_landmarks:
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        pose[i] = [lm.x, lm.y, lm.z]
                if res.left_hand_landmarks:
                    for i, lm in enumerate(res.left_hand_landmarks.landmark):
                        lh[i] = [lm.x, lm.y, lm.z]
                if res.right_hand_landmarks:
                    for i, lm in enumerate(res.right_hand_landmarks.landmark):
                        rh[i] = [lm.x, lm.y, lm.z]
                if res.face_landmarks:
                    for i, idx in enumerate(LIPS_INDICES):
                        lm = res.face_landmarks.landmark[idx]
                        lips[i] = [lm.x, lm.y, lm.z]
                frames.append(np.concatenate([pose, lh, rh, lips]).flatten())
        cap.release()
        return np.array(frames, np.float32) if len(frames) >= 5 else None
    except Exception as e:
        print(f"     [LỖI TRÍCH XUẤT] {video_path}: {e}")
        return None

def normalize_and_interpolate(seq: np.ndarray) -> np.ndarray:
    T = len(seq)
    if T != N_FRAMES:
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, N_FRAMES)
        out   = np.zeros((N_FRAMES, N_FEATURES), np.float32)
        for i in range(N_FEATURES):
            out[:, i] = interp1d(x_old, seq[:, i], kind="linear",
                                  fill_value="extrapolate")(x_new)
        seq = out
    normed = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        kps    = frame.reshape(-1, 3)
        ls, rs = kps[11], kps[12]
        center = (ls + rs) / 2.0
        scale  = np.linalg.norm(rs - ls)
        if scale < 1e-6:
            scale = 1.0
        normed[t] = ((kps - center) / scale).flatten()
    return normed.astype(np.float32)

def augment(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()
    if np.random.rand() < 0.6:
        aug += np.random.normal(0, 0.012, aug.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        aug *= np.random.uniform(0.88, 1.12)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 4)
        if shift > 0:
            aug = np.pad(aug, ((shift, 0), (0, 0)), mode="edge")[:N_FRAMES]
        elif shift < 0:
            aug = np.pad(aug, ((0, -shift), (0, 0)), mode="edge")[-shift:]
    if np.random.rand() < 0.5:
        body = aug[:, :225].reshape(N_FRAMES, 75, 3).copy()
        body[:, :, 0] *= -1
        lh_tmp = body[:, 33:54].copy()
        body[:, 33:54] = body[:, 54:75]
        body[:, 54:75] = lh_tmp
        aug[:, :225] = body.reshape(N_FRAMES, 225)
        lips = aug[:, 225:].reshape(N_FRAMES, 40, 3).copy()
        lips[:, :, 0] *= -1
        aug[:, 225:] = lips.reshape(N_FRAMES, 120)
    return aug.astype(np.float32)

# ==========================================
# 3. NẠP / XÂY DỰNG DỮ LIỆU CHUẨN CACHE
# ==========================================
print("\n[2/8] Đang nạp tập dữ liệu...")

CACHE_X_TRAIN   = os.path.join(MODEL_DIR, "X_train_cached.npy")
CACHE_Y_TRAIN   = os.path.join(MODEL_DIR, "y_train_cached.npy")
CACHE_X_VAL     = os.path.join(MODEL_DIR, "X_val_cached.npy")
CACHE_Y_VAL     = os.path.join(MODEL_DIR, "y_val_cached.npy")
CACHE_META_PATH = os.path.join(MODEL_DIR, "cache_meta.json")

def _cache_is_valid() -> bool:
    required = [CACHE_X_TRAIN, CACHE_Y_TRAIN, CACHE_X_VAL, CACHE_Y_VAL, CACHE_META_PATH]
    if not all(os.path.exists(p) for p in required):
        return False
    try:
        with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("num_classes") != NUM_CLASSES:
            print("     ⚠️  Cache lỗi thời: số lớp thay đổi — tạo lại...", flush=True)
            return False
        if meta.get("labels") != unique_labels:
            print("     ⚠️  Cache lỗi thời: danh sách nhãn thay đổi — tạo lại...", flush=True)
            return False
    except Exception:
        return False
    return True

try:
    if _cache_is_valid():
        print("     [CACHE] Phát hiện cache hợp lệ — đang nạp...", flush=True)
        X_train = np.load(CACHE_X_TRAIN)
        y_train = np.load(CACHE_Y_TRAIN)
        X_val   = np.load(CACHE_X_VAL)
        y_val   = np.load(CACHE_Y_VAL)
        print(f"     ✅ Nạp cache thành công — Train: {X_train.shape} | Val: {X_val.shape}")
    else:
        print("     [FOLDER] Không có cache — đang quét thư mục...", flush=True)
        all_videos = []
        for label in unique_labels:
            folder_path = os.path.join(VIDEOS_UPDATE, label)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".mp4"):
                    all_videos.append((label, os.path.join(folder_path, fname)))

        if len(all_videos) == 0:
            print("[LỖI] Không tìm thấy video nào!")
            sys.exit(1)

        train_videos, val_videos = train_test_split(
            all_videos,
            test_size=0.2,
            random_state=SEED
        )
        print(f"     Tổng: {len(all_videos)} | Train gốc: {len(train_videos)} | Val gốc: {len(val_videos)}")

        X_train_list, y_train_list = [], []
        X_val_list,   y_val_list   = [], []

        print("     Trích xuất tập Train (có augment)...", flush=True)
        for i, (label, vpath) in enumerate(train_videos):
            raw = extract_features(vpath)
            if raw is None:
                continue
            base = normalize_and_interpolate(raw)
            X_train_list.append(base)
            y_train_list.append(label_to_id[label])
            for _ in range(AUGMENT_COUNT):
                X_train_list.append(augment(base))
                y_train_list.append(label_to_id[label])

        print("     Trích xuất tập Val (sạch, không augment)...", flush=True)
        for label, vpath in val_videos:
            raw = extract_features(vpath)
            if raw is None:
                continue
            base = normalize_and_interpolate(raw)
            X_val_list.append(base)
            y_val_list.append(label_to_id[label])

        X_train = np.array(X_train_list, np.float32)
        y_train = np.array(y_train_list, np.int32)
        X_val   = np.array(X_val_list,   np.float32)
        y_val   = np.array(y_val_list,   np.int32)

        # Lưu cache kèm metadata
        np.save(CACHE_X_TRAIN, X_train)
        np.save(CACHE_Y_TRAIN, y_train)
        np.save(CACHE_X_VAL,   X_val)
        np.save(CACHE_Y_VAL,   y_val)
        with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"num_classes": NUM_CLASSES, "labels": unique_labels}, f,
                      ensure_ascii=False, indent=2)
        print(f"     ✅ Đã tạo cache — Train: {X_train.shape} | Val: {X_val.shape}")

except Exception as e:
    print(f"[LỖI] Nạp tập dữ liệu thất bại: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"     Tập Train: {X_train.shape[0]} mẫu | Tập Val: {X_val.shape[0]} mẫu")

# ==========================================
# PHẦN A — ĐÁNH GIÁ MÔ HÌNH CHÍNH
# ==========================================
print("\n[3/8] Phần A: Đánh giá mô hình chính best_bilstm.keras...")
try:
    model_path = os.path.join(MODEL_DIR, "best_bilstm.keras")
    if not os.path.exists(model_path):
        print(f"     [LỖI] Không tìm thấy mô hình: {model_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    print(f"     Đã nạp mô hình: {model.count_params():,} tham số")

    print("     Đang dự đoán...", flush=True)
    y_prob = model.predict(X_val, verbose=1, batch_size=64)
    y_pred = np.argmax(y_prob, axis=1)

    acc_top1 = accuracy_score(y_val, y_pred)
    # FIX: Truyền danh sách đầy đủ 100 lớp qua labels để tránh sập lỗi
    acc_top5 = top_k_accuracy_score(y_val, y_prob, k=min(5, NUM_CLASSES), labels=list(range(NUM_CLASSES)))
    print(f"     Top-1 Accuracy: {acc_top1*100:.2f}%")
    print(f"     Top-5 Accuracy: {acc_top5*100:.2f}%")

    # Lưu báo cáo per-class
    report = classification_report(
        y_val, y_pred,
        target_names=[id_to_label[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )
    pd.DataFrame(report).T.to_csv(
        os.path.join(EVAL_DIR, "per_class_accuracy.csv"),
        encoding="utf-8-sig"
    )
    print("     Đã lưu per_class_accuracy.csv")

    # Confusion matrix
    cm       = confusion_matrix(y_val, y_pred)
    labels   = [id_to_label[i] for i in range(NUM_CLASSES)]
    fig_size = max(16, NUM_CLASSES // 3)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=(NUM_CLASSES <= 30), fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3 if NUM_CLASSES <= 30 else 0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Val Acc: {acc_top1*100:.1f}%")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("     Đã lưu confusion_matrix.png")

except Exception as e:
    print(f"[LỖI] Đánh giá Phần A thất bại: {e}")
    traceback.print_exc()

# ==========================================
# PHẦN B — ABLATION STUDY
# ==========================================
print("\n[4/8] Phần B: Ablation Study — 4 nhóm đặc trưng × 3 kiến trúc = 12 thử nghiệm")

def build_bilstm(n_feat: int, n_cls: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inp)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def build_gru(n_feat: int, n_cls: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x   = tf.keras.layers.GRU(128, return_sequences=True)(inp)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.GRU(64)(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def build_transformer(n_feat: int, n_cls: int) -> tf.keras.Model:
    inp  = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x    = tf.keras.layers.Dense(128)(inp)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x    = tf.keras.layers.LayerNormalization()(x + attn)
    ff   = tf.keras.layers.Dense(256, activation="relu")(x)
    ff   = tf.keras.layers.Dense(128)(ff)
    x    = tf.keras.layers.LayerNormalization()(x + ff)
    x    = tf.keras.layers.GlobalAveragePooling1D()(x)
    x    = tf.keras.layers.Dense(128, activation="relu")(x)
    out  = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m    = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

ARCHITECTURES = {
    "GRU"        : build_gru,
    "Transformer": build_transformer,
    "BiLSTM"     : build_bilstm,
}

# FIX: Tính bộ Class Weight an toàn đủ 100 lớp (giống tệp train) chống lỗi sập
class_weight_dict = {}
total_samples = len(y_train)
unique_classes, class_counts = np.unique(y_train, return_counts=True)
count_dict = dict(zip(unique_classes, class_counts))

for i in range(NUM_CLASSES):
    if i in count_dict:
        class_weight_dict[i] = total_samples / (NUM_CLASSES * count_dict[i])
    else:
        class_weight_dict[i] = 1.0

cb_abl = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15,
        restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, verbose=0),
]

rows      = []
exp_total = len(FEATURE_GROUPS) * len(ARCHITECTURES)
exp_cur   = 0

for feat_name, feat_idx in FEATURE_GROUPS.items():
    for arch_name, build_fn in ARCHITECTURES.items():
        exp_cur += 1
        tag = f"{feat_name} — {arch_name}"
        print(f"\n  [{exp_cur}/{exp_total}] {tag} ({len(feat_idx)} đặc trưng)", flush=True)
        try:
            Xtr = X_train[:, :, feat_idx]
            Xv  = X_val[:,   :, feat_idx]

            m = build_fn(len(feat_idx), NUM_CLASSES)
            m.fit(
                Xtr, y_train,
                validation_data=(Xv, y_val),
                epochs=80,
                batch_size=64,
                class_weight=class_weight_dict,
                callbacks=cb_abl,
                verbose=0,
            )

            _, acc = m.evaluate(Xv, y_val, verbose=0)
            params = m.count_params()
            print(f"     val_acc={acc:.4f} | params={params:,}", flush=True)

            rows.append({
                "Feature Group": feat_name,
                "Architecture" : arch_name,
                "Num Features" : len(feat_idx),
                "Val Accuracy" : round(acc, 4),
                "Params"       : params,
            })
        except Exception as e:
            print(f"     [LỖI] {tag}: {e}")
            traceback.print_exc()
            rows.append({
                "Feature Group": feat_name,
                "Architecture" : arch_name,
                "Num Features" : len(feat_idx),
                "Val Accuracy" : 0.0,
                "Params"       : 0,
            })
        finally:
            tf.keras.backend.clear_session()

# ==========================================
# 5. LƯU KẾT QUẢ ABLATION
# ==========================================
print("\n[5/8] Đang lưu kết quả ablation study...")
try:
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(os.path.join(EVAL_DIR, "ablation_results.csv"),
                  index=False, encoding="utf-8-sig")
    print("     Đã lưu ablation_results.csv")
    print("\n" + abl_df.to_string(index=False))
except Exception as e:
    print(f"[LỖI] Ghi ablation CSV: {e}")
    traceback.print_exc()

# ==========================================
# 6. VẼ BIỂU ĐỒ
# ==========================================
print("\n[6/8] Đang vẽ biểu đồ kết quả...")
ORDER = ["Hand only", "Hand + Pose", "Hand + Lips", "Hand + Pose + Lips"]

try:
    # Heatmap: Feature Group × Architecture
    pivot = abl_df.pivot(
        index="Feature Group", columns="Architecture",
        values="Val Accuracy"
    ) * 100

    for col in ["GRU", "Transformer", "BiLSTM"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["GRU", "Transformer", "BiLSTM"]]
    pivot = pivot.reindex([o for o in ORDER if o in pivot.index])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                vmin=max(0, pivot.values.min() - 5),
                vmax=min(100, pivot.values.max() + 2))
    ax.set_title("Ablation Study — Val Accuracy (%) by Feature Group & Architecture",
                 fontsize=11, pad=12)
    ax.set_xlabel("Architecture", fontsize=10)
    ax.set_ylabel("Feature Group", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ablation_heatmap.png"), dpi=150)
    plt.close()
    print("     Đã lưu ablation_heatmap.png")

    # Biểu đồ cột: so sánh Feature Group trên BiLSTM
    bilstm_df = abl_df[abl_df["Architecture"] == "BiLSTM"].copy()
    bilstm_df = bilstm_df.set_index("Feature Group")
    bilstm_df = bilstm_df.reindex([o for o in ORDER if o in bilstm_df.index])

    colors = ["#A0C4FF", "#74B9FF", "#4C9BE8", "#1565C0"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(bilstm_df.index,
                  bilstm_df["Val Accuracy"] * 100,
                  color=colors[:len(bilstm_df)], width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Feature Group Comparison — BiLSTM Architecture")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ablation_feature_bar.png"), dpi=150)
    plt.close()
    print("     Đã lưu ablation_feature_bar.png")

    # Biểu đồ cột so sánh kiến trúc (full features)
    arch_df = abl_df[abl_df["Feature Group"] == "Hand + Pose + Lips"].copy()
    if len(arch_df) > 0:
        colors2 = ["#74B9FF", "#4C9BE8", "#1565C0"]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars2 = ax.bar(arch_df["Architecture"],
                       arch_df["Val Accuracy"] * 100,
                       color=colors2[:len(arch_df)], width=0.4)
        ax.bar_label(bars2, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Architecture Comparison — Hand + Pose + Lips Features")
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "ablation_arch_bar.png"), dpi=150)
        plt.close()
        print("     Đã lưu ablation_arch_bar.png")

except Exception as e:
    print(f"[LỖI] Vẽ biểu đồ thất bại: {e}")
    traceback.print_exc()

# ==========================================
# 7. TỔNG KẾT
# ==========================================
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HOÀN TẤT — Kết quả lưu tại: {EVAL_DIR}
╠══════════════════════════════════════════════════════════════╣
║  confusion_matrix.png
║  per_class_accuracy.csv
║  ablation_results.csv
║  ablation_heatmap.png
║  ablation_feature_bar.png
║  ablation_arch_bar.png
╚══════════════════════════════════════════════════════════════╝
""")