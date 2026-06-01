import sys
import io
import os
import json
import traceback
import numpy as np
import pandas as pd

# Đồng bộ hiển thị ký tự tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Cấu hình môi trường GPU ───────────────────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Tìm thấy {len(gpus)} GPU: {[g.name for g in gpus]}")
        print(f"[GPU] Sử dụng GPU để tính toán!")
    except RuntimeError as e:
        print(f"[GPU] Lỗi cấu hình GPU: {e}")
else:
    print("[GPU] Không tìm thấy GPU — chuyển sang sử dụng CPU")

print(f"[TF] Phiên bản TensorFlow: {tf.__version__}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  accuracy_score, top_k_accuracy_score)
    from sklearn.model_selection import train_test_split
    from scipy.interpolate import interp1d
    import mediapipe as mp
    import cv2
    print("[OK] Đã nạp thành công toàn bộ thư viện cần thiết")
except ImportError as e:
    print(f"[LỖI] Thiếu thư viện: {e}")
    sys.exit(1)

# ── Đường dẫn hệ thống ─────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\MYPC\Downloads\Dataset"
MODEL_DIR  = os.path.join(BASE_DIR, "Models")
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
CSV_PATH   = os.path.join(BASE_DIR, "Labels", "label_gold.csv")
EVAL_DIR   = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

N_FRAMES      = 30
N_FEATURES    = 345
AUGMENT_COUNT = 30

# ── Phân nhóm các chỉ số đặc trưng (Feature Slices) ────────────────────────
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

# ── Đọc bản đồ nhãn ────────────────────────────────────────────────────────
print("\n[1/8] Đang nạp bản đồ nhãn...")
try:
    with open(os.path.join(MODEL_DIR, "label_map.json"), encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_id = {v: k for k, v in id_to_label.items()}
    NUM_CLASSES = len(id_to_label)
    print(f"     Tìm thấy {NUM_CLASSES} nhãn")
except Exception as e:
    print(f"[LỖI] Không đọc được tệp label_map.json: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Hàm bổ trợ trích xuất dữ liệu ──────────────────────────────────────────
mp_holistic = mp.solutions.holistic

def extract_features(video_path):
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
                pose = np.zeros((33,3), np.float32)
                lh   = np.zeros((21,3), np.float32)
                rh   = np.zeros((21,3), np.float32)
                lips = np.zeros((40,3), np.float32)
                if res.pose_landmarks:
                    for i,lm in enumerate(res.pose_landmarks.landmark):
                        pose[i] = [lm.x, lm.y, lm.z]
                if res.left_hand_landmarks:
                    for i,lm in enumerate(res.left_hand_landmarks.landmark):
                        lh[i] = [lm.x, lm.y, lm.z]
                if res.right_hand_landmarks:
                    for i,lm in enumerate(res.right_hand_landmarks.landmark):
                        rh[i] = [lm.x, lm.y, lm.z]
                if res.face_landmarks:
                    for i,idx in enumerate(LIPS_INDICES):
                        lm = res.face_landmarks.landmark[idx]
                        lips[i] = [lm.x, lm.y, lm.z]
                frames.append(np.concatenate([pose,lh,rh,lips]).flatten())
        cap.release()
        return np.array(frames, np.float32) if len(frames) >= 5 else None
    except Exception as e:
        print(f"     [LỖI TRÍCH XUẤT] {video_path}: {e}")
        return None

def normalize_and_interpolate(seq):
    T = len(seq)
    if T != N_FRAMES:
        x_old = np.linspace(0,1,T)
        x_new = np.linspace(0,1,N_FRAMES)
        out   = np.zeros((N_FRAMES, N_FEATURES), np.float32)
        for i in range(N_FEATURES):
            out[:,i] = interp1d(x_old, seq[:,i], kind="linear",
                                fill_value="extrapolate")(x_new)
        seq = out
    normed = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        kps    = frame.reshape(-1, 3)
        ls, rs = kps[11], kps[12]
        center = (ls + rs) / 2
        scale  = np.linalg.norm(rs - ls)
        if scale < 1e-6:
            scale = 1.0
        normed[t] = ((kps - center) / scale).flatten()
    return normed.astype(np.float32)

def augment(seq):
    aug = seq.copy()
    if np.random.rand() < 0.6:
        aug += np.random.normal(0, 0.012, aug.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        aug *= np.random.uniform(0.88, 1.12)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 4)
        if shift > 0:
            aug = np.pad(aug, ((shift,0),(0,0)), mode="edge")[:N_FRAMES]
        elif shift < 0:
            aug = np.pad(aug, ((0,-shift),(0,0)), mode="edge")[-shift:]
    if np.random.rand() < 0.5:
        body = aug[:,:225].reshape(N_FRAMES,75,3).copy()
        body[:,:,0] *= -1
        lh_tmp = body[:,33:54].copy()
        body[:,33:54] = body[:,54:75]
        body[:,54:75] = lh_tmp
        aug[:,:225] = body.reshape(N_FRAMES,225)
        lips = aug[:,225:].reshape(N_FRAMES,40,3).copy()
        lips[:,:,0] *= -1
        aug[:,225:] = lips.reshape(N_FRAMES,120)
    return aug.astype(np.float32)

# ── Khởi tạo Tập dữ liệu ───────────────────────────────────────────────────
print("\n[2/8] Đang tái tạo tập dữ liệu (Đầy đủ 345 đặc trưng)...")
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    X_list, Y_list = [], []
    total = len(df)
    ok = 0
    for i, (_, row) in enumerate(df.iterrows()):
        vpath = os.path.join(VIDEOS_DIR, row["VIDEO"])
        if not os.path.exists(vpath):
            print(f"     [{i+1}/{total}] BỎ QUA (Tệp không tồn tại): {row['VIDEO']}")
            continue
        raw = extract_features(vpath)
        if raw is None:
            print(f"     [{i+1}/{total}] BỎ QUA (Không thể trích xuất): {row['VIDEO']}")
            continue
        base = normalize_and_interpolate(raw)
        X_list.append(base)
        Y_list.append(label_to_id[row["LABEL"]])
        for _ in range(AUGMENT_COUNT):
            X_list.append(augment(base))
            Y_list.append(label_to_id[row["LABEL"]])
        ok += 1
        if ok % 10 == 0:
            print(f"     Đã xử lý xong {ok}/{total} video...", flush=True)

    X = np.array(X_list, np.float32)
    Y = np.array(Y_list, np.int32)
    print(f"     Tập dữ liệu hoàn thành: Kích thước {X.shape} — Trích xuất thành công {ok}/{total} video")
except Exception as e:
    print(f"[LỖI] Tạo tập dữ liệu thất bại: {e}")
    traceback.print_exc()
    sys.exit(1)

# Thực hiện chia tập dữ liệu nhất quán 80/20 đúng 1 lần
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print(f"     Tập huấn luyện (Train): {X_train.shape[0]} | Tập kiểm thử (Val): {X_val.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════
# PHẦN A — ĐÁNH GIÁ MÔ HÌNH CHÍNH
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/8] Phần A: Đánh giá mô hình chính best_bilstm.keras...")
try:
    model_path = os.path.join(MODEL_DIR, "best_bilstm.keras")
    if not os.path.exists(model_path):
        print(f"     [LỖI] Không tìm thấy mô hình tại: {model_path}")
        print("     -> Vui lòng chạy tệp TRAIN_MODEL_MEDIA.py trước!")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    print(f"     Đã nạp mô hình: Kích thước {model.count_params():,} tham số")

    print("     Đang tiến hành dự đoán...", flush=True)
    y_prob = model.predict(X_val, verbose=1, batch_size=64)
    y_pred = np.argmax(y_prob, axis=1)

    acc_top1 = accuracy_score(y_val, y_pred)
    acc_top5 = top_k_accuracy_score(y_val, y_prob, k=min(5, NUM_CLASSES))
    print(f"     Độ chính xác Top-1 (Top-1 Accuracy): {acc_top1*100:.2f}%")
    print(f"     Độ chính xác Top-5 (Top-5 Accuracy): {acc_top5*100:.2f}%")

    # Tạo tệp báo cáo chi tiết từng nhãn
    report = classification_report(
        y_val, y_pred,
        target_names=[id_to_label[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )
    pd.DataFrame(report).T.to_csv(
        os.path.join(EVAL_DIR, "per_class_accuracy.csv"),
        encoding="utf-8-sig"
    )
    print("     Đã lưu tệp báo cáo per_class_accuracy.csv thành công.")

    # Vẽ ma trận nhầm lẫn Confusion matrix
    cm     = confusion_matrix(y_val, y_pred)
    labels = [id_to_label[i] for i in range(NUM_CLASSES)]
    fig_size = max(16, NUM_CLASSES // 3)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=(NUM_CLASSES <= 30), fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3 if NUM_CLASSES <= 30 else 0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - Val Acc: {acc_top1*100:.1f}%")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("     Đã lưu biểu đồ confusion_matrix.png thành công.")

except Exception as e:
    print(f"[LỖI] Đánh giá Phần A gặp sự cố: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════
# PHẦN B — NGHIÊN CỨU THỬ NGHIỆM ĐỊNH LƯỢNG (Ablation Study)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/8] Phần B: Ablation Study - Thử nghiệm 4 nhóm đặc trưng x 3 cấu trúc = 12 mô hình")

def build_bilstm(n_feat, n_cls):
    inp = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def build_gru(n_feat, n_cls):
    inp = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x = tf.keras.layers.GRU(128, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def build_transformer(n_feat, n_cls):
    inp = tf.keras.Input(shape=(N_FRAMES, n_feat))
    x   = tf.keras.layers.Dense(128)(inp)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x   = tf.keras.layers.LayerNormalization()(x + attn)
    ff  = tf.keras.layers.Dense(256, activation="relu")(x)
    ff  = tf.keras.layers.Dense(128)(ff)
    x   = tf.keras.layers.LayerNormalization()(x + ff)
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

ARCHITECTURES = {
    "GRU"        : build_gru,
    "Transformer": build_transformer,
    "BiLSTM"     : build_bilstm,
}

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
        tag = f"{feat_name} - {arch_name}"
        print(f"\n  [{exp_cur}/{exp_total}] Đang huấn luyện: {tag} ({len(feat_idx)} đặc trưng)",
              flush=True)
        try:
            # Lấy đặc trưng tương ứng từ tập Train/Val nhất quán duy nhất
            Xtr = X_train[:, :, feat_idx]
            Xv  = X_val[:,  :, feat_idx]

            m = build_fn(len(feat_idx), NUM_CLASSES)
            m.fit(Xtr, y_train,
                  validation_data=(Xv, y_val),
                  epochs=80,
                  batch_size=64,
                  callbacks=cb_abl,
                  verbose=0)

            _, acc = m.evaluate(Xv, y_val, verbose=0)
            params = m.count_params()
            print(f"     Hoàn tất: val_acc={acc:.4f} | params={params:,}", flush=True)

            rows.append({
                "Feature Group": feat_name,
                "Architecture" : arch_name,
                "Num Features" : len(feat_idx),
                "Val Accuracy" : round(acc, 4),
                "Params"       : params,
            })
        except Exception as e:
            print(f"     [LỖI THỬ NGHIỆM] {tag}: {e}")
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

# ── Lưu kết quả Ablation Study ─────────────────────────────────────────────
print("\n[5/8] Đang lưu bảng kết quả nghiên cứu định lượng...")
try:
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(os.path.join(EVAL_DIR, "ablation_results.csv"),
                  index=False, encoding="utf-8-sig")
    print("     Đã lưu ablation_results.csv thành công.")
    print("\n" + abl_df.to_string(index=False))
except Exception as e:
    print(f"[LỖI] Ghi dữ liệu ablation CSV gặp lỗi: {e}")
    traceback.print_exc()

# ── Biểu đồ hóa kết quả ───────────────────────────────────────────────────
print("\n[6/8] Đang khởi tạo sơ đồ kết quả định lượng...")
try:
    order = ["Hand only", "Hand + Pose", "Hand + Lips", "Hand + Pose + Lips"]

    # Bản đồ nhiệt (Heatmap) so sánh cấu trúc và nhóm đặc trưng
    pivot = abl_df.pivot(
        index="Feature Group", columns="Architecture",
        values="Val Accuracy") * 100
    for col in ["GRU", "Transformer", "BiLSTM"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["GRU", "Transformer", "BiLSTM"]]
    pivot = pivot.reindex([o for o in order if o in pivot.index])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                vmin=max(0, pivot.values.min()-5),
                vmax=min(100, pivot.values.max()+2))
    ax.set_title("Ablation Study - Val Accuracy (%) by Feature Group & Architecture",
                 fontsize=11, pad=12)
    ax.set_xlabel("Architecture", fontsize=10)
    ax.set_ylabel("Feature Group", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ablation_heatmap.png"), dpi=150)
    plt.close()
    print("     Đã lưu biểu đồ ablation_heatmap.png")

    # Biểu đồ cột so sánh nhóm đặc trưng (Chỉ tính trên kiến trúc BiLSTM)
    bilstm_df = abl_df[abl_df["Architecture"] == "BiLSTM"].copy()
    bilstm_df = bilstm_df.set_index("Feature Group")
    bilstm_df = bilstm_df.reindex([o for o in order if o in bilstm_df.index])

    colors = ["#A0C4FF", "#74B9FF", "#4C9BE8", "#1565C0"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(bilstm_df.index,
                  bilstm_df["Val Accuracy"] * 100,
                  color=colors[:len(bilstm_df)], width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Feature Group Comparison - BiLSTM Architecture")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ablation_feature_bar.png"), dpi=150)
    plt.close()
    print("     Đã lưu biểu đồ ablation_feature_bar.png")

    # Biểu đồ cột so sánh kiến trúc (Chỉ tính trên đầy đủ Pose+Hand+Lips)
    full_df = abl_df[abl_df["Feature Group"] == "Hand + Pose + Lips"].copy()
    if len(full_df) > 0:
        colors2 = ["#74B9FF", "#4C9BE8", "#1565C0"]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars2 = ax.bar(full_df["Architecture"],
                       full_df["Val Accuracy"] * 100,
                       color=colors2[:len(full_df)], width=0.4)
        ax.bar_label(bars2, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Architecture Comparison - Hand + Pose + Lips Features")
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "ablation_arch_bar.png"), dpi=150)
        plt.close()
        print("     Đã lưu biểu đồ ablation_arch_bar.png")

except Exception as e:
    print(f"[LỖI] Tạo sơ đồ đồ thị thất bại: {e}")
    traceback.print_exc()

print(f"""
[HOÀN TẤT] Quá trình thực thi hoàn thành. Kết quả được lưu tại: {EVAL_DIR}
  - confusion_matrix.png
  - per_class_accuracy.csv
  - ablation_results.csv
  - ablation_heatmap.png
  - ablation_feature_bar.png
  - ablation_arch_bar.png
""")