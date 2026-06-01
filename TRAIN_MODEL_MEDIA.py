import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import json
import sys
import traceback

# Thêm vào đầu file train, trước tất cả import
import sys
import traceback

def custom_excepthook(exc_type, exc_value, exc_tb):
    with open("error_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        f.flush()

sys.excepthook = custom_excepthook

# ==========================================
# 1. CẤU HÌNH
# ==========================================
BASE_DIR    = r"C:\Users\MYPC\Downloads\Dataset"
CSV_PATH    = os.path.join(BASE_DIR, "Labels", "label_gold.csv")
VIDEOS_DIR  = os.path.join(BASE_DIR, "Videos")
MODEL_DIR   = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

N_FRAMES      = 30
AUGMENT_COUNT = 30

# ── Layout vector keypoints ──────────────────────────────────────────────
# [  0: 62] Pose      — 33 điểm × 3 = 99   (vai/thân/tay/chân)
# [ 99:161] Left Hand — 21 điểm × 3 = 63
# [162:224] Right Hand— 21 điểm × 3 = 63
# [225:344] Lips      — 40 điểm × 3 = 120
#                                     ───
# Tổng đầy đủ (Tay + Thân + Miệng)   = 345

N_FEATURES = 345   # full vector — ablation sẽ slice theo index

# Index nhóm
POSE_SLICE = slice(0,   99)    # thân / pose
LH_SLICE   = slice(99,  162)   # tay trái
RH_SLICE   = slice(162, 225)   # tay phải
LIPS_SLICE = slice(225, 345)   # miệng

# 40 điểm lips (outer + inner) trong face mesh MediaPipe
LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,          # outer (20)
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95    # inner (20)
]

print("🚀 PIPELINE HUẤN LUYỆN VSL — SERVER MODE (full quality, 345 features)")

# ==========================================
# 2. LABEL MAP
# ==========================================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().upper() for c in df.columns]
unique_labels = sorted(df["LABEL"].unique())
NUM_CLASSES   = len(unique_labels)
label_to_id   = {lbl: i for i, lbl in enumerate(unique_labels)}
id_to_label   = {i: lbl for lbl, i in label_to_id.items()}

with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=2)

print(f"📊 {NUM_CLASSES} nhãn — label_map.json đã lưu")

# ==========================================
# 3. TRÍCH XUẤT KEYPOINTS (FULL — 345)
# ==========================================
mp_holistic = mp.solutions.holistic

def extract_features(video_path: str):
    """
    Trả về mảng (T, 345) float32:
      [:99]   Pose, [99:162] LH, [162:225] RH, [225:345] Lips
    Không resize, không nén JPEG.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

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


# ==========================================
# 4. CHUẨN HÓA VAI + NỘI SUY VỀ N_FRAMES
# ==========================================
def normalize_and_interpolate(seq):
    T = len(seq)
    if T != N_FRAMES:
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, N_FRAMES)
        out   = np.zeros((N_FRAMES, N_FEATURES), np.float32)
        for i in range(N_FEATURES):
            out[:, i] = interp1d(x_old, seq[:, i], kind="linear",
                                  fill_value="extrapolate")(x_new)
        seq = out

    # Shoulder normalization — dùng vai trái (11) và phải (12) trong Pose
    normed = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        kps    = frame.reshape(-1, 3)   # (115, 3)
        ls, rs = kps[11], kps[12]
        center = (ls + rs) / 2.0
        scale  = np.linalg.norm(rs - ls)
        if scale < 1e-6:
            scale = 1.0
        normed[t] = ((kps - center) / scale).flatten()

    return normed.astype(np.float32)


# ==========================================
# 5. AUGMENTATION
# ==========================================
def augment(seq):
    aug = seq.copy()

    # Nhiễu Gaussian
    if np.random.rand() < 0.6:
        aug += np.random.normal(0, 0.012, aug.shape).astype(np.float32)

    # Scale không gian
    if np.random.rand() < 0.5:
        aug *= np.random.uniform(0.88, 1.12)

    # Time shift ±3 frame
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 4)
        if shift > 0:
            aug = np.pad(aug, ((shift, 0), (0, 0)), mode="edge")[:N_FRAMES]
        elif shift < 0:
            aug = np.pad(aug, ((0, -shift), (0, 0)), mode="edge")[-shift:]

    # Time warp
    if np.random.rand() < 0.4:
        warp  = np.random.uniform(0.85, 1.15)
        T_new = max(5, int(N_FRAMES * warp))
        x_old = np.linspace(0, 1, N_FRAMES)
        x_new = np.linspace(0, 1, T_new)
        warped = np.zeros((T_new, N_FEATURES), np.float32)
        for i in range(N_FEATURES):
            warped[:, i] = interp1d(x_old, aug[:, i], kind="linear",
                                     fill_value="extrapolate")(x_new)
        x_back = np.linspace(0, 1, T_new)
        x_orig = np.linspace(0, 1, N_FRAMES)
        for i in range(N_FEATURES):
            aug[:, i] = interp1d(x_back, warped[:, i], kind="linear",
                                  fill_value="extrapolate")(x_orig)

    # Mirror tay trái ↔ phải (chỉ mirror phần tay + pose, KHÔNG mirror lips)
    if np.random.rand() < 0.5:
        # Lấy phần tay + pose (225 feature đầu)
        body = aug[:, :225].reshape(N_FRAMES, 75, 3).copy()
        body[:, :, 0] *= -1                       # flip trục X
        lh_tmp = body[:, 33:54].copy()
        body[:, 33:54] = body[:, 54:75]           # hoán đổi LH ↔ RH
        body[:, 54:75] = lh_tmp
        aug[:, :225] = body.reshape(N_FRAMES, 225)
        # Lips: chỉ flip trục X, không hoán đổi
        lips = aug[:, 225:].reshape(N_FRAMES, 40, 3).copy()
        lips[:, :, 0] *= -1
        aug[:, 225:] = lips.reshape(N_FRAMES, 120)

    return aug.astype(np.float32)


# ==========================================
# 6. TẠO DATASET
# ==========================================
X_list, Y_list = [], []
total = len(df)
print(f"\n🔄 Xử lý {total} video...")
ok = 0

for idx, row in df.iterrows():
    vpath    = os.path.join(VIDEOS_DIR, row["VIDEO"])
    label_id = label_to_id[row["LABEL"]]

    print(f"  [{idx+1}/{total}] {row['VIDEO']} → {row['LABEL']}", end=" ")

    if not os.path.exists(vpath):
        print("❌ không tìm thấy file"); continue

    raw = extract_features(vpath)
    if raw is None:
        print("❌ video lỗi hoặc quá ngắn"); continue

    base = normalize_and_interpolate(raw)
    X_list.append(base);          Y_list.append(label_id)
    for _ in range(AUGMENT_COUNT):
        X_list.append(augment(base)); Y_list.append(label_id)

    ok += 1
    print(f"✅ ({ok} ok)")

X = np.array(X_list, np.float32)
Y = np.array(Y_list, np.int32)
print(f"\n✅ {ok}/{total} video — dataset: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ==========================================
# 7. MÔ HÌNH BiLSTM — train trên full 345 features
# ==========================================
print("\n🧠 XÂY DỰNG MÔ HÌNH BiLSTM (345 features = Tay + Thân + Miệng)...")

inputs = tf.keras.Input(shape=(N_FRAMES, N_FEATURES), name="keypoints")
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_prob")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_bilstm.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20,
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7,
        min_lr=1e-6, verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "training_log.csv")),
]

# ==========================================
# 8. HUẤN LUYỆN
# ==========================================
print("\n⚡ BẮT ĐẦU HUẤN LUYỆN...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150, batch_size=32,
    callbacks=callbacks, verbose=1,
)

# ==========================================
# 9. ĐÁNH GIÁ & LƯU
# ==========================================
best = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_bilstm.keras"))
val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Best model — val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

best.export(os.path.join(MODEL_DIR, "vsl_bilstm_savedmodel"))
print(f"\n🎉 HOÀN TẤT!")
print(f"   Model   : {os.path.join(MODEL_DIR, 'best_bilstm.keras')}")
print(f"   Labels  : {os.path.join(MODEL_DIR, 'label_map.json')}")
print(f"   Log     : {os.path.join(MODEL_DIR, 'training_log.csv')}")