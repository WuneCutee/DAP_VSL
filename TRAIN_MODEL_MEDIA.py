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
from typing import Optional

# Thiết lập hiển thị ký tự tiếng Việt trên Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def custom_excepthook(exc_type, exc_value, exc_tb):
    with open("error_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        f.flush()

sys.excepthook = custom_excepthook

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (ĐỌC TRỰC TIẾP FOLDER)
# ==========================================
BASE_DIR      = r"C:\Users\MYPC\Downloads\Dataset"
VIDEOS_UPDATE = os.path.join(BASE_DIR, "videos_update")
MODEL_DIR     = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

N_FRAMES      = 30
AUGMENT_COUNT = 30
N_FEATURES    = 345   # Full vector: Pose(99) + LH(63) + RH(63) + Lips(120)

POSE_SLICE = slice(0,   99)
LH_SLICE   = slice(99,  162)
RH_SLICE   = slice(162, 225)
LIPS_SLICE = slice(225, 345)

# 40 điểm mốc miệng trong Face Mesh
LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95
]

# Kiểm tra GPU để tối ưu hóa huấn luyện
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Tìm thấy GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU] Lỗi cấu hình GPU: {e}")
else:
    print("[GPU] Không nhận diện được GPU — sử dụng CPU để huấn luyện")

print(f"[TF] Phiên bản TensorFlow: {tf.__version__}")
print("🚀 ĐANG KHỞI ĐỘNG PIPELINE HUẤN LUYỆN VSL (ĐỌC FOLDER TRỰC TIẾP)")

# ==========================================
# 2. TỰ ĐỘNG QUÉT FOLDER ĐỂ TẠO LABEL MAP
# ==========================================
if not os.path.exists(VIDEOS_UPDATE):
    print(f"❌ Không tìm thấy thư mục: {VIDEOS_UPDATE}")
    sys.exit(1)

# Lấy danh sách tên thư mục con làm nhãn
unique_labels = sorted([
    d for d in os.listdir(VIDEOS_UPDATE) 
    if os.path.isdir(os.path.join(VIDEOS_UPDATE, d))
])
NUM_CLASSES   = len(unique_labels)
label_to_id   = {lbl: i for i, lbl in enumerate(unique_labels)}
id_to_label   = {i: lbl for lbl, i in label_to_id.items()}

# Lưu tệp label_map.json
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=2)

print(f"📊 Tìm thấy {NUM_CLASSES} nhãn (thư mục con) — Đã cập nhật label_map.json", flush=True)

# ==========================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG (POSE + HANDS + LIPS)
# ==========================================
mp_holistic = mp.solutions.holistic

def extract_features(video_path: str) -> Optional[np.ndarray]:
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
# 4. CHUẨN HÓA TỌA ĐỘ VÀ NỘI SUY
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

# ==========================================
# 5. AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU)
# ==========================================
def augment(seq):
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
# 6. QUY TRÌNH NẠP DỮ LIỆU TÍCH HỢP BỘ NHỚ ĐỆM (CACHING)
# ==========================================
CACHE_X_PATH = os.path.join(MODEL_DIR, "X_cached.npy")
CACHE_Y_PATH = os.path.join(MODEL_DIR, "y_cached.npy")

# Kiểm tra sự tồn tại của bộ nhớ đệm trên ổ cứng
if os.path.exists(CACHE_X_PATH) and os.path.exists(CACHE_Y_PATH):
    print(f"\n📦 [BỘ NHỚ ĐỆM] Phát hiện dữ liệu tọa độ đã lưu từ trước!")
    print(f"   Đang nạp nhanh tập dữ liệu từ ổ cứng...", flush=True)
    X = np.load(CACHE_X_PATH)
    Y = np.load(CACHE_Y_PATH)
    print(f"✅ Nạp bộ nhớ đệm thành công — Tập dữ liệu: {X.shape}", flush=True)
else:
    # Nếu chưa có bộ nhớ đệm, tiến hành trích xuất MediaPipe từ đầu
    all_videos = []
    for label in unique_labels:
        folder_path = os.path.join(VIDEOS_UPDATE, label)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".mp4"):
                all_videos.append((label, os.path.join(folder_path, fname), fname))

    total = len(all_videos)
    X_list, Y_list = [], []
    print(f"\n🔄 Không tìm thấy bộ nhớ đệm. Bắt đầu trích xuất {total} video bằng MediaPipe...", flush=True)
    ok = 0

    for idx, (label, vpath, fname) in enumerate(all_videos):
        label_id = label_to_id[label]
        print(f"  [{idx+1}/{total}] {label}/{fname}", end=" ", flush=True)

        raw = extract_features(vpath)
        if raw is None:
            print("❌ (Video lỗi hoặc ngắn)", flush=True); continue

        base = normalize_and_interpolate(raw)
        X_list.append(base)
        Y_list.append(label_id)
        
        for _ in range(AUGMENT_COUNT):
            X_list.append(augment(base))
            Y_list.append(label_id)

        ok += 1
        print("✅ (OK)", flush=True)

    X = np.array(X_list, np.float32)
    Y = np.array(Y_list, np.int32)
    print(f"\n✅ Hoàn tất xử lý: {ok}/{total} video — Tập dữ liệu: {X.shape}", flush=True)
    
    # Tiến hành lưu lại bộ nhớ đệm ra file cứng để tái sử dụng nhanh lần sau
    print("💾 Đang ghi bộ nhớ đệm ra ổ cứng để bảo toàn dữ liệu...", flush=True)
    np.save(CACHE_X_PATH, X)
    np.save(CACHE_Y_PATH, Y)
    print("✅ Đã lưu bộ nhớ đệm thành công!", flush=True)

# Phân chia dữ liệu train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ==========================================
# 7. XÂY DỰNG MÔ HÌNH BiLSTM
# ==========================================
print("\n🧠 ĐANG KHỞI TẠO CẤU TRÚC MÔ HÌNH BiLSTM (345 ĐẶC TRƯNG)...", flush=True)

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
print("\n⚡ BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH TRÊN GPU...", flush=True)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150, batch_size=32,
    callbacks=callbacks, verbose=1,
)

# ==========================================
# 9. ĐÁNH GIÁ & XUẤT MÔ HÌNH
# ==========================================
best = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_bilstm.keras"))
val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Kết quả tốt nhất — val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}", flush=True)

best.export(os.path.join(MODEL_DIR, "vsl_bilstm_savedmodel"))
print(f"\n🎉 HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN!", flush=True)
print(f"   Mô hình lưu tại : {os.path.join(MODEL_DIR, 'best_bilstm.keras')}", flush=True)
print(f"   Bản đồ nhãn     : {os.path.join(MODEL_DIR, 'label_map.json')}", flush=True)
print(f"   Nhật ký huấn luyện: {os.path.join(MODEL_DIR, 'training_log.csv')}", flush=True)