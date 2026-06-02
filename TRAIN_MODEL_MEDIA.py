import os
import cv2
import random
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import sys
import traceback
from typing import Optional

# ==========================================
# 0. ENCODING & SEED
# ==========================================
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def custom_excepthook(exc_type, exc_value, exc_tb):
    with open("error_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        f.flush()

sys.excepthook = custom_excepthook

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
BASE_DIR      = r"C:\Users\MYPC\Downloads\Dataset"
VIDEOS_UPDATE = os.path.join(BASE_DIR, "videos_update")
MODEL_DIR     = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

N_FRAMES      = 30
AUGMENT_COUNT = 30
N_FEATURES    = 345   # Pose(99) + LH(63) + RH(63) + Lips(120)

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

# Cấu hình GPU
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
print("🚀 ĐANG KHỞI ĐỘNG PIPELINE HUẤN LUYỆN VSL")

# ==========================================
# 2. QUÉT FOLDER & TẠO LABEL MAP
# ==========================================
if not os.path.exists(VIDEOS_UPDATE):
    print(f"❌ Không tìm thấy thư mục: {VIDEOS_UPDATE}")
    sys.exit(1)

unique_labels = sorted([
    d for d in os.listdir(VIDEOS_UPDATE)
    if os.path.isdir(os.path.join(VIDEOS_UPDATE, d))
])
NUM_CLASSES = len(unique_labels)
label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
id_to_label = {i: lbl for lbl, i in label_to_id.items()}

label_map_path = os.path.join(MODEL_DIR, "label_map.json")
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=2)

print(f"📊 Tìm thấy {NUM_CLASSES} nhãn — Đã cập nhật label_map.json", flush=True)

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

# ==========================================
# 5. AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU TỐI ƯU HÓA)
# ==========================================
def augment(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()

    # Thêm nhiễu Gaussian cực nhẹ để chống ghi nhớ tọa độ tĩnh
    if np.random.rand() < 0.4:  # Giảm xác suất từ 0.6 xuống 0.4
        aug += np.random.normal(0, 0.005, aug.shape).astype(np.float32)  # Giảm biên độ nhiễu

    # Scale ngẫu nhiên nhẹ nhàng
    if np.random.rand() < 0.4:
        aug *= np.random.uniform(0.95, 1.05)  # Giảm biên độ co giãn xuống ±5% (thay vì ±12%)

    # Lật ngang (horizontal flip) + hoán đổi tay trái/phải
    if np.random.rand() < 0.5:
        # Lật x của body (pose + lh + rh)
        body = aug[:, :225].reshape(N_FRAMES, 75, 3).copy()
        body[:, :, 0] *= -1
        # Hoán đổi LH (index 33-53) ↔ RH (index 54-74) trong pose+hands block
        lh_tmp = body[:, 33:54].copy()
        body[:, 33:54] = body[:, 54:75]
        body[:, 54:75] = lh_tmp
        aug[:, :225] = body.reshape(N_FRAMES, 225)

        # Lật x của lips
        lips = aug[:, 225:].reshape(N_FRAMES, 40, 3).copy()
        lips[:, :, 0] *= -1
        aug[:, 225:] = lips.reshape(N_FRAMES, 120)

    # Loại bỏ hoàn toàn Time Shift (Dịch thời gian) và Time Warp (Co giãn thời gian)
    # Vì hai phép này phá vỡ tính nhịp điệu của cử chỉ trên tập dữ liệu quá nhỏ.

    return aug.astype(np.float32)
# ==========================================
# 6. KIỂM TRA CACHE HỢP LỆ
# ==========================================
CACHE_X_TRAIN    = os.path.join(MODEL_DIR, "X_train_cached.npy")
CACHE_Y_TRAIN    = os.path.join(MODEL_DIR, "y_train_cached.npy")
CACHE_X_VAL      = os.path.join(MODEL_DIR, "X_val_cached.npy")
CACHE_Y_VAL      = os.path.join(MODEL_DIR, "y_val_cached.npy")
CACHE_META_PATH  = os.path.join(MODEL_DIR, "cache_meta.json")

def _cache_is_valid() -> bool:
    """
    Cache hợp lệ khi:
    1. Tất cả file .npy tồn tại
    2. Số lớp trong cache khớp với NUM_CLASSES hiện tại
    3. Danh sách nhãn trong cache khớp với unique_labels hiện tại
    """
    required = [CACHE_X_TRAIN, CACHE_Y_TRAIN, CACHE_X_VAL, CACHE_Y_VAL, CACHE_META_PATH]
    if not all(os.path.exists(p) for p in required):
        return False
    try:
        with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("num_classes") != NUM_CLASSES:
            print("⚠️  Cache lỗi thời: số lớp thay đổi — tạo lại cache...", flush=True)
            return False
        if meta.get("labels") != unique_labels:
            print("⚠️  Cache lỗi thời: danh sách nhãn thay đổi — tạo lại cache...", flush=True)
            return False
    except Exception:
        return False
    return True
# ==========================================
# 7. NẠP HOẶC XÂY DỰNG DỮ LIỆU CHUẨN KHOA HỌC (SPLIT BEFORE AUGMENTATION)
# ==========================================
if _cache_is_valid():
    print(f"\n📦 [CACHE] Phát hiện cache Train/Val hợp lệ — đang nạp...", flush=True)
    X_train = np.load(CACHE_X_TRAIN)
    y_train = np.load(CACHE_Y_TRAIN)
    X_val   = np.load(CACHE_X_VAL)
    y_val   = np.load(CACHE_Y_VAL)
    print(f"✅ Nạp cache thành công — Train: {X_train.shape} | Val: {X_val.shape}", flush=True)
else:
    all_videos = []
    for label in unique_labels:
        folder_path = os.path.join(VIDEOS_UPDATE, label)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".mp4"):
                all_videos.append((label, os.path.join(folder_path, fname), fname))

    if len(all_videos) == 0:
        print("❌ Không tìm thấy video nào trong dataset!")
        sys.exit(1)

    # 2. Chia Train/Val trên video gốc (TRƯỚC khi augment) - gỡ bỏ stratify chống ValueError
    train_videos, val_videos = train_test_split(
        all_videos,
        test_size=0.2,
        random_state=SEED
    )
    
    total_train = len(train_videos)
    total_val   = len(val_videos)
    print(f"\n📂 Tổng video: {len(all_videos)} | Train gốc: {total_train} | Val gốc: {total_val}", flush=True)

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    # 3. Trích xuất + Augment trên tập Train (Dùng enumerate chuẩn hóa Windows)
    print(f"\n🔄 Trích xuất & Tăng cường tập Train...", flush=True)
    skipped_train = 0
    for idx, (label, vpath, fname) in enumerate(train_videos):
        label_id = label_to_id[label]
        
        # In tiến độ tức thời
        print(f"  [Train {idx+1}/{total_train}] {label}/{fname}", end=" ", flush=True)
        
        raw = extract_features(vpath)
        if raw is None:
            print("❌ (Lỗi)", flush=True)
            skipped_train += 1
            continue
            
        base = normalize_and_interpolate(raw)
        X_train_list.append(base)
        y_train_list.append(label_id)
        
        # Nhân bản tập Train
        for _ in range(AUGMENT_COUNT):
            X_train_list.append(augment(base))
            y_train_list.append(label_id)
            
        print("✅ (OK)", flush=True)

    if skipped_train > 0:
        print(f"⚠️  Bỏ qua {skipped_train} video lỗi trong tập Train", flush=True)

    # 4. Trích xuất sạch tập Val (KHÔNG augment - Dùng enumerate chuẩn hóa Windows)
    print(f"\n🔄 Trích xuất sạch tập Validation...", flush=True)
    skipped_val = 0
    for idx, (label, vpath, fname) in enumerate(val_videos):
        label_id = label_to_id[label]
        
        # In tiến độ tức thời
        print(f"  [Val {idx+1}/{total_val}] {label}/{fname}", end=" ", flush=True)
        
        raw = extract_features(vpath)
        if raw is None:
            print("❌ (Lỗi)", flush=True)
            skipped_val += 1
            continue
            
        base = normalize_and_interpolate(raw)
        X_val_list.append(base)
        y_val_list.append(label_id)
        
        print("✅ (OK)", flush=True)

    if skipped_val > 0:
        print(f"⚠️  Bỏ qua {skipped_val} video lỗi trong tập Val", flush=True)

    if len(X_train_list) == 0 or len(X_val_list) == 0:
        print("❌ Không đủ dữ liệu để huấn luyện!")
        sys.exit(1)

    X_train = np.array(X_train_list, np.float32)
    y_train = np.array(y_train_list, np.int32)
    X_val   = np.array(X_val_list,   np.float32)
    y_val   = np.array(y_val_list,   np.int32)

    # 5. Lưu cache kèm metadata
    np.save(CACHE_X_TRAIN, X_train)
    np.save(CACHE_Y_TRAIN, y_train)
    np.save(CACHE_X_VAL,   X_val)
    np.save(CACHE_Y_VAL,   y_val)
    cache_meta = {"num_classes": NUM_CLASSES, "labels": unique_labels}
    with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã tạo cache thành công — Train: {X_train.shape} | Val: {X_val.shape}", flush=True)

# ==========================================
# 8. CLASS WEIGHT (XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU TÙY CHỈNH)
# ==========================================
class_weight_dict = {}
total_samples = len(y_train)
unique_classes, class_counts = np.unique(y_train, return_counts=True)
count_dict = dict(zip(unique_classes, class_counts))

for i in range(NUM_CLASSES):
    if i in count_dict:
        # Công thức chuẩn "balanced" của scikit-learn
        class_weight_dict[i] = total_samples / (NUM_CLASSES * count_dict[i])
    else:
        # Nếu nhãn vô tình bị vắng mặt trong tập Train, gán trọng số an toàn = 1.0
        class_weight_dict[i] = 1.0

print(f"\n⚖️ Đã tính toán xong Class Weights an toàn cho toàn bộ {len(class_weight_dict)} lớp.", flush=True)

# ==========================================
# 9. XÂY DỰNG MÔ HÌNH BiLSTM
# ==========================================
print("\n🧠 KHỞI TẠO MÔ HÌNH BiLSTM...", flush=True)

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

# ==========================================
# 10. CALLBACKS
# ==========================================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_bilstm.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,                  # Giảm từ 20 → 15 để tránh tốn thời gian
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "training_log.csv")),
]

# ==========================================
# 11. HUẤN LUYỆN
# ==========================================
print("\n⚡ BẮT ĐẦU HUẤN LUYỆN...", flush=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,   # ← FIX: xử lý mất cân bằng
    callbacks=callbacks,
    verbose=1,
)

# ==========================================
# 12. ĐÁNH GIÁ & XUẤT MÔ HÌNH
# ==========================================
best = tf.keras.models.load_model(BEST_MODEL_PATH)
val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Kết quả tốt nhất — val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}", flush=True)

# Lưu dạng .keras (fine-tune được)
final_keras_path = os.path.join(MODEL_DIR, "vsl_bilstm_final.keras")
best.save(final_keras_path)

# Xuất dạng SavedModel (inference/TFLite)
saved_model_path = os.path.join(MODEL_DIR, "vsl_bilstm_savedmodel")
best.export(saved_model_path)

print(f"\n🎉 HOÀN TẤT!", flush=True)
print(f"   Model (.keras)    : {final_keras_path}", flush=True)
print(f"   Model (SavedModel): {saved_model_path}", flush=True)
print(f"   Label map         : {label_map_path}", flush=True)
print(f"   Training log      : {os.path.join(MODEL_DIR, 'training_log.csv')}", flush=True)