import os
import sys
import io
import json
import numpy as np
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional

# Đồng bộ hiển thị tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Cấu hình hệ thống ────────────────────────────────────────────────────────
BASE_DIR       = r"C:\Users\MYPC\Downloads\Dataset"
MODEL_PATH     = os.path.join(BASE_DIR, "Models", "best_bilstm.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "Models", "label_map.json")

N_FRAMES        = 30     # Kích thước cửa sổ trượt
N_FEATURES      = 345    # 345 đặc trưng
STRIDE          = 5      # Bước nhảy trượt
CONF_THRESHOLD  = 0.65   # Ngưỡng tin cậy chấp nhận nhãn
ENERGY_THRESHOLD= 0.03   # Ngưỡng lọc vùng tĩnh không di chuyển tay

# Tham số kết hợp cải tiến từ 2 bài báo
VOTING_BAG_SIZE = 7      # Quy mô hàng đợi bỏ phiếu (CVPR 2024)
MIN_VOTES       = 4      # Ngưỡng phiếu tối thiểu để chấp nhận nhãn (> B/2)
VALLEY_THRESHOLD = 0.30  # Độ sâu sụt giảm xác suất tối thiểu để tách từ trùng (ESWA 2024)
MAX_GAP_WINDOWS = 3      # Khóa khoảng trống Blank tối đa để gộp từ ký chậm (ESWA 2024)

LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95
]

# ── Tải mô hình và nhãn ──────────────────────────────────────────────────────
def load_model_and_labels(model_path: str = MODEL_PATH, label_map_path: str = LABEL_MAP_PATH):
    import tensorflow as tf
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")
    model = tf.keras.models.load_model(model_path)
    with open(label_map_path, encoding='utf-8') as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    return model, id_to_label

# ── Trích xuất đặc trưng MediaPipe ──────────────────────────────────────────
mp_holistic = mp.solutions.holistic

def extract_all_keypoints(video_path: str) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
    return np.array(frames, np.float32), fps

def normalize_frame(kp: np.ndarray) -> np.ndarray:
    kps = kp.reshape(-1, 3)
    ls, rs = kps[11], kps[12]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm(rs - ls)
    if scale < 1e-6:
        scale = 1.0
    return ((kps - center) / scale).flatten().astype(np.float32)

def compute_hand_energy(seq: np.ndarray) -> float:
    hand = seq[:, 99:225]
    if len(hand) < 2:
        return 0.0
    diffs = np.diff(hand, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))

# ── Dự đoán song song cửa sổ trượt ──────────────────────────────────────────
def sliding_window_predict(
    keypoints: np.ndarray,
    model,
    id_to_label: dict,
    stride: int = STRIDE,
    conf_threshold: float = CONF_THRESHOLD,
    energy_threshold: float = ENERGY_THRESHOLD,
) -> List[Dict]:
    T = len(keypoints)
    results = []

    normed_all = np.array([normalize_frame(kp) for kp in keypoints], np.float32)

    windows = []
    window_meta = []
    w_idx = 0

    for start in range(0, T - N_FRAMES + 1, stride):
        end = start + N_FRAMES
        win = normed_all[start:end]

        energy = compute_hand_energy(win)
        accepted = energy >= energy_threshold

        windows.append(win)
        window_meta.append({
            "window_idx": w_idx,
            "frame_start": start,
            "frame_end": end - 1,
            "energy": round(energy, 4),
            "accepted": accepted,
        })
        w_idx += 1

    if not windows:
        return []

    batch = np.array(windows, np.float32)
    probs = model.predict(batch, verbose=0, batch_size=64)

    for i, meta in enumerate(window_meta):
        pred_id = int(np.argmax(probs[i]))
        pred_conf = float(probs[i][pred_id])
        label = id_to_label[pred_id]

        if not meta["accepted"] or pred_conf < conf_threshold:
            meta["accepted"] = False
            label = None
            pred_conf = 0.0

        results.append({
            **meta,
            "label": label,
            "conf": round(pred_conf, 4),
        })

    return results

# ── THUẬT TOÁN LAI THẾ HỆ MỚI (CÓ GAP SUPPRESSION) ───────────────────────────
def merge_windows_hybrid(
    window_results: List[Dict],
    bag_size: int = VOTING_BAG_SIZE,
    min_votes: int = MIN_VOTES,
    valley_thresh: float = VALLEY_THRESHOLD,
    max_gap: int = MAX_GAP_WINDOWS,
) -> List[Dict]:
    """
    Thuật toán lai nâng cấp:
    1. Ổn định hóa nhãn đầu ra bằng phương pháp Bỏ phiếu đa số (CVPR 2024).
    2. Gom phân đoạn có khoảng trống ngắn (Gap Suppression - ESWA 2024) để tránh tách đôi từ khi ký chậm.
    3. Định vị đỉnh xác suất cao nhất (Peak-Valley - ESWA 2024) để tách từ lặp thực sự.
    """
    if not window_results:
        return []

    # BƯỚC 1: Áp dụng hàng đợi bỏ phiếu đa số (CVPR 2024) để lọc nhiễu nháy nhãn
    voted_series = []
    bag = []

    for r in window_results:
        label = r["label"] if (r["accepted"] and r["label"]) else None
        bag.append(label)
        
        if len(bag) > bag_size:
            bag.pop(0)

        pv = None
        if len(bag) == bag_size:
            counts = {}
            for item in bag:
                if item is not None:
                    counts[item] = counts.get(item, 0) + 1
            
            if counts:
                best_label = max(counts, key=counts.get)
                if counts[best_label] >= min_votes:
                    pv = best_label

        voted_series.append({
            "window_idx" : r["window_idx"],
            "frame_start": r["frame_start"],
            "frame_end"  : r["frame_end"],
            "label"      : pv,
            "conf"       : r["conf"] if pv is not None else 0.0
        })

    # BƯỚC 2: Phân nhóm liên tiếp tích hợp Gap Suppression chống lỗi người ký chậm
    raw_segments = []
    current = None
    gap_counter = 0

    for item in voted_series:
        lbl = item["label"]
        
        if lbl is None:
            if current is not None:
                gap_counter += 1
                if gap_counter > max_gap:
                    # Khoảng tĩnh thực sự kéo dài -> Đóng phân đoạn từ hiện tại
                    raw_segments.append(current)
                    current = None
                    gap_counter = 0
            continue

        # Khi xuất hiện nhãn hợp lệ mới
        if current is None:
            current = {
                "label": lbl,
                "items": [item]
            }
            gap_counter = 0
        elif current["label"] == lbl:
            # Gộp nhãn trùng ngay cả khi ở giữa có các cửa sổ tĩnh ngắn (Gap Lock)
            current["items"].append(item)
            gap_counter = 0
        else:
            # Gặp nhãn mới hoàn toàn -> Đóng từ cũ ngay để mở phân đoạn mới
            raw_segments.append(current)
            current = {
                "label": lbl,
                "items": [item]
            }
            gap_counter = 0

    if current is not None:
        raw_segments.append(current)

    # BƯỚC 3: Định vị Đỉnh (Peak) và Phân tách Thung lũng (Valley - ESWA 2024)
    final_segments = []
    seg_idx = 0

    for raw_seg in raw_segments:
        label = raw_seg["label"]
        items = raw_seg["items"]
        
        if not items:
            continue

        confs = [it["conf"] for it in items]
        peak_idx = int(np.argmax(confs))
        peak_val = confs[peak_idx]
        
        # Dò tìm thung lũng (Valley) để xem từ đó có thực sự bị lặp lại không
        split_points = []
        for i in range(1, len(confs) - 1):
            if confs[i] < confs[i-1] and confs[i] < confs[i+1]:
                if (peak_val - confs[i]) > valley_thresh:
                    split_points.append(i)

        if split_points:
            parts = []
            prev_idx = 0
            for sp in split_points:
                parts.append(items[prev_idx:sp])
                prev_idx = sp
            parts.append(items[prev_idx:])
        else:
            parts = [items]

        for part in parts:
            part_confs = [p["conf"] for p in part]
            p_idx = int(np.argmax(part_confs))
            
            final_segments.append({
                "segment_idx": seg_idx,
                "label"      : label,
                "frame_start": part[0]["frame_start"],
                "frame_end"  : part[-1]["frame_end"],
                "peak_frame" : part[p_idx]["frame_start"] + (N_FRAMES // 2),
                "avg_conf"   : round(float(np.mean(part_confs)), 4),
                "max_conf"   : round(float(part_confs[p_idx]), 4),
            })
            seg_idx += 1

    return final_segments

# ── API chạy nhận diện chính ───────────────────────────────────────────────
def recognize_video(
    video_path: str,
    model=None,
    id_to_label: Optional[dict] = None,
    stride: int = STRIDE,
    conf_threshold: float = CONF_THRESHOLD,
    energy_threshold: float = ENERGY_THRESHOLD,
    verbose: bool = True,
) -> Dict:
    if model is None or id_to_label is None:
        if verbose:
            print("Đang nạp mô hình và bản đồ nhãn...")
        model, id_to_label = load_model_and_labels()

    if verbose:
        print(f"Bắt đầu xử lý: {os.path.basename(video_path)}")

    keypoints, fps = extract_all_keypoints(video_path)
    T = len(keypoints)

    if verbose:
        print(f"  -> Tổng: {T} khung hình | FPS: {fps:.2f}")

    window_results = sliding_window_predict(
        keypoints, model, id_to_label,
        stride=stride,
        conf_threshold=conf_threshold,
        energy_threshold=energy_threshold,
    )

    # Khởi chạy bộ lọc lai nâng cấp chống nháy nhãn và khóa khoảng trống từ lặp
    segments = merge_windows_hybrid(
        window_results,
        bag_size=VOTING_BAG_SIZE,
        min_votes=MIN_VOTES,
        valley_thresh=VALLEY_THRESHOLD,
        max_gap=MAX_GAP_WINDOWS
    )

    gloss_sequence = [s["label"] for s in segments]

    if verbose:
        print(f"\n===== KẾT QUẢ PHÂN ĐOẠN LAI KHỬ NHIỄU (CẢI TIẾN) =====")
        for s in segments:
            t_start = s["frame_start"] / fps
            t_end = s["frame_end"] / fps
            t_peak = s["peak_frame"] / fps
            print(f"  [{s['segment_idx']+1}] {s['label']:20s} "
                  f"Mốc thời gian: {t_start:.1f}s - {t_end:.1f}s | "
                  f"Điểm cực đại (Peak): {t_peak:.1f}s (conf={s['max_conf']:.2f})")
        print(f"\nChuỗi ký hiệu chuẩn (Gloss Sequence): {' -> '.join(gloss_sequence)}")

    return {
        "video": os.path.basename(video_path),
        "total_frames": T,
        "fps": fps,
        "segments": segments,
        "gloss_sequence": gloss_sequence,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Sign Language Recognition (Hybrid Upgrade)")
    parser.add_argument("video", help="Đường dẫn đến tệp video cần nhận diện")
    args = parser.parse_args()

    recognize_video(args.video)