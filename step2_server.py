"""
BƯỚC 2 — FASTAPI SERVER + WEB DEMO REALTIME TÍCH HỢP LLM
Chạy: uvicorn step2_server:app --host 0.0.0.0 --port 8000
Mở trình duyệt: http://localhost:8000
"""

import os
import json
import time
import base64
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Nhúng module dịch thuật LLM từ step3
try:
    from step3_llm import translate_gloss_to_text
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠ Không tìm thấy module step3_llm.py, hệ thống sẽ chạy không có LLM.")

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
# Tính toán đường dẫn gốc (lùi ra 1 cấp từ thư mục Code)
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.dirname(CURRENT_DIR)
MODEL_PATH    = os.path.join(BASE_DIR, "Models", "best_bilstm.keras")
LABEL_MAP     = os.path.join(BASE_DIR, "Models", "label_map.json")
TEMPLATE_DIR  = os.path.join(CURRENT_DIR, "templates")

N_FRAMES       = 30
N_FEATURES     = 345   # Pose(99) + LH(63) + RH(63) + Lips(120)
CONF_THRESHOLD = 0.75

LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95
]

# ── Nạp Model & Label Map ──────────────────────────────────────────────────
print("⏳ Đang tải mô hình nhận diện hành động...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP, encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
print(f"✅ Mô hình sẵn sàng — Nhận diện {len(id_to_label)} nhãn — 345 Đặc trưng")

# ── Khởi tạo MediaPipe ─────────────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

holistic_instance = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Chuẩn hóa khung hình ───────────────────────────────────────────────────
def normalize_frame(frame_features: np.ndarray) -> np.ndarray:
    kps    = frame_features.reshape(-1, 3)
    ls, rs = kps[11], kps[12]
    center = (ls + rs) / 2.0
    scale  = np.linalg.norm(rs - ls)
    if scale < 1e-6:
        scale = 1.0
    return ((kps - center) / scale).flatten().astype(np.float32)

# ── Quản lý Trạng thái ──────────────────────────────────────────────────────
class SignState:
    def __init__(self):
        self.buffer           = deque(maxlen=N_FRAMES)
        self.velocity         = deque(maxlen=10)
        self.gloss_seq        = []
        self.last_gloss       = ""
        self.last_conf        = 0.0
        self.fps_times        = deque(maxlen=30)
        self.current_sentence = "Đang chờ ký hiệu..."
        self.is_translating   = False

# Chạy dịch LLM trên luồng phụ để không chặn Video Stream
def background_translate(gloss_seq_copy, state: SignState):
    if not LLM_AVAILABLE or len(gloss_seq_copy) == 0:
        state.is_translating = False
        return
    try:
        # Gọi hàm dịch của Step 3
        text, _ = translate_gloss_to_text(gloss_seq_copy, use_gpu=True, skip_exact_match=False)
        state.current_sentence = text
    except Exception as e:
        state.current_sentence = f"Lỗi dịch: {e}"
    finally:
        state.is_translating = False

# ── Thiết lập FastAPI ──────────────────────────────────────────────────────
app = FastAPI(title="VSL Realtime Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory=TEMPLATE_DIR)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── Vòng lặp WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = SignState()
    prev_kp: Optional[np.ndarray] = None
    prev_lips_kp: Optional[np.ndarray] = None

    signing_frames    = 0
    silence_frames    = 0
    current_state     = "REST"
    SIGNING_THRESHOLD = 0.04
    CONFIRM_FRAMES    = 6

    try:
        while True:
            t0   = time.perf_counter()
            data = await ws.receive()

            if "text" in data:
                cmd = json.loads(data["text"])
                if cmd.get("action") == "clear_gloss":
                    state.gloss_seq = []
                    state.last_gloss = ""
                    state.current_sentence = "Đang chờ ký hiệu..."
                    state.is_translating = False
                continue

            # Đọc khung hình
            jpg   = np.frombuffer(data["bytes"], np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic_instance.process(rgb)

            # Vẽ Landmarks
            frame.flags.writeable = True
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style())
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            lips_detected = False
            if res.face_landmarks:
                lips_detected = True
                for idx in LIPS_INDICES:
                    lm = res.face_landmarks.landmark[idx]
                    cx = int(lm.x * 640); cy = int(lm.y * 480)
                    cv2.circle(frame, (cx, cy), 2, (0, 215, 255), -1)

            # Trích xuất 345 điểm
            pose = np.zeros((33, 3), np.float32)
            lh   = np.zeros((21, 3), np.float32)
            rh   = np.zeros((21, 3), np.float32)
            lips = np.zeros((40, 3), np.float32)

            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark): pose[i] = [lm.x, lm.y, lm.z]
            if res.left_hand_landmarks:
                for i, lm in enumerate(res.left_hand_landmarks.landmark): lh[i] = [lm.x, lm.y, lm.z]
            if res.right_hand_landmarks:
                for i, lm in enumerate(res.right_hand_landmarks.landmark): rh[i] = [lm.x, lm.y, lm.z]
            if res.face_landmarks:
                for i, idx in enumerate(LIPS_INDICES):
                    lm = res.face_landmarks.landmark[idx]
                    lips[i] = [lm.x, lm.y, lm.z]

            raw_kp  = np.concatenate([pose, lh, rh, lips]).flatten()
            norm_kp = normalize_frame(raw_kp)
            state.buffer.append(norm_kp)

            # Tính năng lượng chuyển động
            energy_hand = 0.0
            if prev_kp is not None:
                energy_hand = float(np.linalg.norm(norm_kp[99:225] - prev_kp[99:225]))
            prev_kp = norm_kp.copy()
            state.velocity.append(energy_hand)
            avg_energy_hand = float(np.mean(state.velocity))

            energy_lips = 0.0
            lips_flat = norm_kp[225:]
            if prev_lips_kp is not None:
                energy_lips = float(np.linalg.norm(lips_flat - prev_lips_kp))
            prev_lips_kp = lips_flat.copy()

            # State Machine Nhận Diện Hành Động
            label, conf, is_new = state.last_gloss, state.last_conf, False

            if avg_energy_hand > SIGNING_THRESHOLD:
                signing_frames += 1
                silence_frames  = 0
                current_state   = "SIGNING"
            else:
                silence_frames += 1
                if current_state == "SIGNING" and silence_frames >= CONFIRM_FRAMES:
                    current_state = "CONFIRMED"
                    if len(state.buffer) >= N_FRAMES:
                        seq   = np.array(list(state.buffer)[-N_FRAMES:], np.float32)
                        seq   = seq[np.newaxis, ...]
                        probs = model.predict(seq, verbose=0)[0]
                        pid   = int(np.argmax(probs))
                        pconf = float(probs[pid])
                        
                        if pconf >= CONF_THRESHOLD:
                            label = id_to_label[pid]
                            conf  = pconf
                            if label != state.last_gloss:
                                state.gloss_seq.append(label)
                                state.last_gloss = label
                                state.last_conf  = conf
                                is_new = True
                                
                                # KHI CÓ TỪ MỚI -> GỌI LLM DỊCH TRÊN LUỒNG PHỤ
                                state.is_translating = True
                                threading.Thread(target=background_translate, args=(state.gloss_seq.copy(), state)).start()
                                
                    signing_frames = 0
                elif silence_frames > CONFIRM_FRAMES * 2:
                    current_state = "REST"

            # FPS & Gửi luồng về Web
            state.fps_times.append(time.perf_counter())
            fps = 0.0
            if len(state.fps_times) >= 2:
                fps = (len(state.fps_times) - 1) / (state.fps_times[-1] - state.fps_times[0] + 1e-9)

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            _, jpg_out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64  = base64.b64encode(jpg_out.tobytes()).decode()

            await ws.send_text(json.dumps({
                "frame"         : frame_b64,
                "state"         : current_state,
                "label"         : label,
                "conf"          : round(conf, 4),
                "is_new"        : is_new,
                "gloss_seq"     : state.gloss_seq[-10:],
                "sentence"      : state.current_sentence,
                "is_translating": state.is_translating,
                "energy_hand"   : round(avg_energy_hand, 4),
                "energy_lips"   : round(energy_lips, 4),
                "lips_detected" : lips_detected,
                "fps"           : round(fps, 1),
                "latency_ms"    : latency_ms,
            }))

    except WebSocketDisconnect:
        pass
    finally:
        print("📴 Client ngắt kết nối")

if __name__ == "__main__":
    import uvicorn
    # Bạn hãy chạy lệnh "python Code/step2_server.py" từ thư mục Dataset
    uvicorn.run("step2_server:app", host="0.0.0.0", port=8000, reload=False)