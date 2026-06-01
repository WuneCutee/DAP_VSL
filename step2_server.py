"""
BƯỚC 2 — FASTAPI SERVER + WEB DEMO REALTIME
Features: Pose + Hand + Lips (345 keypoints)
Chạy: uvicorn step2_server:app --host 0.0.0.0 --port 8000
Mở:   http://localhost:8000
"""

import os, json, time, base64
from collections import deque
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Cấu hình ──────────────────────────────────────────────────────────────
MODEL_PATH     = r"C:\Users\MYPC\Downloads\Dataset\Models\best_bilstm.keras"
LABEL_MAP_PATH = r"C:\Users\MYPC\Downloads\Dataset\Models\label_map.json"
N_FRAMES       = 30
N_FEATURES     = 345   # Pose(99) + LH(63) + RH(63) + Lips(120)
CONF_THRESHOLD = 0.75

# 40 điểm lips trong face mesh MediaPipe
LIPS_INDICES = [
    61,185,40,39,37,0,267,269,270,409,291,375,
    321,405,314,17,84,181,91,146,
    78,191,80,81,82,13,312,311,310,415,
    308,324,318,402,317,14,87,178,88,95
]

# ── Load model & label map ─────────────────────────────────────────────────
print("⏳ Đang tải model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
print(f"✅ Model sẵn sàng — {len(id_to_label)} nhãn — 345 features (Pose+Hand+Lips)")

# ── MediaPipe ──────────────────────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

holistic_instance = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Shoulder normalization (giống train) ───────────────────────────────────
def normalize_frame(frame_features: np.ndarray) -> np.ndarray:
    """
    frame_features: (345,) — Pose(99)+LH(63)+RH(63)+Lips(120)
    Normalize theo trung điểm vai và khoảng cách vai.
    """
    kps    = frame_features.reshape(-1, 3)   # (115, 3)
    ls, rs = kps[11], kps[12]               # vai trái, phải (Pose index)
    center = (ls + rs) / 2.0
    scale  = np.linalg.norm(rs - ls)
    if scale < 1e-6:
        scale = 1.0
    return ((kps - center) / scale).flatten().astype(np.float32)

# ── State per WebSocket connection ─────────────────────────────────────────
class SignState:
    def __init__(self):
        self.buffer    : deque = deque(maxlen=N_FRAMES)
        self.velocity  : deque = deque(maxlen=10)
        self.gloss_seq : list  = []
        self.last_gloss: str   = ""
        self.last_conf : float = 0.0
        self.fps_times : deque = deque(maxlen=30)

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="VSL Realtime Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Web Dashboard HTML ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>VSL Realtime Demo</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0f1117;color:#e2e8f0;font-family:'Segoe UI',sans-serif;
       display:flex;flex-direction:column;align-items:center;min-height:100vh}
  header{width:100%;background:#1a1d2e;padding:14px 24px;
         display:flex;align-items:center;gap:12px;border-bottom:1px solid #2d3250}
  header h1{font-size:1.2rem;font-weight:700;letter-spacing:.5px}
  .badge{font-size:.7rem;background:#4c9be8;padding:3px 10px;
         border-radius:99px;font-weight:600}
  .badge-green{background:#276749}
  .main{display:flex;gap:20px;padding:20px;width:100%;max-width:1280px;flex-wrap:wrap}

  /* Video */
  .video-panel{position:relative;flex:1 1 640px}
  #canvas{width:100%;border-radius:12px;background:#000;display:block}
  .overlay-badges{position:absolute;top:10px;left:10px;display:flex;gap:8px}
  .state-badge{padding:4px 12px;border-radius:99px;font-size:.75rem;font-weight:700}
  .state-REST     {background:#2d3250;color:#a0aec0}
  .state-SIGNING  {background:#276749;color:#9ae6b4}
  .state-CONFIRMED{background:#1a365d;color:#90cdf4}
  .fps-badge{position:absolute;top:10px;right:10px;background:rgba(0,0,0,.6);
             padding:3px 8px;border-radius:6px;font-size:.75rem}
  /* Lips indicator */
  .lips-badge{position:absolute;bottom:10px;left:10px;background:rgba(0,0,0,.6);
              padding:3px 8px;border-radius:6px;font-size:.72rem;color:#fbd38d}

  /* Side */
  .side{flex:0 0 320px;display:flex;flex-direction:column;gap:14px}
  .card{background:#1a1d2e;border-radius:12px;padding:16px;border:1px solid #2d3250}
  .card h3{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;
           color:#718096;margin-bottom:10px}

  /* Prediction */
  #pred-label{font-size:2.4rem;font-weight:800;color:#90cdf4;letter-spacing:1px;
              min-height:3rem;transition:all .2s}
  #pred-conf{font-size:.85rem;color:#a0aec0;margin-top:4px}
  .conf-bar{height:6px;background:#2d3250;border-radius:99px;margin-top:8px}
  #conf-fill{height:100%;background:#4c9be8;border-radius:99px;
             transition:width .3s;width:0%}

  /* Energy */
  .energy-row{display:flex;gap:8px;align-items:center;margin-top:6px}
  .energy-label{font-size:.7rem;color:#718096;min-width:32px}
  .energy-track{flex:1;height:7px;background:#2d3250;border-radius:99px}
  .energy-fill-hand{height:100%;background:#68d391;border-radius:99px;
                    transition:width .1s;width:0%}
  .energy-fill-lips{height:100%;background:#fbd38d;border-radius:99px;
                    transition:width .1s;width:0%}

  /* Gloss */
  #gloss-box{display:flex;flex-wrap:wrap;gap:6px;min-height:36px}
  .gloss-chip{background:#2d3250;border-radius:6px;padding:4px 10px;
              font-size:.8rem;font-weight:600}
  .gloss-chip.new{background:#2c5282;color:#bee3f8}

  /* Sentence */
  #sentence-box{font-size:1rem;line-height:1.6;color:#e2e8f0;min-height:2rem}

  /* Buttons */
  .btn{width:100%;padding:10px;border:none;border-radius:8px;
       font-size:.9rem;font-weight:600;cursor:pointer;transition:opacity .2s}
  .btn:hover{opacity:.85}
  .btn-start{background:#276749;color:#fff}
  .btn-stop {background:#742a2a;color:#fff}
  .btn-clear{background:#2d3250;color:#a0aec0}

  /* Metrics */
  .metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .metric{background:#0f1117;border-radius:8px;padding:10px;text-align:center}
  .metric .val{font-size:1.2rem;font-weight:700;color:#90cdf4}
  .metric .lbl{font-size:.68rem;color:#718096;margin-top:2px}
</style>
</head>
<body>
<header>
  <span>🤟</span>
  <h1>VSL Realtime Recognition</h1>
  <span class="badge">BiLSTM + MediaPipe</span>
  <span class="badge badge-green">Pose + Hand + Lips</span>
</header>

<div class="main">
  <!-- Video -->
  <div class="video-panel">
    <canvas id="canvas" width="640" height="480"></canvas>
    <div class="overlay-badges">
      <span id="state-badge" class="state-badge state-REST">REST</span>
    </div>
    <span class="fps-badge" id="fps-display">0 FPS</span>
    <span class="lips-badge" id="lips-status">👄 Lips: —</span>
  </div>

  <!-- Side -->
  <div class="side">
    <!-- Prediction -->
    <div class="card">
      <h3>Nhận diện hiện tại</h3>
      <div id="pred-label">—</div>
      <div id="pred-conf">Confidence: —</div>
      <div class="conf-bar"><div id="conf-fill"></div></div>
    </div>

    <!-- Energy -->
    <div class="card">
      <h3>Năng lượng chuyển động</h3>
      <div class="energy-row">
        <span class="energy-label">🖐 Tay</span>
        <div class="energy-track"><div class="energy-fill-hand" id="energy-hand"></div></div>
      </div>
      <div class="energy-row">
        <span class="energy-label">👄 Miệng</span>
        <div class="energy-track"><div class="energy-fill-lips" id="energy-lips"></div></div>
      </div>
    </div>

    <!-- Gloss -->
    <div class="card">
      <h3>Chuỗi gloss</h3>
      <div id="gloss-box"><span style="color:#4a5568;font-size:.8rem">Chưa có...</span></div>
    </div>

    <!-- Sentence -->
    <div class="card">
      <h3>Câu tiếng Việt (LLM)</h3>
      <div id="sentence-box" style="color:#4a5568">Chờ LLM tích hợp (Bước 3)...</div>
    </div>

    <!-- Controls -->
    <div class="card">
      <h3>Điều khiển</h3>
      <div style="display:flex;flex-direction:column;gap:8px">
        <button class="btn btn-start" id="btn-start" onclick="startCam()">▶ Bắt đầu</button>
        <button class="btn btn-stop"  id="btn-stop"  onclick="stopCam()" disabled>■ Dừng</button>
        <button class="btn btn-clear" onclick="clearGloss()">✕ Xoá chuỗi gloss</button>
      </div>
    </div>

    <!-- Metrics -->
    <div class="card">
      <h3>Thống kê</h3>
      <div class="metrics">
        <div class="metric"><div class="val" id="m-latency">—</div><div class="lbl">Latency (ms)</div></div>
        <div class="metric"><div class="val" id="m-total">0</div><div class="lbl">Ký hiệu nhận</div></div>
        <div class="metric"><div class="val" id="m-frames">0</div><div class="lbl">Frames xử lý</div></div>
        <div class="metric"><div class="val" id="m-conf-avg">—</div><div class="lbl">Conf TB</div></div>
      </div>
    </div>
  </div>
</div>

<script>
let ws, stream, animId;
let frameCount=0, totalSigns=0, confSum=0, confCnt=0;
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
let video;

async function startCam(){
  stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  video=document.createElement('video');
  video.srcObject=stream; video.play();
  document.getElementById('btn-start').disabled=true;
  document.getElementById('btn-stop').disabled=false;
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType='arraybuffer';
  ws.onmessage=handleMsg;
  ws.onopen=()=>loop();
}

function stopCam(){
  cancelAnimationFrame(animId);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(ws) ws.close();
  document.getElementById('btn-start').disabled=false;
  document.getElementById('btn-stop').disabled=true;
}

function clearGloss(){
  document.getElementById('gloss-box').innerHTML=
    '<span style="color:#4a5568;font-size:.8rem">Chưa có...</span>';
  if(ws&&ws.readyState===1) ws.send(JSON.stringify({action:'clear_gloss'}));
}

let lastFrameTime=0;
function loop(){
  animId=requestAnimationFrame(loop);
  const now=performance.now();
  if(now-lastFrameTime<66) return; // ~15 FPS
  lastFrameTime=now;
  if(!video||video.readyState<2) return;
  ctx.drawImage(video,0,0,640,480);
  canvas.toBlob(blob=>{
    if(ws&&ws.readyState===1){
      blob.arrayBuffer().then(buf=>ws.send(buf));
      frameCount++;
      document.getElementById('m-frames').textContent=frameCount;
    }
  },'image/jpeg',0.7);
}

function handleMsg(evt){
  const d=JSON.parse(evt.data);

  // Annotated frame
  if(d.frame){
    const img=new Image();
    img.onload=()=>ctx.drawImage(img,0,0,640,480);
    img.src='data:image/jpeg;base64,'+d.frame;
  }

  // State badge
  const badge=document.getElementById('state-badge');
  badge.textContent=d.state||'REST';
  badge.className='state-badge state-'+(d.state||'REST');

  // FPS
  if(d.fps) document.getElementById('fps-display').textContent=d.fps.toFixed(1)+' FPS';

  // Energy — hand + lips
  if(d.energy_hand!==undefined){
    document.getElementById('energy-hand').style.width=Math.min(d.energy_hand*200,100)+'%';
  }
  if(d.energy_lips!==undefined){
    const lipsPct=Math.min(d.energy_lips*500,100);
    document.getElementById('energy-lips').style.width=lipsPct+'%';
    document.getElementById('lips-status').textContent=
      '👄 Lips: '+(d.lips_detected?'✅':'—');
  }

  // Prediction
  if(d.label){
    document.getElementById('pred-label').textContent=d.label;
    document.getElementById('pred-conf').textContent=
      'Confidence: '+(d.conf*100).toFixed(1)+'%';
    document.getElementById('conf-fill').style.width=(d.conf*100)+'%';
    document.getElementById('m-latency').textContent=d.latency_ms||'—';
    if(d.is_new){
      totalSigns++; confSum+=d.conf; confCnt++;
      document.getElementById('m-total').textContent=totalSigns;
      document.getElementById('m-conf-avg').textContent=
        (confSum/confCnt*100).toFixed(1)+'%';
    }
  }

  // Gloss sequence
  if(d.gloss_seq&&d.gloss_seq.length>0){
    document.getElementById('gloss-box').innerHTML=
      d.gloss_seq.map((g,i)=>
        `<span class="gloss-chip${i===d.gloss_seq.length-1?' new':''}">${g}</span>`
      ).join('');
  }

  // LLM sentence
  if(d.sentence){
    document.getElementById('sentence-box').textContent=d.sentence;
    document.getElementById('sentence-box').style.color='#e2e8f0';
  }
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


# ── WebSocket inference loop ───────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = SignState()
    prev_kp: Optional[np.ndarray] = None
    prev_lips_kp: Optional[np.ndarray] = None

    signing_frames    = 0
    silence_frames    = 0
    current_state     = "REST"
    SIGNING_THRESHOLD = 0.04   # hand energy
    CONFIRM_FRAMES    = 6      # frames im lặng để trigger inference

    try:
        while True:
            t0   = time.perf_counter()
            data = await ws.receive()

            # Lệnh text từ client (clear gloss, v.v.)
            if "text" in data:
                cmd = json.loads(data["text"])
                if cmd.get("action") == "clear_gloss":
                    state.gloss_seq = []
                continue

            # Decode JPEG frame
            jpg   = np.frombuffer(data["bytes"], np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic_instance.process(rgb)

            # ── Vẽ landmarks lên frame ─────────────────────────────────────
            frame.flags.writeable = True
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style())
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # Vẽ lips points (màu vàng)
            lips_detected = False
            if res.face_landmarks:
                lips_detected = True
                for idx in LIPS_INDICES:
                    lm = res.face_landmarks.landmark[idx]
                    cx = int(lm.x * 640); cy = int(lm.y * 480)
                    cv2.circle(frame, (cx, cy), 2, (0, 215, 255), -1)

            # ── Extract keypoints (345) ────────────────────────────────────
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

            raw_kp  = np.concatenate([pose, lh, rh, lips]).flatten()
            norm_kp = normalize_frame(raw_kp)
            state.buffer.append(norm_kp)

            # ── Hand energy (velocity tay) ─────────────────────────────────
            energy_hand = 0.0
            if prev_kp is not None:
                # index 99:225 = LH + RH trong vector 345
                energy_hand = float(np.linalg.norm(norm_kp[99:225] - prev_kp[99:225]))
            prev_kp = norm_kp.copy()
            state.velocity.append(energy_hand)
            avg_energy_hand = float(np.mean(state.velocity))

            # ── Lips energy (chuyển động miệng) ───────────────────────────
            energy_lips = 0.0
            lips_flat   = norm_kp[225:]   # 120 features
            if prev_lips_kp is not None:
                energy_lips = float(np.linalg.norm(lips_flat - prev_lips_kp))
            prev_lips_kp = lips_flat.copy()

            # ── State machine ──────────────────────────────────────────────
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
                        seq   = seq[np.newaxis, ...]        # (1, 30, 345)
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
                    signing_frames = 0
                elif silence_frames > CONFIRM_FRAMES * 2:
                    current_state = "REST"

            # ── FPS ────────────────────────────────────────────────────────
            state.fps_times.append(time.perf_counter())
            fps = 0.0
            if len(state.fps_times) >= 2:
                fps = (len(state.fps_times) - 1) / (
                    state.fps_times[-1] - state.fps_times[0] + 1e-9)

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            # ── Encode & gửi ───────────────────────────────────────────────
            _, jpg_out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64  = base64.b64encode(jpg_out.tobytes()).decode()

            await ws.send_text(json.dumps({
                "frame"        : frame_b64,
                "state"        : current_state,
                "label"        : label,
                "conf"         : round(conf, 4),
                "is_new"       : is_new,
                "gloss_seq"    : state.gloss_seq[-10:],
                "energy_hand"  : round(avg_energy_hand, 4),
                "energy_lips"  : round(energy_lips, 4),
                "lips_detected": lips_detected,
                "fps"          : round(fps, 1),
                "latency_ms"   : latency_ms,
            }))

    except WebSocketDisconnect:
        pass
    finally:
        print("📴 Client ngắt kết nối")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step2_server:app", host="0.0.0.0", port=8000, reload=False)