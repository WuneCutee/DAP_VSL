import os
import sys
import io
import cv2
import random
import numpy as np
from typing import List, Dict

# Đồng bộ hiển thị tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\MYPC\Downloads\Dataset"
VIDEOS_UPDATE = os.path.join(BASE_DIR, "videos_update")

# ══════════════════════════════════════════════════════════════════════════
# PHẦN 1 — QUÉT VÀ THU THẬP DANH SÁCH TỪ VỰNG HIỆN CÓ
# ══════════════════════════════════════════════════════════════════════════
if not os.path.exists(VIDEOS_UPDATE):
    print(f"❌ Không tìm thấy thư mục: {VIDEOS_UPDATE}")
    sys.exit(1)

# Quét tất cả các nhãn (thư mục con) đang có sẵn
available_labels = sorted([
    d for d in os.listdir(VIDEOS_UPDATE) 
    if os.path.isdir(os.path.join(VIDEOS_UPDATE, d))
])

# Gom toàn bộ file video của từng nhãn vào dictionary
label_videos_map = {}
for label in available_labels:
    folder_path = os.path.join(VIDEOS_UPDATE, label)
    videos = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
    if videos:
        label_videos_map[label.lower().strip()] = {
            "original_label": label,
            "folder_path": folder_path,
            "video_files": videos
        }

print("=" * 60)
print(f"🎬 TRÌNH PHÁT GHÉP CÂU VSL — TỰ ĐỘNG PHÂN ĐOẠN")
print("=" * 60)
print(f"📊 Hệ thống đang quản lý: {len(label_videos_map)} từ vựng cử chỉ đơn lẻ.")
print(f"💡 Hướng dẫn: Nhập các từ cách nhau bằng dấu phẩy (,)")
print(f"   Ví dụ: chào, bạn, đi học, nhà trường, vui mừng")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# PHẦN 2 — ĐỘNG CƠ PHÁT VIDEO NỐI TIẾP (OPENCV PLAYER)
# ══════════════════════════════════════════════════════════════════════════
def play_sentence_videos(words_input: str):
    # Tách chuỗi nhập vào và chuẩn hóa
    words = [w.strip().lower() for w in words_input.split(",") if w.strip()]
    
    valid_playlist = []
    missing_words = []

    # Kiểm tra tính hợp lệ của từng từ trong câu
    for w in words:
        if w in label_videos_map:
            # Chọn ngẫu nhiên 1 video mẫu của từ đó để phát
            meta = label_videos_map[w]
            chosen_video = random.choice(meta["video_files"])
            full_path = os.path.join(meta["folder_path"], chosen_video)
            valid_playlist.append({
                "label": meta["original_label"],
                "video_path": full_path,
                "filename": chosen_video
            })
        else:
            missing_words.append(w)

    if missing_words:
        print(f"\n⚠ [BỎ QUA] Các từ này chưa có dữ liệu video: {missing_words}")
        print("   -> Bạn có thể kiểm tra lại chính tả hoặc thư mục con trong 'videos_update'.")

    if not valid_playlist:
        print("❌ Không có từ nào hợp lệ để phát!")
        return

    print(f"\n▶ Bắt đầu phát chuỗi gồm {len(valid_playlist)} cử chỉ liên tiếp:")
    print("   " + " -> ".join([item["label"] for item in valid_playlist]))
    print("   (Nhấn 'N' để bỏ qua từ hiện tại | Nhấn 'Q' để dừng phát toàn bộ)\n")

    # Khởi tạo cửa sổ OpenCV
    window_name = "VSL Continuous Playback Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    for idx, item in enumerate(valid_playlist):
        cap = cv2.VideoCapture(item["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = int(1000 / fps)

        print(f"  [{idx+1}/{len(valid_playlist)}] Đang phát: {item['label']} ({item['filename']})", flush=True)

        skip_word = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Vẽ thanh trạng thái (Status Bar) lên đầu video để dễ quan sát từ đang phát
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
            
            # Text 1: Từ đang phát
            cv2.putText(frame, f"DANG PHAT: {item['label'].upper()} ({idx+1}/{len(valid_playlist)})", 
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA)
            
            # Text 2: Phím điều khiển
            cv2.putText(frame, "Space/N: Next Word | Q: Quit", 
                        (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:  # Nhấn Q hoặc ESC để thoát toàn bộ
                print("\n■ Đã dừng phát bởi người dùng.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n') or key == ord(' '):  # Nhấn N hoặc Space để bỏ qua từ
                skip_word = True
                break

        cap.release()
        if skip_word:
            continue

    # Kết thúc câu phát, dừng màn hình 0.5 giây rồi đóng
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print("✅ Đã hoàn tất phát chuỗi câu.")

# ══════════════════════════════════════════════════════════════════════════
# VÒNG LẶP CHƯƠNG TRÌNH CHÍNH (INTERACTIVE LOOP)
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    while True:
        try:
            print("\n" + "-" * 60)
            user_input = input("Nhập chuỗi cử chỉ (hoặc gõ 'exit' để thoát): ").strip()
            
            if user_input.lower() == 'exit':
                print("Tạm biệt!")
                break
                
            if not user_input:
                continue
                
            play_sentence_videos(user_input)
            
        except KeyboardInterrupt:
            print("\nThoát chương trình.")
            break