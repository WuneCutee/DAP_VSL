"""
BƯỚC 3 — LLM SINH CÂU TIẾNG VIỆT + ĐO BLEU
Tích hợp vào step2_server.py sau khi chạy độc lập để lấy số liệu bài báo.

Hai chế độ:
  A) Offline eval  → python step3_llm.py --mode eval
  B) API endpoint  → import và gọi gloss_to_sentence() từ step2_server.py
"""

import os, json, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

# ── Cấu hình ──────────────────────────────────────────────────────────────
MODEL_DIR  = r"C:\Users\MYPC\Downloads\Dataset\Models"
EVAL_DIR   = r"C:\Users\MYPC\Downloads\Dataset\Evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PHẦN A — GLOSS → CÂU (2 backend: OpenAI hoặc local Vistral/vit5)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ ký hiệu Việt Nam (VSL).
Nhiệm vụ: Chuyển chuỗi gloss VSL thành một câu tiếng Việt tự nhiên, hoàn chỉnh.
Quy tắc:
- Chỉ trả về câu tiếng Việt, KHÔNG giải thích thêm.
- Thêm đại từ nhân xưng và từ hư phù hợp nếu cần.
- Giữ nguyên ý nghĩa, không thêm thông tin mới.
Ví dụ:
  Input:  XIN CHÀO TÊN TÔI LÀ NAM
  Output: Xin chào, tên tôi là Nam."""


def gloss_to_sentence_openai(gloss_list: list[str],
                              api_key: str | None = None) -> tuple[str, float]:
    """Gọi OpenAI GPT-4o. Trả về (câu, latency_s)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": " ".join(gloss_list)},
            ],
            max_tokens=128,
            temperature=0.2,
        )
        sentence = resp.choices[0].message.content.strip()
        return sentence, time.perf_counter() - t0
    except Exception as e:
        return f"[LLM error: {e}]", 0.0


def gloss_to_sentence_local(gloss_list: list[str]) -> tuple[str, float]:
    """Dùng VietAI/vit5-base (seq2seq) chạy local — không cần internet."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        MODEL_NAME = "VietAI/vit5-base"
        tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
        model      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        prompt = "VSL: " + " ".join(gloss_list)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=80, num_beams=4)
        sentence = tokenizer.decode(ids[0], skip_special_tokens=True)
        return sentence, time.perf_counter() - t0
    except Exception as e:
        return f"[Local LLM error: {e}]", 0.0


# Wrapper chọn backend
def gloss_to_sentence(gloss_list: list[str],
                      backend: str = "openai",
                      api_key: str | None = None) -> tuple[str, float]:
    if backend == "openai":
        return gloss_to_sentence_openai(gloss_list, api_key)
    return gloss_to_sentence_local(gloss_list)


# ══════════════════════════════════════════════════════════════════════════
# PHẦN B — BLEU SCORE (sacrebleu)
# ══════════════════════════════════════════════════════════════════════════

def compute_bleu(hypotheses: list[str], references: list[str]) -> dict:
    """
    hypotheses: danh sách câu mô hình sinh ra
    references:  danh sách câu tham chiếu (ground truth)
    Trả về dict BLEU-1..4 + METEOR nếu có nltk
    """
    try:
        import sacrebleu
        result = {}
        for n in [1, 2, 3, 4]:
            bleu = sacrebleu.corpus_bleu(
                hypotheses,
                [references],
                max_ngram_order=n,
            )
            result[f"BLEU-{n}"] = round(bleu.score, 2)
        return result
    except ImportError:
        # Fallback: NLTK sentence_bleu
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sf = SmoothingFunction().method1
        refs_tok  = [[r.split() for r in references]]
        hyps_tok  = [h.split() for h in hypotheses]
        result = {}
        for n in [1, 2, 3, 4]:
            weights = tuple([1/n]*n + [0]*(4-n))
            score   = corpus_bleu(
                [[r] for r in [r.split() for r in references]],
                hyps_tok,
                weights=weights,
                smoothing_function=sf,
            )
            result[f"BLEU-{n}"] = round(score * 100, 2)
        return result


# ══════════════════════════════════════════════════════════════════════════
# PHẦN C — OFFLINE EVALUATION (tạo số liệu cho bảng bài báo)
# ══════════════════════════════════════════════════════════════════════════

# Dataset cặp (gloss → câu tham chiếu) — tạo tay hoặc bằng GPT-4o
# Format: list of {"gloss": [...], "reference": "câu tham chiếu"}
SAMPLE_PAIRS = [
    {"gloss": ["XIN", "CHÀO"],                    "reference": "Xin chào."},
    {"gloss": ["TÔI", "TÊN", "LÀ", "NAM"],        "reference": "Tôi tên là Nam."},
    {"gloss": ["BẠN", "KHOẺ", "KHÔNG"],           "reference": "Bạn có khoẻ không?"},
    {"gloss": ["CẢM", "ƠN", "BẠN"],               "reference": "Cảm ơn bạn."},
    {"gloss": ["TÔI", "MUỐN", "ĂN", "CƠM"],       "reference": "Tôi muốn ăn cơm."},
    {"gloss": ["HÔM", "NAY", "TRỜI", "ĐẸP"],      "reference": "Hôm nay trời đẹp."},
    {"gloss": ["GIA", "ĐÌNH", "TÔI", "CÓ", "BỐN", "NGƯỜI"],
                                                    "reference": "Gia đình tôi có bốn người."},
    {"gloss": ["TÔI", "ĐI", "HỌC"],               "reference": "Tôi đi học."},
    {"gloss": ["BẠN", "Ở", "ĐÂU"],                "reference": "Bạn ở đâu?"},
    {"gloss": ["XIN", "LỖI", "TÔI", "KHÔNG", "HIỂU"],
                                                    "reference": "Xin lỗi, tôi không hiểu."},
]

def run_offline_eval(backend: str = "openai", api_key: str | None = None):
    print(f"\n🔬 Offline BLEU Evaluation — backend: {backend}")
    print(f"   {len(SAMPLE_PAIRS)} cặp test\n")

    hypotheses, references, rows = [], [], []

    for pair in SAMPLE_PAIRS:
        gloss = pair["gloss"]; ref = pair["reference"]
        hyp, latency = gloss_to_sentence(gloss, backend=backend, api_key=api_key)
        hypotheses.append(hyp)
        references.append(ref)
        rows.append({
            "Gloss"    : " ".join(gloss),
            "Reference": ref,
            "Generated": hyp,
            "Latency_s": round(latency, 3),
        })
        print(f"  [{' '.join(gloss)}]")
        print(f"   Ref: {ref}")
        print(f"   Gen: {hyp}  ({latency*1000:.0f}ms)\n")

    # BLEU
    bleu = compute_bleu(hypotheses, references)
    print("📊 BLEU Scores:")
    for k,v in bleu.items():
        print(f"   {k}: {v}")

    # Lưu kết quả
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EVAL_DIR, "llm_eval_results.csv"), index=False, encoding="utf-8-sig")

    bleu_df = pd.DataFrame([bleu])
    bleu_df.to_csv(os.path.join(EVAL_DIR, "bleu_scores.csv"), index=False)

    print(f"\n✅ Đã lưu:")
    print(f"   {os.path.join(EVAL_DIR, 'llm_eval_results.csv')}")
    print(f"   {os.path.join(EVAL_DIR, 'bleu_scores.csv')}")
    return bleu


# ══════════════════════════════════════════════════════════════════════════
# PHẦN D — TÍCH HỢP VÀO step2_server.py
# ══════════════════════════════════════════════════════════════════════════
# Thêm vào cuối file step2_server.py:
#
#   from step3_llm import gloss_to_sentence
#
#   # Trong websocket_endpoint, sau khi cập nhật state.gloss_seq:
#   if is_new and len(state.gloss_seq) >= 2:
#       sentence, _ = gloss_to_sentence(state.gloss_seq, backend="openai")
#       # gửi sentence về client trong JSON response


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="eval",   choices=["eval", "demo"])
    parser.add_argument("--backend", default="openai", choices=["openai", "local"])
    parser.add_argument("--api-key", default=None,     help="OpenAI API key (hoặc set env OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.mode == "eval":
        run_offline_eval(backend=args.backend, api_key=args.api_key)
    else:
        # Demo tương tác
        print("Demo mode — nhập gloss cách nhau bằng dấu cách (Ctrl+C để thoát)")
        while True:
            try:
                line  = input("Gloss: ").strip().upper()
                if not line: continue
                gloss = line.split()
                sent, lat = gloss_to_sentence(gloss, backend=args.backend, api_key=args.api_key)
                print(f"→ {sent}  ({lat*1000:.0f}ms)\n")
            except KeyboardInterrupt:
                break