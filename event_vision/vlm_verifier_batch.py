#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 3) Kanana VLM 기반 비실시간 검증 모듈(주기 실행)
- 입력: PostgreSQL에서 최근의 detections / realtime_events 조회
- 처리: Kanana VLM 또는 mock 검증(환경변수로 토글)으로 신뢰성 검토
- 출력: Redis Stream('vlm_verifications')로 검증 결과 publish
- 주기: 기본 10초 간격
"""
import os
import io
import json
import time
import uuid
import base64
import redis
import psycopg2
import psycopg2.extras
from PIL import Image
import torch
import yaml

# ---------------- config.yaml 불러오기 ----------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Redis 설정
REDIS_HOST = config["redis"]["host"]
REDIS_PORT = int(config["redis"]["port"])
REDIS_DB   = int(config["redis"]["db"])
OBJDET_STREAM = config["redis"]["objdet_stream"]   # YOLO producer stream
VLM_STREAM    = config["redis"]["vlm_stream"]      # 검증 결과 stream

# Postgres 설정
PG_HOST = config["postgres"]["host"]
PG_PORT = int(config["postgres"]["port"])
PG_DB   = config["postgres"]["dbname"]
PG_USER = config["postgres"]["user"]
PG_PW   = config["postgres"]["password"]

# VLM 관련 설정
INTERVAL_SEC = int(config.get("vlm", {}).get("interval_sec", 10))
WINDOW_SEC   = int(config.get("vlm", {}).get("window_sec", 30))
USE_VLM      = config.get("vlm", {}).get("use_vlm", False)
KANANA_MODEL = config.get("vlm", {}).get("kanana_model", "kakaocorp/kanana-1.5-v-3b-instruct")


vlm_model = None
vlm_processor = None
def init_vlm():
    global vlm_model, vlm_processor
    if not USE_VLM:
        return
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        KANANA_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    vlm_processor = AutoProcessor.from_pretrained(KANANA_MODEL, trust_remote_code=True)

def ask_kanana_yesno(pil_image: Image.Image, question: str):

    sample = {
        "image": [pil_image],
        "conv": [
            {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": question},
        ]
    }
    batch = [sample]
    inputs = vlm_processor.batch_encode_collate(batch, padding_side="left", add_generation_prompt=True, max_length=8192)
    inputs = {k: (v.to(vlm_model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    outputs = vlm_model.generate(**inputs, max_new_tokens=32, temperature=0.0, top_p=1.0, num_beams=1, do_sample=False)
    text = vlm_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    low = text.lower()
    if ("네" in text) and ("아니" not in text): return True, text
    if ("아니" in text): return False, text
    if ("yes" in low) and ("no" not in low): return True, text
    if ("no" in low): return False, text
    return None, text

def b64_to_pil(b64_str: str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def detection_question(det_class: str):
    m = {
        "fire": "불꽃이 보이나요? '네', '아니요'로 대답하세요.",
        "smoke": "연기가 보이나요? '네', '아니요'로 대답하세요.",
        "person": "사람이 보이나요? '네', '아니요'로 대답하세요.",
        "motorbike": "오토바이가 보이나요? '네', '아니요'로 대답하세요."
    }
    return m.get((det_class or '').lower(), f"{det_class}가 보이나요? '네', '아니요'로 대답하세요.")

def event_question(event_type: str):
    m = {
        "fall": "쓰러짐이 발생하였나요? '네', '아니요'로 대답하세요.",
        "fire": "화재가 발생하였나요? '네', '아니요'로 대답하세요.",
        "smoke": "연기가 발생하였나요? '네', '아니요'로 대답하세요.",
        "violence": "폭력이 발생하였나요? '네', '아니요'로 대답하세요.",
        "loitering": "배회가 발생하였나요? '네', '아니요'로 대답하세요."
    }
    return m.get((event_type or '').lower(), f"{event_type}가 발생하였나요? '네', '아니요'로 대답하세요.")

def main():
    init_vlm()

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PW)
    conn.autocommit = True

    print(f"[m3] vlm verifier batch: interval={INTERVAL_SEC}s window={WINDOW_SEC}s USE_VLM={int(USE_VLM)}")
    try:
        while True:
            now = time.time()
            since = now - WINDOW_SEC

            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT detection_id, camera_id, frame_id, EXTRACT(EPOCH FROM ts) AS ts, class, confidence, roi_b64
                    FROM detections
                    WHERE ts >= to_timestamp(%s);
                """, (since,))
                det_rows = cur.fetchall()

                cur.execute("""
                    SELECT event_id, detection_id, camera_id, frame_id, EXTRACT(EPOCH FROM ts) AS ts, event_type, score, decision
                    FROM realtime_events
                    WHERE ts >= to_timestamp(%s);
                """, (since,))
                evt_rows = cur.fetchall()

            for row in det_rows:
                det_class = row["class"]
                question = detection_question(det_class)
                verified, answer_raw = (None, "mock: not-checked")

                if USE_VLM:
                    try:
                        pil = b64_to_pil(row["roi_b64'])
                        verified, answer_raw = ask_kanana_yesno(pil, question)
                    except Exception as e:
                        verified, answer_raw = None, f"vlm_error: {e}"
                else:
                    verified = (row["confidence"] or 0.0) >= 0.6
                    answer_raw = f"mock:{'네' if verified else '아니요'}"

                msg = {
                    "version": 1,
                    "source": "vlm",
                    "target": "detection",
                    "camera_id": row["camera_id"],
                    "frame_id": row["frame_id"],
                    "ref_id": row["detection_id"],
                    "ts": time.time(),
                    "question": question,
                    "answer_raw": answer_raw,
                    "verified": verified,
                }
                r.xadd(VLM_STREAM, {"data": json.dumps(msg).encode("utf-8")})

            for row in evt_rows:
                evt_type = row["event_type"]
                question = event_question(evt_type)
                verified, answer_raw = (None, "mock: not-checked")

                if USE_VLM:
                    try:
                        verified, answer_raw = (True if row["decision"] else False,
                                                f"heuristic_from_decision={row['decision']}")
                    except Exception as e:
                        verified, answer_raw = None, f"vlm_error: {e}"
                else:
                    verified = True if row["decision"] else False
                    answer_raw = f"mock_from_realtime_decision={row['decision']}"

                msg = {
                    "version": 1,
                    "source": "vlm",
                    "target": "event",
                    "camera_id": row["camera_id"],
                    "frame_id": row["frame_id"],
                    "ref_id": row["event_id"],
                    "ts": time.time(),
                    "question": question,
                    "answer_raw": answer_raw,
                    "verified": verified,
                }
                r.xadd(VLM_STREAM, {"data": json.dumps(msg).encode("utf-8")})

            time.sleep(INTERVAL_SEC)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
