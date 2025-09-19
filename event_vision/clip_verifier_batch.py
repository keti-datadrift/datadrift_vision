#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 3) CLIP 기반 비실시간 검증 모듈(주기 실행)
- 입력: PostgreSQL에서 최근의 detections / realtime_events 조회
- 처리: CLIP을 이용한 임베딩 매칭 기반 신뢰성 검토
- 출력: Redis Stream('vlm_verifications')로 검증 결과 publish
- 주기: 기본 10초 간격
"""
import os
import io
import json
import time
import base64
import redis
import psycopg2
import psycopg2.extras
from PIL import Image

import torch
import clip
import numpy as np

import yaml

# config.yaml 불러오기
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Redis
REDIS_HOST = config["redis"]["host"]
REDIS_PORT = int(config["redis"]["port"])
REDIS_DB   = int(config["redis"]["db"])
VLM_STREAM = config["redis"]["vlm_stream"]

# Postgres
PG_HOST = config["postgres"]["host"]
PG_PORT = int(config["postgres"]["port"])
PG_DB   = config["postgres"]["db"]
PG_USER = config["postgres"]["user"]
PG_PW   = config["postgres"]["password"]

# Intervals
INTERVAL_SEC = int(config["intervals"]["interval_sec"])
WINDOW_SEC   = int(config["intervals"]["window_sec"])

# ---------------- CLIP 초기화 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 이벤트 라벨 딕셔너리(TEXT_LABELS)
TEXT_LABELS = {
    "felldown_0": "a man is lay on the floor",
    "felldown_1": "a woman is lay on the floor",
    "fire_0": "a house is burning",
    "fire_1": "a car is burning",
    "smoke_0": "smoke is rising from a building",
    "helmet_0": "a worker is wearing a safety helmet",
    "nohelmet_0": "a worker without a helmet"
}

TEXT_KEYS = list(TEXT_LABELS.keys())
TEXT_PROMPTS = list(TEXT_LABELS.values())

with torch.no_grad():
    text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)
    text_embeddings = clip_model.encode_text(text_tokens)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

def b64_to_pil(b64_str: str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def get_clip_embedding(pil_image: Image.Image):
    img = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb

def verify_with_clip(roi_b64_list):
    results = []
    img_embeddings = []

    for roi_b64 in roi_b64_list:
        pil_img = b64_to_pil(roi_b64).convert("RGB")
        emb = get_clip_embedding(pil_img)
        img_embeddings.append(emb)

    if not img_embeddings:
        return []

    img_embeddings = torch.cat(img_embeddings, dim=0)
    sims = (img_embeddings @ text_embeddings.T).cpu().numpy()

    for i, sim in enumerate(sims):
        best_idx = int(np.argmax(sim))
        best_key = TEXT_KEYS[best_idx]
        best_prompt = TEXT_PROMPTS[best_idx]
        best_score = sim[best_idx]

        results.append({
            "bbox_index": i,
            "event_name": best_key,
            "matched_prompt": best_prompt,
            "score": float(best_score)
        })

    return results

def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PW)
    conn.autocommit = True

    print(f"[m3] CLIP verifier batch: interval={INTERVAL_SEC}s window={WINDOW_SEC}s")
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

            for row in det_rows:
                clip_results = verify_with_clip([row["roi_b64"]])
                for r_res in clip_results:
                    msg = {
                        "version": 1,
                        "source": "clip",
                        "camera_id": row["camera_id"],
                        "frame_id": row["frame_id"],
                        "ref_id": row["detection_id"],
                        "ts": time.time(),
                        "event_name": r_res["event_name"],
                        "matched_prompt": r_res["matched_prompt"],
                        "score": r_res["score"]
                    }
                    r.xadd(VLM_STREAM, {"data": json.dumps(msg).encode("utf-8")})

            time.sleep(INTERVAL_SEC)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
