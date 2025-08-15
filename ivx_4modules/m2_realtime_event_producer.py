#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 2) 실시간 이벤트 인지 프로듀서
- 입력: Redis Stream('detections')의 객체검출(ROI 포함) 메시지 구독
- 처리: 실시간 이벤트 인지(모델 미정 → 모듈러 인터페이스 + mock 로직 제공)
- 출력: Redis Stream('realtime_events')로 결과 publish
- DB 저장은 수행하지 않으며, 파일 4(DB Writer)가 구독하여 저장
"""
import os
import io
import json
import time
import base64
import redis
import uuid
from PIL import Image

# -------------------- 설정 --------------------
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

DETECTIONS_STREAM = os.getenv("DETECTIONS_STREAM", "detections")
REALTIME_STREAM   = os.getenv("REALTIME_STREAM", "realtime_events")
BLOCK_MS          = int(os.getenv("BLOCK_MS", "1000"))
COUNT             = int(os.getenv("COUNT", "1"))
START_ID          = os.getenv("START_ID", "0-0")

EVENT_MODEL = os.getenv("EVENT_MODEL", "mock")

def mock_event_infer(roi_img: Image.Image, det_class: str):
    det = (det_class or "unknown").lower()
    w, h = roi_img.size
    if det in {"fire", "flame"}:
        return "fire", 0.95, True
    if det in {"smoke"}:
        return "smoke", 0.90, True
    if det in {"person"}:
        if w > h * 1.3:
            return "fall", 0.72, True
        return "loitering", 0.35, False
    return None, 0.0, False

def run_event_model(roi_img: Image.Image, det_class: str):
    if EVENT_MODEL == "mock":
        return mock_event_infer(roi_img, det_class)
    # TODO: 실제 모델 연동
    return None, 0.0, False

def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    last_id = START_ID
    print(f"[m2] realtime-event producer: read {DETECTIONS_STREAM} start={last_id} → publish {REALTIME_STREAM}")
    while True:
        messages = r.xread({DETECTIONS_STREAM: last_id}, block=BLOCK_MS, count=COUNT)
        if not messages:
            continue
        for stream_name, entries in messages:
            for msg_id, fields in entries:
                last_id = msg_id
                data = fields.get(b"data")
                if not data:
                    continue
                try:
                    det = json.loads(data.decode("utf-8"))
                except Exception:
                    continue

                roi_b64 = det.get("roi_b64")
                if not roi_b64:
                    continue
                try:
                    roi_img = Image.open(io.BytesIO(base64.b64decode(roi_b64)))
                except Exception:
                    continue

                det_class = det.get("class", "unknown")
                event_type, score, decision = run_event_model(roi_img, det_class)

                evt = {
                    "version": 1,
                    "source": "realtime",
                    "camera_id": det.get("camera_id"),
                    "frame_id": det.get("frame_id"),
                    "detection_id": det.get("detection_id"),
                    "event_id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "event_type": event_type,
                    "score": score,
                    "decision": bool(decision),
                    "ref_class": det_class,
                }
                r.xadd(REALTIME_STREAM, {"data": json.dumps(evt).encode("utf-8")})

if __name__ == "__main__":
    main()
