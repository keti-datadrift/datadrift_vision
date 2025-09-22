#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 1) YOLO 기반 객체 검출 결과를 Redis Stream('detections')에 발행(publish)하는 프로듀서
- 영상 입력(웹캠/파일) → YOLO 검출 → ROI 추출(base64 JPEG) → Redis XADD
- DB 저장은 수행하지 않으며, 파일 4(DB Writer)가 구독하여 저장합니다.

필수 패키지:
  pip install ultralytics opencv-python redis pillow
선택/권장:
  pip install numpy
"""
import os
import sys
import cv2
import time
import uuid
import json
import base64
import redis
from PIL import Image
from io import BytesIO
import yaml
from ultralytics import YOLO
# import locale, sys
# locale.setlocale(locale.LC_ALL, "ko_KR.UTF-8")
# sys.stdout.reconfigure(encoding="utf-8")
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(base_abspath)

event_vision_abspath = f'{base_abspath}/event_vision/'
sys.path.append(event_vision_abspath)
# -------------------- 설정 --------------------


# config.yaml 불러오기
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Redis
REDIS_HOST = config["redis"]["host"]
REDIS_PORT = int(config["redis"]["port"])
REDIS_DB   = int(config["redis"]["db"])
STREAM     = config["redis"]["objdet_stream"]

# Video
VIDEO_SOURCE = config["video"]["source"]
CAMERA_ID    = config["video"]["camera_id"]

# Model
YOLO_MODEL   = config["yolo_model"]["model_name"]
CONF_THRESH  = float(config["yolo_model"]["conf_thresh"])

ALLOW_CLASSES_STR = config["yolo_model"].get("allow_classes", "").strip()
ALLOW_CLASSES = [c.strip() for c in ALLOW_CLASSES_STR.split(",") if c.strip()] if ALLOW_CLASSES_STR else None

# Preview
ROI_JPEG_QUALITY = int(config["preview"]["roi_jpeg_quality"])
SHOW_PREVIEW = config["preview"]["show_preview"] == 1


# -------------------- 유틸 --------------------
def crop_roi(bgr, x1, y1, x2, y2, margin=0.06):
    h, w = bgr.shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    X1 = max(0, x1 - dx)
    Y1 = max(0, y1 - dy)
    X2 = min(w - 1, x2 + dx)
    Y2 = min(h - 1, y2 + dy)
    if X2 <= X1 or Y2 <= Y1:
        return None
    return bgr[Y1:Y2, X1:X2].copy(), [X1, Y1, X2, Y2]

def bgr_to_b64jpg(bgr, quality=80):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    # Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

    # YOLO
    model = YOLO(YOLO_MODEL)

    # Video
    cap = cv2.VideoCapture(0 if VIDEO_SOURCE == "0" else VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    frame_idx = 0
    names = None

    print(f"[m1] start YOLO producer: stream={STREAM}, camera_id={CAMERA_ID}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[m1] video read failed/broken. exiting.")
                break
            frame_idx += 1
            frame_id = str(uuid.uuid4())
            ts = time.time()

            # YOLO inference
            results = model(frame, verbose=False, device=1)
            for result in results:
                if names is None:
                    names = result.names

                for b in result.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf)
                    cls_id = int(b.cls)
                    cls_name = names.get(cls_id, str(cls_id))

                    if conf < CONF_THRESH:
                        continue
                    if ALLOW_CLASSES and (cls_name not in ALLOW_CLASSES):
                        continue

                    roi, bbox = crop_roi(frame, x1, y1, x2, y2, margin=0.06)
                    if roi is None:
                        continue
                    roi_b64 = bgr_to_b64jpg(roi, quality=ROI_JPEG_QUALITY)

                    detection_id = str(uuid.uuid4())
                    msg = {
                        "version": 1,
                        "source": "yolo",
                        "camera_id": CAMERA_ID,
                        "frame_id": frame_id,
                        "detection_id": detection_id,
                        "ts": ts,
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": bbox,
                        "roi_b64": roi_b64,
                    }
                    # r.xadd(STREAM, {"data": json.dumps(msg).encode("utf-8")})
                    r.lpush(STREAM, json.dumps(msg))

                    if SHOW_PREVIEW:
                        label = f"{cls_name} {conf:.2f}"
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                        cv2.putText(frame, label, (bbox[0], max(10, bbox[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if SHOW_PREVIEW:
                cv2.imshow("YOLO Producer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
