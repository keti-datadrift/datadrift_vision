# 4-모듈 파이프라인 구성 안내

## 개요
- **파일 1:** `m1_yolo_producer.py` — YOLO 객체 검출 결과를 Redis Stream(`detections`)으로 발행
- **파일 2:** `m2_realtime_event_producer.py` — `detections`를 구독해 실시간 이벤트 인지 후 `realtime_events` 발행
- **파일 3:** `m3_vlm_verifier_batch.py` — DB에서 최근 결과 조회, Kanana VLM(mock/실사용)으로 검증, `vlm_verifications` 발행 (비실시간, 주기 실행)
- **파일 4:** `m4_db_writer.py` — `detections`/`realtime_events`/`vlm_verifications` 스트림을 구독해 PostgreSQL 저장

모든 결과는 동일한 DB에 저장됩니다.

## 설치
```bash
pip install ultralytics opencv-python redis psycopg2-binary pillow
# (선택) 실제 VLM 사용 시
pip install torch transformers accelerate bitsandbytes
```

## 환경변수 예시
```bash
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_DB=0

export PG_HOST=127.0.0.1
export PG_PORT=5432
export PG_DB=postgres
export PG_USER=postgres
export PG_PW=postgres

export DETECTIONS_STREAM=detections
export REALTIME_STREAM=realtime_events
export VLM_STREAM=vlm_verifications
```

### 모듈별
```bash
# 파일 1
export VIDEO_SOURCE=0            # 또는 /path/to/video.mp4
export YOLO_MODEL=yolov8n.pt
export CAMERA_ID=camera_1
export CONF_THRESH=0.3
export ALLOW_CLASSES=person,fire,smoke
export ROI_JPEG_QUALITY=80
export SHOW_PREVIEW=0

# 파일 2
export EVENT_MODEL=mock          # (추후 실제 모델로 교체)

# 파일 3
export INTERVAL_SEC=10
export WINDOW_SEC=30
export USE_VLM=0                 # 1이면 실제 Kanana VLM 사용
export KANANA_MODEL=kakaocorp/kanana-1.5-v-3b-instruct
```

## 실행 순서
별도 터미널 3~4개로 동시에 실행합니다.

```bash
# 1) DB Writer (파일 4)
python m4_db_writer.py

# 2) YOLO Producer (파일 1)
python m1_yolo_producer.py

# 3) Realtime Event Producer (파일 2)
python m2_realtime_event_producer.py

# 4) (선택) VLM Verifier Batch (파일 3) - 비실시간 주기 실행
python m3_vlm_verifier_batch.py
```

## 스키마
- `detections(detection_id, camera_id, frame_id, ts, class, confidence, bbox, roi_b64)`
- `realtime_events(event_id, detection_id, camera_id, frame_id, ts, event_type, score, decision, ref_class)`
- `vlm_verifications(verification_id, target, ref_id, camera_id, frame_id, ts, question, answer_raw, verified)`

## 커스터마이징 포인트
- **파일 2**의 `run_event_model()`에 실제 실시간 이벤트 인지 모델 연결
- **파일 3**의 `detection_question()`, `event_question()` 프롬프트 커스터마이즈 및 ROI 확보 로직 확장
- 테이블/인덱스/파티션 전략은 운영 요건에 맞게 조정
