#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 4) DB Writer (통합 컨슈머)
- Redis Stream: 'detections', 'realtime_events', 'vlm_verifications' 구독
- 표준 스키마로 PostgreSQL 저장 (테이블 자동 생성)
"""
import os
import json
import time
import redis
import psycopg2
import psycopg2.extras

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

STREAM_DET = os.getenv("DETECTIONS_STREAM", "detections")
STREAM_EVT = os.getenv("REALTIME_STREAM", "realtime_events")
STREAM_VLM = os.getenv("VLM_STREAM", "vlm_verifications")
BLOCK_MS   = int(os.getenv("BLOCK_MS", "1000"))
COUNT      = int(os.getenv("COUNT", "10"))

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "postgres")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PW   = os.getenv("PG_PW", "postgres")

START_ID_DET = os.getenv("START_ID_DET", "0-0")
START_ID_EVT = os.getenv("START_ID_EVT", "0-0")
START_ID_VLM = os.getenv("START_ID_VLM", "0-0")

DDL_DET = """
CREATE TABLE IF NOT EXISTS detections (
    id            BIGSERIAL PRIMARY KEY,
    detection_id  UUID UNIQUE,
    camera_id     TEXT,
    frame_id      TEXT,
    ts            TIMESTAMPTZ,
    class         TEXT,
    confidence    DOUBLE PRECISION,
    bbox          JSONB,
    roi_b64       TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
"""

DDL_EVT = """
CREATE TABLE IF NOT EXISTS realtime_events (
    id            BIGSERIAL PRIMARY KEY,
    event_id      UUID UNIQUE,
    detection_id  UUID,
    camera_id     TEXT,
    frame_id      TEXT,
    ts            TIMESTAMPTZ,
    event_type    TEXT,
    score         DOUBLE PRECISION,
    decision      BOOLEAN,
    ref_class     TEXT
);
CREATE INDEX IF NOT EXISTS idx_evt_ts ON realtime_events(ts);
"""

DDL_VLM = """
CREATE TABLE IF NOT EXISTS vlm_verifications (
    id            BIGSERIAL PRIMARY KEY,
    verification_id UUID UNIQUE,
    target        TEXT,
    ref_id        TEXT,
    camera_id     TEXT,
    frame_id      TEXT,
    ts            TIMESTAMPTZ,
    question      TEXT,
    answer_raw    TEXT,
    verified      BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_vlm_ts ON vlm_verifications(ts);
"""

INS_DET = """
INSERT INTO detections (detection_id, camera_id, frame_id, ts, class, confidence, bbox, roi_b64)
VALUES (%(detection_id)s, %(camera_id)s, %(frame_id)s, to_timestamp(%(ts)s), %(class)s, %(confidence)s, %(bbox)s, %(roi_b64)s)
ON CONFLICT (detection_id) DO NOTHING;
"""

INS_EVT = """
INSERT INTO realtime_events (event_id, detection_id, camera_id, frame_id, ts, event_type, score, decision, ref_class)
VALUES (%(event_id)s, %(detection_id)s, %(camera_id)s, %(frame_id)s, to_timestamp(%(ts)s), %(event_type)s, %(score)s, %(decision)s, %(ref_class)s)
ON CONFLICT (event_id) DO NOTHING;
"""

INS_VLM = """
INSERT INTO vlm_verifications (verification_id, target, ref_id, camera_id, frame_id, ts, question, answer_raw, verified)
VALUES (%(verification_id)s, %(target)s, %(ref_id)s, %(camera_id)s, %(frame_id)s, to_timestamp(%(ts)s), %(question)s, %(answer_raw)s, %(verified)s)
ON CONFLICT (verification_id) DO NOTHING;
"""

def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(DDL_DET)
        cur.execute(DDL_EVT)
        cur.execute(DDL_VLM)
    conn.commit()

def handle_det(cur, payload):
    row = {
        "detection_id": payload.get("detection_id"),
        "camera_id": payload.get("camera_id"),
        "frame_id": payload.get("frame_id"),
        "ts": payload.get("ts"),
        "class": payload.get("class"),
        "confidence": payload.get("confidence"),
        "bbox": json.dumps(payload.get("bbox")),
        "roi_b64": payload.get("roi_b64"),
    }
    cur.execute(INS_DET, row)

def handle_evt(cur, payload):
    row = {
        "event_id": payload.get("event_id"),
        "detection_id": payload.get("detection_id"),
        "camera_id": payload.get("camera_id"),
        "frame_id": payload.get("frame_id"),
        "ts": payload.get("ts"),
        "event_type": payload.get("event_type"),
        "score": payload.get("score"),
        "decision": payload.get("decision"),
        "ref_class": payload.get("ref_class"),
    }
    cur.execute(INS_EVT, row)

def handle_vlm(cur, payload):
    row = {
        "verification_id": str(payload.get("verification_id") or os.urandom(8).hex()),
        "target": payload.get("target"),
        "ref_id": payload.get("ref_id"),
        "camera_id": payload.get("camera_id"),
        "frame_id": payload.get("frame_id"),
        "ts": payload.get("ts"),
        "question": payload.get("question"),
        "answer_raw": payload.get("answer_raw"),
        "verified": payload.get("verified"),
    }
    cur.execute(INS_VLM, row)

def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PW)
    conn.autocommit = True
    ensure_tables(conn)

    last_ids = {STREAM_DET: START_ID_DET, STREAM_EVT: START_ID_EVT, STREAM_VLM: START_ID_VLM}
    print(f"[m4] db-writer consuming: {STREAM_DET}, {STREAM_EVT}, {STREAM_VLM}")

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            while True:
                streams = {k: v for k, v in last_ids.items()}
                msgs = r.xread(streams, block=BLOCK_MS, count=COUNT)
                if not msgs:
                    continue
                for stream_name, entries in msgs:
                    sname = stream_name.decode() if isinstance(stream_name, (bytes, bytearray)) else stream_name
                    for msg_id, fields in entries:
                        last_ids[sname] = msg_id
                        data = fields.get(b"data")
                        if not data:
                            continue
                        try:
                            payload = json.loads(data.decode("utf-8"))
                        except Exception:
                            continue

                        if sname == STREAM_DET:
                            handle_det(cur, payload)
                        elif sname == STREAM_EVT:
                            handle_evt(cur, payload)
                        elif sname == STREAM_VLM:
                            handle_vlm(cur, payload)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
