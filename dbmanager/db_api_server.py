import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor
# import datetime
from datetime import datetime  # ✅ 이렇게 해야 함
# from celery import Celery
# from celery.result import AsyncResult
import os
import yaml
import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging to both file and console"""
    base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"db_api_server_{datetime.now().strftime('%Y%m%d')}.log"

    # File handler - detailed logs
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler - important logs
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Setup logging first
logger = setup_logging()

base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
with open(base_abspath+'/config.yaml', encoding="utf-8") as f:
    config = yaml.full_load(f)

# # DB 연결 설정
# config = {
#     "dbname": "your_db",
#     "user": "your_user",
#     "password": "your_password",
#     "host": "127.0.0.1",
#     "port": "5432"
# }
table_name = config['datadrift_table']   
host = config['postgres']['host']
port = config['postgres']['port']
dbname = config['postgres']['dbname']
user = config['postgres']['user']
password = config['postgres']['password'] 
# Pydantic 모델
class EventData(BaseModel):
    request_id: int
    event_name: str
    validation: str
    event_time: datetime
    camera_name: str

# FastAPI 앱 생성
app = FastAPI()

# def connect_db():
#     # return psycopg2.connect(**config)

#     host=config["postgres"]["host"]
#     port=config["postgres"]["port"]
#     dbname=config["postgres"]["dbname"]
#     user=config["postgres"]["user"]
#     password=config["postgres"]["password"]

#     conn = psycopg2.connect(
#         host=host,
#         port=port,
#         dbname=dbname,
#         user=user,
#         password=password,
#         client_encoding='UTF8'
#     )
#         # options="-c client_encoding=UTF8",
#         # cursor_factory=RealDictCursor)
#     return conn

def connect_db():
    with open(base_abspath+'/config.yaml',encoding='utf-8') as f:
        config = yaml.full_load(f)

    conn = None
    cur = None
    try:     
    
        # PostgreSQL 서버에 연결
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            client_encoding='UTF8'
        )
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        log_msg = f'Exception: {traceback.format_exc()}'
        logging.error(log_msg)

    return conn, cur

def save_full_frame_with_annotation(frame_b64: str, save_dir: str, frame_id: str,
                                   camera_name: str, class_name: str, confidence: float,
                                   bboxes: list, description: str, event_time) -> bool:
    """
    전체 프레임 이미지를 저장하고, YOLO 어노테이션 파일을 함께 생성하는 함수.
    """
    import base64
    import cv2
    import numpy as np
    import os
    from datetime import datetime

    try:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{frame_id}.jpg")
        label_path = os.path.join(save_dir, f"{frame_id}.txt")

        # Base64 → 이미지 디코딩 및 저장
        img_data = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logging.warning("Failed to decode full frame image")
            return False

        cv2.imwrite(img_path, img)
        logging.info(f"Full frame image saved to: {img_path}")

        # 이미지 크기
        h, w = img.shape[:2]

        # 어노테이션 파일 생성
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated annotation\n")
            f.write(f"class_name: {description}\n")
            f.write(f"camera_id: {camera_name}\n")
            f.write(f"confidence: {confidence}\n")
            f.write(f"event_time: {event_time}\n")

            if isinstance(bboxes, list) and len(bboxes) > 0:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                f.write("0 0.5 0.5 1.0 1.0\n")

        logging.info(f"Annotation saved to: {label_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to save frame or annotation: {e}")
        return False

@app.post("/api/db_insert_event/")
async def db_insert_event(response: dict):
    """
    vlm_valid 여부에 따라 data/good 또는 data/wrong_corrected에
    전체 프레임 이미지와 어노테이션을 저장
    """

    try:
        frame_id    = response.get('frame_id')
        camera_name = response.get('camera_id')
        class_name  = response.get('class') or response.get('class_name')
        confidence  = response.get('confidence')
        vlm_valid   = response.get('vlm_valid')
        frame_b64   = response.get('frame_b64')
        bboxes      = response.get('bboxes', [])
        description = response.get('description', class_name)
        event_time  = response.get('event_time')

        if isinstance(event_time, (int, float)):
            event_time = datetime.fromtimestamp(event_time)
        elif event_time is None:
            event_time = datetime.now()

        # 저장 경로 결정
        save_dir = os.path.join(base_abspath, "data", "good" if vlm_valid else "wrong_corrected")

        # 이미지 및 어노테이션 저장
        if not frame_b64:
            logging.warning("No frame_b64 found in response")
            return {"status": "error", "message": "No frame_b64 provided"}

        ok = save_full_frame_with_annotation(
            frame_b64=frame_b64,
            save_dir=save_dir,
            frame_id=frame_id,
            camera_name=camera_name,
            class_name=class_name,
            confidence=confidence,
            bboxes=bboxes,
            description=description,
            event_time=event_time,
        )

        if not ok:
            return {"status": "error", "message": "Failed to save frame or annotation"}

        return {"status": "success", "message": f"Saved to {save_dir}"}

    except Exception:
        logging.error(f"Unexpected error while saving data: {traceback.format_exc()}")
        return {"status": "error", "message": "Unexpected error while saving data"}


@app.get("/api/db_check_drift/")
async def db_check_drift(
    class_name: str = Query("person", description="클래스명 (예: 'person', 'car')"),
    threshold: float = Query(0.3, description="drift 문턱치 (0~1 사이 비율)")
):
    """
    테이블에서 특정 기간 동안 특정 클래스를 조회하고,
    vlm_valid == 'no' 비율이 threshold 이상이면 drift 발생으로 판단.

    단, 최근에 모델이 업데이트되었다면 (cooldown_after_update 기간 이내),
    drift 계산을 수행하지 않고 스킵함.
    """

    try:
        # Load config to get drift check parameters
        with open(base_abspath+'/config.yaml', encoding='utf-8') as f:
            config = yaml.full_load(f)

        # Get drift detection configuration
        drift_config = config.get('drift_detection', {})
        drift_check_period = drift_config.get('drift_check_period', '1 day')
        cooldown_after_update = drift_config.get('cooldown_after_update', '1 day')

        # Get last model update timestamp
        last_model_update = config.get('yolo_model', {}).get('last_model_update', None)

        # Check if model was recently updated (within cooldown period)
        if last_model_update:
            try:
                from dateutil import parser
                last_update_time = parser.isoparse(last_model_update)
                now = datetime.now()

                # Check if we're still within the cooldown period
                # We'll use a database query to leverage PostgreSQL's interval arithmetic
                conn, cur = connect_db()
                if conn is None or cur is None:
                    return {"status": "error", "message": "Database connection failed"}

                # Check time difference using PostgreSQL
                cooldown_query = """
                SELECT CASE
                    WHEN %s + INTERVAL %s > NOW() THEN TRUE
                    ELSE FALSE
                END AS within_cooldown
                """
                cur.execute(cooldown_query, (last_update_time, cooldown_after_update))
                within_cooldown = cur.fetchone()[0]

                if within_cooldown:
                    cur.close()
                    conn.close()
                    return {
                        "status": "skipped",
                        "message": "Drift check skipped - model was recently updated",
                        "class_name": class_name,
                        "last_model_update": last_model_update,
                        "cooldown_period": cooldown_after_update,
                        "drift_detected": False,
                        "reason": f"Model updated at {last_model_update}. Cooldown period: {cooldown_after_update}"
                    }

                # If we're past cooldown, proceed with drift check
                cur.close()
                conn.close()

            except Exception as e:
                logging.warning(f"Failed to parse last_model_update timestamp: {e}")
                # Continue with drift check if timestamp parsing fails

        # Proceed with normal drift detection
        conn, cur = connect_db()
        if conn is None or cur is None:
            return {"status": "error", "message": "Database connection failed"}

        # 최근 기간(예: 1 day, 1 hour)을 WHERE 조건으로 적용
        query = f"""
        SELECT
            COUNT(*) FILTER (WHERE vlm_valid = 'no') AS false_count,
            COUNT(*) AS total_count
        FROM {table_name}
        WHERE class_name = %s
          AND created_at >= NOW() - INTERVAL %s;
        """

        cur.execute(query, (class_name, drift_check_period))
        result = cur.fetchone()

        cur.close()
        conn.close()

        # None 체크 개선
        if result is None:
            return {"status": "no_data", "message": "조회 결과가 없습니다."}

        false_count = result[0] if result[0] is not None else 0
        total_count = result[1] if result[1] is not None else 0

        if total_count == 0:
            return {
                "status": "no_data",
                "message": "조회 기간 동안 데이터가 없습니다.",
                "class_name": class_name,
                "period": drift_check_period
            }

        false_ratio = false_count / total_count
        logging.info(f"Drift check result: false_ratio={false_ratio:.3f}, threshold={threshold}")
        drift_detected = false_ratio >= threshold

        return {
            "status": "success",
            "class_name": class_name,
            "drift_check_period": drift_check_period,
            "total_count": total_count,
            "false_count": false_count,
            "false_ratio": round(false_ratio, 3),
            "threshold": threshold,
            "drift_detected": drift_detected,
            "last_model_update": last_model_update,
            "message": f"Drift {'감지됨' if drift_detected else '정상'}"
        }

    except Exception as e:
        logging.error(f"Error in db_check_drift: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    
@app.get("/api/db_retrain/")
async def db_retrain():
    """재학습 - 백그라운드 프로세스로 train_model.py 실행"""
    try:
        import subprocess

        # 백그라운드에서 학습 프로세스 시작
        subprocess.Popen(["python", f"{base_abspath}/retrain/train_model.py"])

        return {
            "status": "success",
            "message": "Training started in background"
        }

    except Exception as e:
        logging.error(f"Error in db_retrain: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="localhost", port=8000)

