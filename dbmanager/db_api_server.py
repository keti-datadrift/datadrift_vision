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

# Pydantic 모델
class EventData(BaseModel):
    request_id: int
    event_name: str
    validation: str
    event_time: datetime
    camera_name: str

# FastAPI 앱 생성
app = FastAPI()

def get_db_connection():
    # return psycopg2.connect(**config)
    conn = psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
        options="-c client_encoding=UTF8",
        cursor_factory=RealDictCursor)
    return conn

@app.post("/api/db_insert_event/")
async def db_insert_event(item: EventData):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        query = f"""
        INSERT INTO datadrift_events (
            request_id, event_name, validation, event_time, camera_name
        ) VALUES (%s, %s, %s, %s, %s);
        """
        cur.execute(query, (
            item.request_id,
            item.event_name,
            item.validation,
            item.event_time,
            item.camera_name
        ))
        conn.commit()

        cur.close()
        conn.close()

        return {"status": "success", "message": "Event inserted successfully!"}
    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

def get_db_connection():
    # return psycopg2.connect(**config, cursor_factory=RealDictCursor)
    conn = psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
        options="-c client_encoding=UTF8",
        cursor_factory=RealDictCursor)
    return conn

@app.get("/api/db_check_drift/")
async def db_check_drift(
    period: str = Query("1 day", description="기간 (예: '1 day', '1 hour')"),
    event_name: str = Query("실시간 인식", description="이벤트명"),
    threshold: float = Query(0.3, description="drift 문턱치 (0~1 사이 비율)")
):
    """
    datadrift_events 테이블에서 특정 기간 동안 특정 이벤트를 조회하고,
    validation == 'False' 비율이 threshold 이상이면 drift 발생으로 판단.
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 최근 기간(예: 1 day, 1 hour)을 WHERE 조건으로 적용
        query = f"""
        SELECT 
            COUNT(*) FILTER (WHERE validation = 'False') AS false_count,
            COUNT(*) AS total_count
        FROM datadrift_events
        WHERE event_name = %s
          AND created_at >= NOW() - INTERVAL %s;
        """

        cur.execute(query, (event_name, period))
        result = cur.fetchone()

        cur.close()
        conn.close()

        false_count = result["false_count"] if result["false_count"] else 0
        total_count = result["total_count"] if result["total_count"] else 0

        if total_count == 0:
            return {"status": "no_data", "message": "조회 기간 동안 데이터가 없습니다."}

        false_ratio = false_count / total_count
        drift_detected = false_ratio >= threshold

        return {
            "status": "success",
            "event_name": event_name,
            "period": period,
            "total_count": total_count,
            "false_count": false_count,
            "false_ratio": round(false_ratio, 3),
            "threshold": threshold,
            "drift_detected": drift_detected
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="localhost", port=8000)

