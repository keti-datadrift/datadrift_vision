import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import psycopg2
import datetime
# from celery import Celery
# from celery.result import AsyncResult

# DB 연결 설정
DB_CONFIG = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "127.0.0.1",
    "port": "5432"
}

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
    return psycopg2.connect(**DB_CONFIG)

@app.post("/api/datadrift_db_api/")
async def insert_event(item: EventData):
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

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="localhost", port=8000)

