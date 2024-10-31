import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult

class Item(BaseModel):
    search_type: int
    query_text : str = None

def make_celery(app_name=__name__):
    backend = "redis://127.0.0.1:6379"
    broker  = "redis://127.0.0.1:6379"
    
    celery = Celery(__name__, backend=backend, broker=broker)

    return celery


celery_app = make_celery('vllm')

app = FastAPI()
@app.post("/api/drift/v1/vlm_verify/")
async def send_msg(item: Item):
    print('@app.post("/items/")')
    print(item)
    task = celery_app.send_task('generate_text_task', args=[item.query_text])
    task.wait(timeout=30)
    print(task.info)
    object_type = ''

    return task.info

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="localhost", port=8000)

