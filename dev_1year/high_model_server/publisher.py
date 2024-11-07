import uvicorn
from pydantic import BaseModel
from celery import Celery
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import json
import os
cwd = os.getcwd()
print(cwd)
os.chdir(cwd+'/dev_1year/high_model_server/')
cwd = os.getcwd()
print(cwd)
# Define the Celery task and configuration
def make_celery(app_name=__name__):
    backend = "redis://127.0.0.1:6379"
    broker = "redis://127.0.0.1:6379"
    celery = Celery(app_name, backend=backend, broker=broker)
    return celery

celery_app = make_celery('high_model')

# Define the Pydantic model for structured data
class Item(BaseModel):
    ai_memo: str
    score: float
    event: str
    camera_uid: str

# Initialize FastAPI
app = FastAPI()

@app.post("/api/drift/v1/high_model/")
async def send_msg(
    item: str = Form(...),  # Receive `item` as a JSON string
    image: UploadFile = File(...)
):
    # Parse `item` JSON string into the `Item` model
    item_data = json.loads(item)
    item = Item(**item_data)
    print(cwd)    
    # Process the variables in `item_model`
    print(f"ai_memo: {item.ai_memo}, score: {item.score}, event: {item.event}, camera_uid: {item.camera_uid}")
    print(f"Received file: {image.filename}")
    print(image.size)

    # Save the uploaded image file
    file_location = f"upload/{image.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await image.read())
    print(f"file_location={file_location}")    

    # # Simulate sending a task with Celery
    # task = celery_app.send_task('high_model_task', args=[item.ai_memo])
    # try:
    #     task.wait(timeout=30)
    #     print(task.info)
    #     return {"info": task.info}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail="Task execution failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
