from pydantic import BaseModel
import uvicorn
import redis
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
# class Item(BaseModel):
#     aimemo: str = None
#     score: float

r = redis.Redis(host='127.0.0.1', port=6379, db=0)
# r = redis.Redis(host='localhost', port=6379, db=0)
pub_sub = r.pubsub()
pub_sub.subscribe('aimemo_server')

app = FastAPI()

@app.post("/api/vlm/v1/aimemo_update/")
async def aimemo_update(
    image: UploadFile = File(...),  # Accepting image file
    aimemo: str = Form(...),  # Accepting text description
    score: float = Form(...),
    event: str = Form(...),
    camera_uid: str = Form(...)        
):
    print('aimemo',aimemo)

    # Save the uploaded image file
    file_location = f"upload/{image.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await image.read())
    print(f"file_location={file_location}")
    print(f'event={event}^score={score}^aimemo={aimemo}^camera_uid={camera_uid}')
    r.publish('aimemo_server', f'{file_location}^{event}^{score}^{aimemo}^{camera_uid}')
    
    return JSONResponse(
        content={
            "filename": image.filename,
            "aiout": aimemo,
            "message": "File and text received successfully!"
        }
    )

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="172.16.15.77", port=2222)
    # uvicorn.run(app, host="172.16.15.83", port=2222)
    # uvicorn.run(app, host="127.0.0.1", port=2222)
    uvicorn.run(app, host="0.0.0.0", port=2222)
    # uvicorn.run(app, host="localhost", port=2222)
