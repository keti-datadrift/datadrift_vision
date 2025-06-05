import cv2
import redis
import json
import time
import base64
# import psycopg2
# from detic.predictor import DeticPredictor
import numpy as np
from io import BytesIO
from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# Redis 클라이언트 초기화
# redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(host='127.0.0.1', port=6379, db=0)
# r = redis.Redis(host='localhost', port=6379, db=0)
pub_sub = r.pubsub()
pub_sub.subscribe('vlm_server')

def run():
    # Listen for messages
    # i = 0
    for message in pub_sub.listen():
        if message['type'] == 'message':
            try:
                msg = message['data'].decode()
                # JSON 문자열을 파싱
                payload = json.loads(msg)

                # 각 필드 추출
                frame_id = payload.get("frame_id")
                detections = payload.get("detections", [])
                img_base64 = payload.get("image")
                processing_time = payload.get("processing_time")

                image_data = base64.b64decode(img_base64)
                pil_img = Image.open(BytesIO(image_data)).convert("RGB")
                rgb_img = np.array(pil_img)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                # 필요한 처리 로직 삽입 (예: 로깅, 이미지 디코딩 등)
                print(f"[frame_id={frame_id}] {len(detections)} detections, time: {processing_time:.3f}s")

                cv2.imshow("vlm receiver", bgr_img)

                # 'q' 키를 누르면 종료
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break                
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")

if __name__ == '__main__':
    run()

# # # PostgreSQL 연결 설정
# # conn = psycopg2.connect(
# #     dbname="your_db_name",
# #     user="your_username",
# #     password="your_password",
# #     host="localhost",
# #     port="5432"
# # )
# # cursor = conn.cursor()

# # # PostgreSQL 테이블 생성 (최초 1회 실행)
# # cursor.execute("""
# #     CREATE TABLE IF NOT EXISTS detections (
# #         id SERIAL PRIMARY KEY,
# #         frame_id INTEGER,
# #         rt_class VARCHAR(50),
# #         rt_confidence FLOAT,
# #         rt_bbox JSONB,
# #         rt_time TIMESTAMP,
# #         event_info TEXT,
# #         frame_path TEXT,
# #         vlm_description TEXT,
# #         processing_time FLOAT
# #     );
# # """)
# # conn.commit()

# # # Detic 모델 초기화
# # detic_predictor = DeticPredictor(
# #     model_type="Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size",
# #     confidence_threshold=0.5
# # )

# # VLM (CLIP) 모델 초기화
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Redis Stream 이름
# STREAM_NAME = "video_frames_stream"

# # 프레임 처리 간격 (초)
# FRAME_PROCESSING_INTERVAL = 1.0

# # 결과 저장을 위한 리스트
# results_list = []

# # 마지막으로 처리한 메시지 ID 초기화
# last_id = "0-0"

# try:
#     # while True:
#         # Redis Stream에서 메시지 읽기 (일정 간격으로 폴링)

#         # messages = redis_client.xread({STREAM_NAME: last_id}, block=1000, count=1)
#         # if isinstance(detection_str, bytes):
#     for message in pub_sub.listen():
#         if message['type'] != 'message':
#             continue

#         try:
#             # Decode JSON
#             payload = json.loads(message['data'].decode('utf-8'))

#             # Extract detection info
#             detection = payload.get("detection", {})
#             frame_id = detection.get("frame_id")
#             cls = detection.get("rt_class")
#             conf = detection.get("rt_confidence")
#             bbox = detection.get("rt_bbox")
#             timestamp = detection.get("rt_time")

#             print(f"📦 Detection - Frame: {frame_id}, Class: {cls}, Score: {conf}, BBox: {bbox}, Time: {timestamp}")

#             # Decode image
#             img_base64 = payload.get("image")
#             img_bytes = base64.b64decode(img_base64)
#             img_array = np.frombuffer(img_bytes, dtype=np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#             if img is not None:
#                 # Optional: draw bbox
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 cv2.imshow("Received Image", img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             else:
#                 print("⚠️ Failed to decode image")

#         except Exception as e:
#             print("❌ Error processing message:", e)

#                 # Detic으로 객체 검출
#                 predictions = detic_predictor.predict(frame)

#                 # 검출 결과 처리
#                 detections = []
#                 for pred in predictions:
#                     bbox = pred["bbox"]  # [x_min, y_min, x_max, y_max]
#                     class_name = pred["class_name"]
#                     confidence = pred["confidence"]

#                     # ROI 추출
#                     x_min, y_min, x_max, y_max = map(int, bbox)
#                     roi = frame[y_min:y_max, x_min:x_max]
#                     if roi.size == 0:  # ROI가 비어있으면 스킵
#                         continue

#                     # ROI를 PIL 이미지로 변환
#                     roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                     roi_pil = Image.fromarray(roi_rgb)

#                     # VLM (CLIP)으로 ROI 설명 생성
#                     inputs = clip_processor(
#                         text=["A person walking", "A car on the road", "A dog running", "An empty street"],  # VLM 프롬프트
#                         images=roi_pil,
#                         return_tensors="pt",
#                         padding=True
#                     )
#                     outputs = clip_model(**inputs)
#                     logits_per_image = outputs.logits_per_image
#                     probs = logits_per_image.softmax(dim=1).detach().numpy()
#                     max_prob_idx = np.argmax(probs[0])
#                     vlm_description = inputs["text"][max_prob_idx]

#                     # 검출 결과 저장
#                     detection = {
#                         "rt_class": class_name,
#                         "rt_confidence": float(confidence),
#                         "rt_bbox": [x_min, y_min, x_max, y_max],
#                         "rt_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#                         "event_info": event_info,
#                         "vlm_description": vlm_description
#                     }
#                     detections.append(detection)

#                     # 바운딩 박스와 레이블 그리기
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                     label = f"{class_name} {confidence:.2f} ({vlm_description})"
#                     cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # 결과 리스트에 추가
#                 frame_result = {
#                     "frame_id": len(results_list),
#                     "detections": detections,
#                     "event_info": event_info,
#                     "frame_path": frame_path,
#                     "processing_time": time.time() - start_time
#                 }
#                 results_list.append(frame_result)

#                 # PostgreSQL에 저장
#                 for detection in detections:
#                     cursor.execute("""
#                         INSERT INTO detections (frame_id, rt_class, rt_confidence, rt_bbox, rt_time, event_info, frame_path, vlm_description, processing_time)
#                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
#                     """, (
#                         frame_result["frame_id"],
#                         detection["rt_class"],
#                         detection["rt_confidence"],
#                         json.dumps(detection["rt_bbox"]),
#                         detection["rt_time"],
#                         detection["event_info"],
#                         frame_path,
#                         detection["vlm_description"],
#                         frame_result["processing_time"]
#                     ))
#                 conn.commit()

#                 # 결과 화면 표시
#                     cv2.imshow("Detic + VLM Real-time Detection", frame)

#         # 'q' 키를 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # 처리 간격 조정
#         elapsed_time = time.time() - start_time
#         if elapsed_time < FRAME_PROCESSING_INTERVAL:
#             time.sleep(FRAME_PROCESSING_INTERVAL - elapsed_time)

# except Exception as e:
#     print(f"Error: {e}")

# finally:
#     # 리소스 해제
#     cv2.destroyAllWindows()
#     cursor.close()
#     conn.close()

#     # 결과 JSON 파일로 저장 (추가 확인용)
#     with open("detection_results.json", "w") as f:
#         json.dump(results_list, f, indent=4)