import cv2
import redis
import json
import time
from detic.predictor import DeticPredictor
import numpy as np

# Redis 클라이언트 초기화
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Detic 모델 초기화
detic_predictor = DeticPredictor(
    model_type="Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size",
    confidence_threshold=0.5
)

# Redis Stream 이름
STREAM_NAME = "video_frames_stream"

# 프레임 처리 간격 (초)
FRAME_PROCESSING_INTERVAL = 1.0

# 결과 저장을 위한 리스트
results_list = []

# 마지막으로 처리한 메시지 ID 초기화
last_id = "0-0"

try:
    while True:
        # Redis Stream에서 메시지 읽기 (일정 간격으로 폴링)
        start_time = time.time()
        messages = redis_client.xread({STREAM_NAME: last_id}, block=1000, count=1)

        # 메시지가 있으면 처리
        if messages:
            for stream, message_list in messages:
                for message_id, message in message_list:
                    last_id = message_id  # 마지막 메시지 ID 업데이트

                    # 메시지에서 프레임 위치와 이벤트 정보 추출
                    frame_path = message.get("frame_path")
                    event_info = message.get("event_info", "No event info")

                    if not frame_path:
                        print(f"No frame path in message {message_id}")
                        continue

                    # 프레임 이미지 읽기
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"Failed to read frame from {frame_path}")
                        continue

                    # Detic으로 객체 검출
                    predictions = detic_predictor.predict(frame)

                    # 검출 결과 처리
                    detections = []
                    for pred in predictions:
                        bbox = pred["bbox"]  # [x_min, y_min, x_max, y_max]
                        class_name = pred["class_name"]
                        confidence = pred["confidence"]

                        # 검출 결과 저장 (스키마 형식 참고)
                        detection = {
                            "rt_class": class_name,
                            "rt_confidence": float(confidence),
                            "rt_bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            "rt_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "event_info": event_info
                        }
                        detections.append(detection)

                        # 바운딩 박스와 레이블 그리기
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 결과 리스트에 추가
                    results_list.append({
                        "frame_id": len(results_list),
                        "detections": detections,
                        "event_info": event_info,
                        "frame_path": frame_path,
                        "processing_time": time.time() - start_time
                    })

                    # 결과 화면 표시
                    cv2.imshow("Detic Real-time Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 처리 간격 조정
        elapsed_time = time.time() - start_time
        if elapsed_time < FRAME_PROCESSING_INTERVAL:
            time.sleep(FRAME_PROCESSING_INTERVAL - elapsed_time)

except Exception as e:
    print(f"Error: {e}")

finally:
    # 리소스 해제
    cv2.destroyAllWindows()

    # 결과 JSON 파일로 저장
    with open("detection_results.json", "w") as f:
        json.dump(results_list, f, indent=4)