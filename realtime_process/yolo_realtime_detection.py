import cv2
from ultralytics import YOLO
import json
import time

# YOLOv8 모델 로드 (사전 학습된 모델 사용, 예: yolov8n.pt)
model = YOLO("yolov8n.pt")

# 비디오 스트림 소스 (웹캠: 0, 비디오 파일: "path/to/video.mp4")
# 웹캠 사용 예시
cap = cv2.VideoCapture(0)

# 비디오 파일 사용 예시
# cap = cv2.VideoCapture("path/to/video.mp4")

# 결과 저장을 위한 리스트
results_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8으로 객체 검출
    start_time = time.time()
    results = model(frame)
    end_time = time.time()

    # 검출 결과 처리
    detections = []
    for result in results:
        boxes = result.boxes  # 바운딩 박스 정보
        for box in boxes:
            # 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = float(box.conf)  # 신뢰도
            class_id = int(box.cls)  # 클래스 ID
            class_name = model.names[class_id]  # 클래스 이름

            # 검출 결과 저장 (스키마 형식 참고)
            detection = {
                "rt_class": class_name,
                "rt_confidence": confidence,
                "rt_bbox": [x_min, y_min, x_max, y_max],
                "rt_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            detections.append(detection)

            # 바운딩 박스와 레이블 그리기
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 리스트에 추가
    results_list.append({
        "frame_id": len(results_list),
        "detections": detections,
        "processing_time": end_time - start_time
    })

    # 결과 화면 표시
    cv2.imshow("YOLOv8 Real-time Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 결과 JSON 파일로 저장 (필요 시)
with open("detection_results.json", "w") as f:
    json.dump(results_list, f, indent=4)