#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 기반 객체 검출 결과를 FastAPI 서버로 HTTP POST 요청을 보내는 프로듀서
- 영상 입력(웹캠/파일) → YOLO 검출 → ROI 추출(base64 JPEG) → FastAPI POST(/inference)
"""

import os
import sys
import cv2
import numpy as np
import time
import uuid
import json
import base64
import requests
import yaml
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

from ultralytics import YOLO
from clip_caller import verify_with_clip

base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(base_abspath)
# vision_analysis_abspath = f"{base_abspath}/vision_analysis/"
# sys.path.append(vision_analysis_abspath)
from util import *
from vision_analysis.class_names_yolo80n import TEXT_LABELS_80
# -------------------- 설정 --------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# FastAPI 서버 정보
# FASTAPI_HOST = config.get("fastapi", {}).get("host", "localhost")
VLM_HOST = "172.16.15.189"
VLM_PORT = config.get("fastapi", {}).get("port", 8009)
VLM_URL  = f"http://{VLM_HOST}:{VLM_PORT}/inference"

# Video 설정
VIDEO_SOURCE = config["video"]["source"]
CAMERA_ID    = config["video"]["camera_id"]

# YOLO 모델 설정
YOLO_MODEL   = config["yolo_model"]["model_name"]
CONF_THRESH  = float(config["yolo_model"]["conf_thresh"])
ALLOW_CLASSES_STR = config["yolo_model"].get("allow_classes", "").strip()
ALLOW_CLASSES = [c.strip() for c in ALLOW_CLASSES_STR.split(",") if c.strip()] if ALLOW_CLASSES_STR else None
if 'all'==ALLOW_CLASSES_STR:
    ALLOW_CLASSES = False
# Preview
ROI_JPEG_QUALITY = int(config["preview"]["roi_jpeg_quality"])
SHOW_PREVIEW = config["preview"]["show_preview"] == 1
SHOW_PREVIEW = False
TABLE_NAME = config["datadrift_table"]
def get_db_connection():
    # return psycopg2.connect(**config)
    conn = psycopg2.connect(
        dbname=config["postgres"]["dbname"],
        user=config["postgres"]["user"],
        password=config["postgres"]["password"],
        host=config["postgres"]["host"],
        port=config["postgres"]["port"],
        options="-c client_encoding=UTF8",
        cursor_factory=RealDictCursor)
    return conn

from datetime import datetime
import traceback
def save_full_frame_with_annotation(frame_b64: str, 
                                    save_dir: str,
                                    frame_id: str,
                                #    camera_name: str, 
                                #    class_name: str, 
                                #    confidence: float,
                                   bboxes: list, 
                                #    description: str,
                                    #  event_time
                                     ) -> bool:
    """
    전체 프레임 이미지를 저장하고, YOLO 어노테이션 파일을 함께 생성하는 함수.
    """
    import base64
    import cv2
    import numpy as np
    import os
    from datetime import datetime

    try:
        # Create separate directories for images and labels
        images_dir = os.path.join(save_dir, "images")
        labels_dir = os.path.join(save_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        img_path = os.path.join(images_dir, f"{frame_id}.jpg")
        label_path = os.path.join(labels_dir, f"{frame_id}.txt")

        # Base64 → 이미지 디코딩 및 저장
        img_data = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("[Warning] Failed to decode full frame image")
            return False

        cv2.imwrite(img_path, img)
        print(f"[SAVE] Full frame image saved to: {img_path}")

        # 이미지 크기
        h, w = img.shape[:2]
        yolo80_names = list(TEXT_LABELS_80.values())
        # 어노테이션 파일 생성
        with open(label_path, "w", encoding="utf-8") as f:
            # f.write("# Auto-generated annotation\n")
            # f.write(f"class_name: {description}\n")
            # f.write(f"camera_id: {camera_name}\n")
            # f.write(f"confidence: {confidence}\n")
            # f.write(f"event_time: {event_time}\n")

            if isinstance(bboxes, list) and len(bboxes) > 0:
                for bbox in bboxes:
                    confidence, (x1, y1, x2, y2), cls = bbox

                    cls_num = class_choose(cls,yolo80_names)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_num} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                f.write("0 0.5 0.5 1.0 1.0\n")

        print(f"[SAVE] Annotation saved to: {label_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to save frame or annotation: {e}")
        return False

def class_choose(cls_nm, class_names):
    try:
        return class_names.index(cls_nm)
    except ValueError:
        print(f"⚠️ Warning: '{cls_nm}' not found in class_names")
        return -1
    
# def save_full_frame_with_annotation_v2(frame_b64: str, 
#                                     save_dir: str,
#                                     frame_id: str,
#                                 #    camera_name: str, 
#                                 #    class_name: str, 
#                                 #    confidence: float,
#                                    bboxes: list, 
#                                 #    description: str,
#                                     #  event_time
#                                      ) -> bool:
# import json
# import cv2
# for file in names:
#     json_path = os.path.join(label_dir, file+'.Json')
#     img_path = os.path.join(image_dir, file+'.jpg')
#     image = cv2.imread(img_path)
#     h,w,_ = image.shape    # 이미지 처리에서는 이미지 shape를 height, width, channal 순으로 정의한다.
#     with open(json_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#     objs = []
#     for obj in json_data['Bounding']:
#         if obj['Drawing'] != 'BOX':    # bbox가 아닌 polygon 등 다른 방식의 annotation 제외
#             break
#         cls = obj['DETAILS']
#         cls_num = class_choose(cls)
#         x1 = int(obj['x1'])
#         x2 = int(obj['x2'])
#         y1 = int(obj['y1'])
#         y2 = int(obj['y2'])
#         center_x = (x1 + x2) / (2*w)
#         center_y = (y1 + y2) / (2*h)
#         width = (x2 - x1) / w
#         height = (y2 - y1) / h    # YOLO format의 normalized bbox
#         objs.append(f'{cls_num} {center_x} {center_y} {width} {height}\n')
#     with open('path/to/yolo/labels/{file}.txt', 'w') as f:
#         f.writelines(objs)

def db_insert_event(response: dict,frame_b64,bboxes):
    conn = None
    cur = None
    try:
        # 1) 값 병합/전처리
        frame_id    = response.get('frame_id')
        camera_name = response.get('camera_id')
        class_name  = response.get('class')
        event_time  = response.get('event_time')
        confidence  = response.get('confidence')
        vlm_valid   = response.get('vlm_valid')

        # event_time 정규화: item.event_time이 epoch(숫자)이면 datetime으로 변환
        event_time = None #getattr(item, 'event_time', None)
        if isinstance(event_time, (int, float)):
            event_time = datetime.fromtimestamp(event_time)
        elif event_time is None:
            event_time = datetime.now()

        if isinstance(event_time, (int, float)):
            event_time = datetime.fromtimestamp(event_time)
        elif event_time is None:
            event_time = datetime.now()


    
        # 2) DB 연결
        conn = get_db_connection()
        cur = conn.cursor()

        # 3) INSERT: response 필드 포함
        query = f"""
            INSERT INTO {TABLE_NAME} (
                frame_id, camera_name, class_name,
                event_time, confidence, vlm_valid
            ) VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            query,
            (
                # request_id,
                frame_id,
                camera_name,
                class_name,
                # event_name,
                event_time,
                confidence,
                vlm_valid   # ← validation 컬럼
            )
        )
        conn.commit()

        return {"status": "success", "message": "Event inserted successfully!"}

    except Exception as e:
        if conn:
            conn.rollback()
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    
def b64_to_cv2(b64_str: str):
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def main():
    print(f"[m1] start YOLO producer:")# FastAPI={VLM_URL}, camera_id={CAMERA_ID}")
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(0 if VIDEO_SOURCE == "0" else VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    frame_idx = 0
    names = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[m1] video read failed/broken. exiting.")
                break
            frame_idx += 1
            if frame_idx%10!=0:
                # print(frame_idx)
                continue

            ts = time.time()
            frame_id = str(uuid.uuid4())

            results = model(frame, verbose=False, device=0)
            # for result in results:
            #     if names is None:
            #         names = result.names

            #     for b in result.boxes:
            #         x1, y1, x2, y2 = map(int, b.xyxy[0])
            #         conf = float(b.conf)
            #         cls_id = int(b.cls)
            #         cls_name = names.get(cls_id, str(cls_id))

            #         if ALLOW_CLASSES and (cls_name not in ALLOW_CLASSES):
            #             continue

            #         if conf < CONF_THRESH:
            #             continue                    
            # names = results.names

            # Collect boxes above threshold
            bboxes_filtered = [
                (float(b.conf), tuple(map(int, b.xyxy[0])), result.names.get(int(b.cls), str(int(b.cls))))
                for result in results
                for b in result.boxes
                if (
                    float(b.conf) >= CONF_THRESH and
                    (not ALLOW_CLASSES or result.names.get(int(b.cls), str(int(b.cls))) in ALLOW_CLASSES)
                )
            ]

            minmax_boxes = None
            if bboxes_filtered is None or len(bboxes_filtered)==0:
                continue
            # Find min and max boxes
            min_box = min(bboxes_filtered, key=lambda x: x[0])
            max_box = max(bboxes_filtered, key=lambda x: x[0])

            # Combine into one list
            # minmax_boxes = [min_box, max_box]
            minmax_boxes = bboxes_filtered

            #     # Optional: unpack for clarity
            #     conf_min, bbox_min, class_min = min_box
            #     conf_max, bbox_max, class_max = max_box

            #     print(f"[MIN] {class_min}: conf={conf_min:.3f}, bbox={bbox_min}")
            #     print(f"[MAX] {class_max}: conf={conf_max:.3f}, bbox={bbox_max}")
            #     print("\nCombined list:", minmax_boxes)
            # else:
            #     print("[Warning] No detections above threshold")
            false_class_detected = False
            bboxes_filtered_new = []
            for result in minmax_boxes:
                    conf, bbox, cls_name = result
                    x1, y1, x2, y2 = bbox #map(int, bbox.xyxy[0])
                    # roi, bbox = crop_roi(frame, x1, y1, x2, y2, margin=0.15)
                    roi, bbox = crop_roi(frame, x1, y1, x2, y2, margin=0.)
                    if roi is None:
                        continue
                    roi_b64 = bgr_to_b64jpg(roi, quality=ROI_JPEG_QUALITY)

                    # CLIP 검증
                    clip_res = verify_with_clip(
                        # roi_b64, model_name="ViT-L/14@336px", model_size="small"
                        roi_b64, model_name="ViT-B/32", model_size="small"
                    )
                    # cls_name = clip_res["event_name"]
                    msg = {
                        "camera_id": CAMERA_ID,
                        "frame_id": frame_id,
                        "ts": ts,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bboxes": [x1, y1, x2, y2],
                        "roi_b64": roi_b64,
                        "event": clip_res["event_name"],
                        "similarity": clip_res["score"],
                    }
                    if SHOW_PREVIEW:
                        label = f"{cls_name} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - text_h - baseline),
                                      (x1 + text_w, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - baseline),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    #     cv2.imshow("YOLO Producer", frame)
                    #     if cv2.waitKey(1) & 0xFF == ord("q"):
                    #         break                    
                    frame_b64 = bgr_to_b64jpg(frame)
                    # Redis 대신 FastAPI로 전송
                    try:
                        resp = requests.post(VLM_URL, json=msg, timeout=10)
                        if resp.status_code == 200:
                            print(f"[POST] Sent detection -> {cls_name}, response: {resp.json()}")
                            resp_dict = resp.json()
                            if 'yes'!=resp_dict['vlm_valid']:
                                false_class_detected = True
                            else:
                                bboxes_filtered_new.append((conf, bbox, cls_name))
                            db_insert_event(resp.json(), frame_b64,minmax_boxes)
                        else:
                            print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"[HTTP ERROR] {e}")
                    # 저장 경로 결정
            save_dir = os.path.join(base_abspath, "datasets","collected" ,"good_images" if False==false_class_detected else "wronged_images")

            # 이미지 및 어노테이션 저장
            if not frame_b64:
                print("[Warning] No frame_b64 found in response")
                return {"status": "error", "message": "No frame_b64 provided"}

            ok = save_full_frame_with_annotation(
                frame_b64=frame_b64,
                save_dir=save_dir,
                frame_id=frame_id,
                # camera_name=camera_name,
                # class_name=class_name,
                # confidence=confidence,
                bboxes=bboxes_filtered_new,
                # description=description,
                # event_time=event_time,
                )

            if not ok:
                # return {"status": "error", "message": "Failed to save frame or annotation"}
                print("error!!!, Failed to save frame or annotation")

            # return {"status": "success", "message": f"Saved to {save_dir}"}
            print(f"success Saved to {save_dir}")


    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
