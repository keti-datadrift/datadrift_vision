#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kanana VLM 기반 검증 모듈 (FastAPI 버전)
- 입력: YOLO Producer에서 HTTP POST (/inference)
- 처리: VLM 모델을 통해 검증 수행
- 출력: JSON 응답 (검증 결과 반환)
"""

import os
import sys
import io
import json
import base64
import time
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
import traceback
import torch
import uvicorn
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(base_abspath)

vision_analysis_abspath = f'{base_abspath}/vision_analysis/'
sys.path.append(vision_analysis_abspath)

dbmanager_abspath = f'{base_abspath}/dbmanager/'
sys.path.append(dbmanager_abspath)

from dbmanager.create_table import connect_db
# -----------------------------------------------------
# 환경 및 모델 설정
# -----------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"

# 4bit 로딩 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 모델 로드
print("[VLM] Loading Kanana model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 프로세서 로드
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print("[VLM] Model and processor ready.")

# -----------------------------------------------------
# 설정 로드
# -----------------------------------------------------
if os.path.exists("config.yaml"):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}

USE_VLM = config.get("vlm", {}).get("use_vlm", True)
WINDOW_SEC = int(config.get("vlm", {}).get("window_sec", 30))

# -----------------------------------------------------
# FastAPI 초기화
# -----------------------------------------------------
app = FastAPI(title="VLM Verifier Service", version="2.0.0")

# -----------------------------------------------------
# 유틸 함수
# -----------------------------------------------------
def b64_to_pil(b64_str: str) -> Image.Image:
    """Base64 문자열을 PIL 이미지로 변환"""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def detection_question(det_class: str) -> str:
    """클래스명에 따라 적절한 질문 생성"""
    # qmap = {
    #     "fire": "불꽃이 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    #     "smoke": "연기가 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    #     "person": "사람이 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    #     # "person": "'남자', '여자','모름'으로 대답하고, 이유를 성명하세요.",
    #     "motorbike": "오토바이가 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    # }
    qmap = {
    # "fire": "불꽃이 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    # "smoke": "연기가 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    # "person": "사람이 보이나요? '네', '아니요'로 대답하고, '아니요'이면 무엇인가요?.",
    # "person": "'남자', '여자','모름'으로 대답하고, 이유를 성명하세요.",
    # "motorbike": "오토바이가 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.",
    # f"{det_class}": f"is this {det_class}? 'yes' or 'no', explain the reason."
    f"{det_class}": f"{det_class}가 보이나요? '네' or '아니요', 이유를 설명하세요."
    }
    print(qmap)
    # return qmap.get(det_class.lower(), f"{det_class}가 보이나요? '네', '아니요'로 대답하고, 이유를 성명하세요.")
    return qmap.get(det_class.lower(), f"{det_class}: is this {det_class}? 'yes' or 'no', explain the reason.")

def ask_kanana_yesno(pil_image: Image.Image, question: str):
    """Kanana 모델에 질의하여 Yes/No 응답 추론"""
    sample = {
        "image": [pil_image],
        "conv": [
            {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": question},
        ]
    }
    batch = [sample]
    inputs = processor.batch_encode_collate(batch, padding_side="left", add_generation_prompt=True, max_length=8192)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        do_sample=False
    )
    text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    low = text.lower()
    print(low)
    # 결과 판별
    if ("네" in text) and ("아니" not in text): return 'yes', text
    if ("아니" in text): return 'no', text
    if ("yes" in low) and ("no" not in low): return 'yes', text
    if ("no" in low): return 'no', text
    return 'unknown', text

# -----------------------------------------------------
# 요청 데이터 구조 정의
# -----------------------------------------------------
class InferenceRequest(BaseModel):
    camera_id: str
    frame_id: str
    ts: float
    class_name: str
    confidence: float
    bboxes: list
    roi_b64: str
    event: str | None = None
    similarity: float | None = None

# -----------------------------------------------------
# FastAPI 엔드포인트 정의
# -----------------------------------------------------
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "use_vlm": USE_VLM}

@app.post("/inference")
async def inference(req: InferenceRequest):
    """YOLO Producer에서 객체 검출 결과 수신 → VLM 검증 수행"""
    start_time = time.time()
    # print(req)
    print(req.class_name)
    # conn, cur = connect_db() 
    print(f'=================== inference: {req.frame_id} ===================')   
    try:
        # ROI 복원
        pil_img = b64_to_pil(req.roi_b64)

        # 질의문 생성
        question = detection_question(req.class_name)

        # Kanana 모델 실행
        result, answer_text = ask_kanana_yesno(pil_img, question)
        # print(result['frame_id'])
        print(answer_text)

        elapsed = round(time.time() - start_time, 3)
        print(elapsed)
        # # DB 저장
        # query = """
        #     INSERT INTO event_validation_log
        #     (request_id, event_name, validation, event_time, camera_name)
        #     VALUES (%s, %s, %s, to_timestamp(%s), %s)
        # """
        # cur.execute(query, (
        #     int(req.frame_id),         # request_id
        #     req.event or req.class_name,  # event_name
        #     result,                    # validation
        #     req.ts,                    # timestamp (float → to_timestamp)
        #     req.camera_id              # camera_name
        # ))
        # conn.commit()
        if 'no'==result:
            print(f'****************************** no!!! {req.frame_id} ******************************')

        temp_dict = {
            "camera_id": req.camera_id,
            "frame_id": req.frame_id,
            "class": req.class_name,
            "confidence": req.confidence,
            "event_name": req.event,
            # "event_time": ???,
            "confidence": req.confidence,
            "vlm_valid": result
            # "answer_text": answer_text,
            # "elapsed_sec": elapsed,
        }
        print(temp_dict)
        return temp_dict
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# -----------------------------------------------------
# 실행 명령 예시
# -----------------------------------------------------
# uvicorn vlm_verifier:app --host 0.0.0.0 --port 8000

# Example Test
# curl -X POST "http://localhost:8000/inference" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "camera_id": "cam01",
#     "frame_id": "abcd-1234",
#     "ts": 1734105600,
#     "class_name": "fire",
#     "confidence": 0.97,
#     "bboxes": [100,150,300,400],
#     "roi_b64": "<base64 string>",
#     "event": "fire",
#     "similarity": 0.93
#   }'

# Expected response:
# {
#   "camera_id": "cam01",
#   "frame_id": "abcd-1234",
#   "class": "fire",
#   "confidence": 0.97,
#   "event": "fire",
#   "similarity": 0.93,
#   "result": true,
#   "answer_text": "네, 불꽃이 보입니다.",
#   "elapsed_sec": 2.153
# }

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn.run(app, host="localhost", port=8009)
    uvicorn.run(app, host="0.0.0.0", port=8009)    