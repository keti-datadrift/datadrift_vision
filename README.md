# datadrift_vision

## 주소
- https://github.com/keti-datadrift/datadrift_vision.git

## 개요
- 데이터 드리프트 관리 기술의 기반 프레임워크입니다.
- 개발 및 유지 관리 기관 : __(주)인텔리빅스__
- 최종 검토 기관 : 한국전자기술연구원(KETI)

## Acknowledgements (사사)
- 이 연구는 2024년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No. RS-2024-00337489, 분석 모델의 성능저하 극복을 위한 데이터 드리프트 관리 기술 개발)
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00337489, Development of data drift management technology to overcome performance degradation of AI analysis models)

## 시스템 구성

DriftVision2는 YOLOv8 기반 객체 검출 모델의 데이터 드리프트를 자동으로 감지하고 모델을 재학습하는 시스템입니다.

### 주요 기능

- **실시간 객체 검출**: YOLOv8 모델을 사용한 실시간 비디오/카메라 객체 검출
- **자동 드리프트 감지**: VLM(Vision Language Model)을 활용한 검출 결과 검증 및 드리프트 감지
- **자동 모델 재학습**: 드리프트 감지 시 자동으로 새로운 데이터로 모델 재학습
- **모델 성능 평가**: 재학습된 모델의 mAP 평가 및 자동 배포 결정
- **PostgreSQL 기반 데이터 관리**: 검출 결과 및 검증 데이터 저장

### 시스템 아키텍처

```
┌─────────────────┐
│  Video Source   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  YOLO Detection │ ───► │  PostgreSQL  │
│   (FastAPI)     │      │   Database   │
└────────┬────────┘      └──────┬───────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐      ┌──────────────┐
│ VLM Verification│ ◄─── │    Drift     │
│   (CLIP/LLM)    │      │   Checker    │
└─────────────────┘      └──────┬───────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │   Retrain    │
                         │   (YOLO)     │
                         └──────────────┘
```

## 프로젝트 구조

```
drift_vision2/
├── config.yaml                 # 메인 설정 파일
├── vision_analysis/            # 객체 검출 및 분석
│   ├── yolo_producer_fastapi.py   # YOLO 실시간 검출 서버 (FastAPI)
│   └── clip_verifier.py           # CLIP 기반 검증
├── retrain/                    # 모델 재학습
│   ├── train_model.py             # YOLO 모델 학습
│   └── evaluate_model.py          # 모델 성능 평가
├── cron/                       # 스케줄러
│   ├── drift_scheduler.py         # 드리프트 체크 스케줄러
│   ├── drift_checker.py           # 드리프트 감지 로직
│   └── cron_config.yaml          # Cron 설정
├── dbmanager/                  # 데이터베이스 관리
│   └── postgres_manager.py        # PostgreSQL 연결 및 쿼리
├── model/                      # YOLO 모델 저장소
├── datasets/                   # 학습 데이터셋
└── logs/                       # 로그 파일
```

## 설치 및 환경 설정

### 필수 요구사항

- Python 3.8+
- PostgreSQL 12+
- CUDA 11.8+ (GPU 사용 시)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/keti-datadrift/datadrift_vision.git
cd drift_vision2
```

2. **Python 패키지 설치**
```bash
pip install -r datadrift_venv_requirements.txt
```

3. **PostgreSQL 데이터베이스 설정**
```bash
# PostgreSQL 접속
psql -U postgres

# 데이터베이스 생성
CREATE DATABASE postgres;
```

4. **config.yaml 설정**
```yaml
postgres:
  host: 127.0.0.1
  port: 5432
  dbname: postgres
  user: postgres
  password: postgres

yolo_model:
  original_model_name: .\model\yolov8n_gdr.pt
  conf_thresh: 0.5
  criteria_classes: person
```

자세한 설정 내용은 [CONFIG_MIGRATION.md](CONFIG_MIGRATION.md)를 참조하세요.

## 사용 방법



### 1. 드리프트 스케줄러 실행

```bash
cd datadrift_vision
python cron/drift_scheduler.py
```

설정된 주기마다 자동으로 드리프트를 감지를 요청하고 필요시 재학습을 트리거합니다.

### 2. YOLO 검출 서버 실행

```bash
cd datadrift_vision
python vision_analysis/yolo_producer_fastapi.py
```

서버는 `http://0.0.0.0:18880`에서 실행됩니다.

### 3. 수동 모델 재학습 및 자동평가

```bash
cd datadrift_vision
python dbmanager/db_api_server.py
```
요청될때마다 드리프트 발생여부 확인, 및 필요 시 재합습 실행합니다.

### 4. 수동 모델 재학습 및 자동평가

```bash
cd datadrift_vision
python retrain/train_model.py
```




## 주요 설정 항목

### 객체 검출 설정

```yaml
yolo_model:
  original_model_name: .\model\yolov8n_gdr.pt  # 원본 YOLO 모델
  updated_model_name: .\model\yolov8n_gdr_v2.pt  # 재학습된 모델
  use_original_model: false  # true: 원본 사용, false: 재학습 모델 사용
  conf_thresh: 0.5  # 객체 검출 confidence 임계값
  criteria_classes: person  # 드리프트 감지 대상 클래스
```

### 드리프트 감지 설정

```yaml
drift_detection:
  drift_check_period: 1 days  # 드리프트 분석 기간
  drift_check_interval_minutes: 60  # 체크 주기 (분)
  drift_threshold: 0.01  # 드리프트 임계값 (false_ratio)
  cooldown_after_update: 30 minutes  # 재학습 후 쿨다운
```

### 모델 업데이트 설정

```yaml
model_update:
  overall_map_threshold: 0.0  # 전체 mAP 향상도 임계값
  criteria_class_map_threshold: 0.01  # 핵심 클래스 mAP 향상도 임계값
  auto_use_updated_model: true  # 자동 모델 전환 여부
  monitor_train: false  # 학습 과정 모니터링
```

### 학습 설정

```yaml
training:
  use_previous_model_finetune: true  # true: 파인튜닝, false: 처음부터 학습
  finetune_lr0: 0.001  # 파인튜닝 시작 learning rate
  finetune_lrf: 0.01  # 파인튜닝 최종 learning rate 비율
  fresh_lr0: 0.002  # Fresh 학습 시작 learning rate
  fresh_lrf: 0.05  # Fresh 학습 최종 learning rate 비율
```

자세한 threshold 설명은 [THRESHOLDS_EXPLAINED.md](THRESHOLDS_EXPLAINED.md)를 참조하세요.

## API 엔드포인트

### FastAPI 서버 (`vision_analysis/yolo_producer_fastapi.py`)

- 실시간 비디오 스트리밍

### FastAPI 서버 (`dbmanager/db_api_server.py.py`)
- `POST /api/db_check_drift/`: 드리프트 체크 트리거
- `GET /api/config_reload/`: 설정 파일 리로드 (미완성)
- `GET /health`: 헬스체크 (미완성)

### 사용 예시

```bash
# 드리프트 체크 트리거
curl -X POST http://localhost:18880/api/db_check_drift/

# 설정 리로드
curl http://localhost:18880/api/config_reload/

# 헬스체크
curl http://localhost:18880/health
```

## 데이터베이스 스키마

### datadrift_db 테이블

```sql
CREATE TABLE datadrift_db (
    id SERIAL PRIMARY KEY,
    detection_id VARCHAR(255),
    camera_id VARCHAR(50),
    timestamp TIMESTAMP,
    class_name VARCHAR(50),
    confidence FLOAT,
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,
    roi_image_path TEXT,
    is_valid BOOLEAN,
    validation_method VARCHAR(50),
    validation_timestamp TIMESTAMP,
    false_reason TEXT
);
```

테이블은 자동으로 일별 파티션이 생성됩니다.

## 모델 관리

### 모델 전환

config.yaml에서 간단히 전환 가능:

```yaml
yolo_model:
  use_original_model: true  # 원본 모델 사용
  # use_original_model: false  # 재학습 모델 사용
```

### 모델 롤백

문제 발생 시 즉시 이전 모델로 복원:

```yaml
yolo_model:
  use_original_model: true  # 원본으로 롤백
```

### A/B 테스트

여러 카메라에서 다른 모델을 사용하여 성능 비교 가능합니다.

## 모니터링 및 로그

### 로그 파일 위치

- `logs/drift_scheduler.log`: 드리프트 스케줄러 로그
- `logs/yolo_detection.log`: 객체 검출 로그
- `logs/training.log`: 모델 학습 로그

### 학습 모니터링

```yaml
model_update:
  monitor_train: true  # 학습 과정 실시간 모니터링
```

활성화 시 학습 중 실시간으로 loss, mAP 등을 확인할 수 있습니다.

## 트러블슈팅

### 자주 발생하는 문제

#### 1. 재학습했는데 모델이 배포되지 않음

**증상**: "Overall mAP improvement below threshold" 로그 출력

**해결책**:
- `overall_map_threshold`를 낮추거나 음수로 설정
- 더 많은 학습 데이터 수집 후 재학습
- learning rate 조정

#### 2. 드리프트가 너무 자주 감지됨

**증상**: 재학습이 너무 빈번하게 발생

**해결책**:
```yaml
drift_detection:
  drift_threshold: 0.15  # 임계값 상향 (기본: 0.01)
  drift_check_interval_minutes: 180  # 체크 주기 증가 (기본: 60)
```

### 로그 분석

주요 로그 메시지:

```
✅ Drift detected! → 드리프트 감지됨, 재학습 시작
✅ Model deployed → 새 모델 배포 성공
❌ Overall mAP improvement below threshold → mAP 향상 부족, 배포 취소
🔄 Fine-tuning mode → 파인튜닝 모드로 학습 중
🆕 Fresh training mode → 처음부터 학습 중
```

## 설정 마이그레이션

시스템 업데이트 시 설정 변경사항은 [CONFIG_MIGRATION.md](CONFIG_MIGRATION.md)를 참조하세요.

주요 변경 이력:
- 2025-11-13: 학습 설정 통합 및 중복 제거
- 2025-11-12: Database 설정 분리
- 2025-11-11: Drift 설정 통합 및 모델 관리 개선

## 성능 최적화

### 권장 하드웨어

- **CPU**: Intel Core i7 이상 또는 AMD Ryzen 7 이상
- **RAM**: 16GB 이상
- **GPU**: NVIDIA RTX 3060 이상 (VRAM 6GB+)
- **Storage**: SSD 100GB 이상

### 최적화 팁

1. **GPU 사용 활성화**
```python
# CUDA 사용 가능 확인
import torch
print(torch.cuda.is_available())
```

2. **배치 처리 최적화**
```yaml
# 학습 시 배치 크기 조정
training:
  batch_size: 16  # GPU 메모리에 맞게 조정
```

3. **데이터베이스 인덱스 추가**
<!-- ```sql
CREATE INDEX idx_timestamp ON datadrift_db(timestamp);
CREATE INDEX idx_camera_id ON datadrift_db(camera_id);
``` -->

## 기여하기

프로젝트에 기여하려면:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 라이선스를 따릅니다.

## 연락처

- 개발 및 유지 관리: (주)인텔리빅스
- 최종 검토 기관: 한국전자기술연구원(KETI)
- 이슈 및 문의: [GitHub Issues](https://github.com/keti-datadrift/datadrift_vision/issues)

## 참고 자료

- [THRESHOLDS_EXPLAINED.md](THRESHOLDS_EXPLAINED.md) - Threshold 파라미터 상세 설명
- [CONFIG_MIGRATION.md](CONFIG_MIGRATION.md) - 설정 변경 이력 및 마이그레이션 가이드
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
