# VLM semantic drift sample

이 디렉터리는 VLM을 활용한 **의미적 드리프트 후보 검토**를 설명하기 위한
독립 교육용 샘플의 골격입니다. 저장소의 Apache-2.0 라이선스를 따릅니다.

## 현재 구현 범위: PR-01

- 부작용 없는 Python 패키지 선언
- JSON 설정 예제 및 구조·기본값 검증 테스트
- 외부 의존성 없는 import 스모크 테스트

Python 3.8 이상을 대상으로 하며 표준 라이브러리만 사용합니다.
GPU, PostgreSQL, 카메라, 모델 가중치, API 키, 네트워크 연결 및 패키지 설치가
필요하지 않습니다. 루트 requirements와 config.yaml을 사용하지 않습니다.
기존 vision_analysis, cron, retrain, dbmanager 모듈도 import하지 않습니다.

**MockVLM은 아직 구현되지 않았습니다.** PR-06에서 결정적인 모의 응답을 제공할
예정이며, 실제 VLM의 인식 성능을 나타내지 않습니다. 이벤트 검증, ROI 추출,
품질 평가, 프롬프트, 응답 정규화, 시간 누적, 후보 판단, 피드백 및 리포트도
후속 작업입니다. 전체 실행 진입점 run_demo.py는 PR-12 범위입니다.

## 실행 및 검증

저장소 루트에서 실행합니다.

```shell
python -B -S -c "import examples.vlm_semantic_drift"
python -B -m unittest discover -s examples/vlm_semantic_drift/tests -p "test_*.py" -v
```

첫 명령은 출력 없이 종료되면 import 성공입니다. `-B`는 Python의 바이트코드
캐시 생성을 막고, `-S`는 site-packages 초기화를 생략합니다. 테스트는 별도
프로세스에서도 import를 검사하며 파일 쓰기, 외부 서비스 모듈 import,
네트워크 및 프로세스 시작 시도를 거부합니다. 테스트용 임시 폴더는 정리됩니다.

설정은 명시적으로 읽을 때만 접근합니다. 패키지 import는 설정을 읽지 않습니다.
다음 예제는 저장소 루트에서 설정을 출력할 뿐 추론이나 판단을 실행하지 않습니다.

```python
import json
from pathlib import Path
import examples.vlm_semantic_drift as sample

config_path = Path(sample.__file__).parent / "config.example.json"
with config_path.open(encoding="utf-8") as stream:
    config = json.load(stream)
print(config)
```

## 설정 계약

PR-01은 아래 설정 파일의 계약을 테스트합니다. 사용자 설정을 받는 런타임
로더·검증기는 아직 없습니다. 시간 누적 설정은 후속 구현을 위한 제안값이며
현재 어떤 판단에도 적용되지 않습니다.

| 필드 | 기본값 | 의미 및 유효 범위 |
|---|---:|---|
| schema_version | 1 | 설정 계약 버전, 현재는 정수 1 |
| vlm.provider | mock | 향후 기본 어댑터, 현재 계약에서는 mock만 허용 |
| aggregation.window_size | 10 | 그룹별 보관할 최근 이벤트 수, 양의 정수 |
| aggregation.min_valid_samples | 5 | 판정 가능한 최소 표본 수, 2 이상이며 window_size 이하 |
| aggregation.mismatch_ratio_threshold | 0.6 | 유효 표본 중 NO 비율 기준, 0 초과 1 이하 |
| aggregation.consecutive_mismatches | 3 | 연속 NO 기준, 2 이상이며 min_valid_samples 이하 |

임계값은 실측 성능으로 정한 값이 아닌 **교육용 예제값**입니다. 후속 구현에서는
최근 window_size개 이벤트 중 품질을 통과하고 YES/NO로 판정된 표본만 비율의
분모에 포함하는 방식을 제안합니다. 최소 표본 수, 비율, 연속 NO 조건을 모두
충족했을 때 후보를 표시하며 UNKNOWN 또는 품질 문제는 연속 NO를 끊습니다.
최종 시간·그룹 처리 및 판단 규칙은 PR-08/09에서 테스트와 함께 확정합니다.

## 가설과 응답의 의미: 후속 구현 계약

계획된 흐름은 이벤트 ROI와 검출 정보 → 품질 점검 → H0/H1 질문 → VLM 어댑터
→ 응답 정규화 → 시간 누적 → 후보 리포트 → 선택적 사용자 피드백입니다.

- H0: 검출기가 붙인 이벤트 라벨이 사용 가능한 시각 증거와 일치합니다.
- H1: 라벨이 시각 증거와 불일치하거나 일치 여부가 불확실하여 검토가 필요합니다.

질문은 항상 **“이벤트 라벨이 시각 증거와 일치하는가?”**로 해석합니다.
다음은 향후 정규화 계약이며 지금 생성되는 출력은 아닙니다.

| 응답 | label_match | supports_h1 | 누적 처리 |
|---|---|---|---|
| YES | true | false | 유효한 일치 표본 |
| NO | false | true | 유효한 불일치 표본 |
| UNKNOWN | null | null | 판단 유보, 불일치 비율의 분자·분모에서 제외 |

UNKNOWN은 H1의 불확실성 검토 대상이지만 확정 불일치의 증거는 아닙니다.
supports_h1의 null은 지지 여부를 결정할 수 없다는 뜻이며 false와 다릅니다.
Python에서는 JSON null을 None으로 표현합니다. 응답에는 향후 provider와
is_mock=true를 포함하여 모의 결과임을 구분할 예정입니다.

블러, 저노출, 눈부심·역광 등의 품질 문제는 별도 quality_issue로 표시하고
의미적 불일치로 집계하지 않습니다. 품질 평가 자체는 아직 구현되지 않았습니다.
누적 결과의 예정 상태는 no_alert, quality_issue, drift_candidate입니다.
no_alert는 정상임을 증명하지 않으며, drift_candidate는 사람의 추가 검토가
필요하다는 신호입니다. 단일 NO 또는 UNKNOWN은 데이터 드리프트의 증명이
아니며, 자동 재학습이나 모델 배포를 실행하지 않습니다.

## 데이터 및 공개 시 유의점

현재 입력 영상·이벤트 데이터는 포함하지 않습니다. 후속 테스트는 합성 자료를
사용하며 실제 현장 영상, 개인정보, 내부 주소, 인증정보, 모델 파일을 포함하지
않습니다. Mock 결과를 실제 정확도나 성능 향상 수치로 보고하지 않습니다.

루트 .gitignore의 README.md 규칙은 이 파일도 제외합니다. 공개 준비 시 이
파일의 누락 여부를 확인해야 합니다. 승인된 스테이징 단계에서 해당 파일만
명시적으로 강제 추가할 수 있으며, PR-01 개발 단계에서는 스테이징하지 않습니다.
