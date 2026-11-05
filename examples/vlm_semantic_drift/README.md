# VLM semantic drift sample

이 디렉터리는 VLM을 활용한 **의미적 드리프트 후보 검토**를 설명하기 위한
독립 교육용 샘플의 골격입니다. 저장소의 Apache-2.0 라이선스를 따릅니다.

## 현재 구현 범위: PR-01 / PR-02

- 부작용 없는 Python 패키지 선언
- JSON 설정 예제 및 구조·기본값 검증 테스트
- 외부 의존성 없는 import 스모크 테스트
- 불변 DetectionEvent 입력 스키마, 값 검증 및 JSON 왕복 변환
- 합성 이벤트 메타데이터 2건과 유효·무효 입력 테스트

Python 3.8 이상을 대상으로 하며 표준 라이브러리만 사용합니다.
GPU, PostgreSQL, 카메라, 모델 가중치, API 키, 네트워크 연결 및 패키지 설치가
필요하지 않습니다. 루트 requirements와 config.yaml을 사용하지 않습니다.
기존 vision_analysis, cron, retrain, dbmanager 모듈도 import하지 않습니다.

**MockVLM은 아직 구현되지 않았습니다.** PR-06에서 결정적인 모의 응답을 제공할
예정이며, 실제 VLM의 인식 성능을 나타내지 않습니다. ROI 추출,
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

## 이벤트 입력 계약 (PR-02)

`event_schema.DetectionEvent`는 검출기 주장 한 건을 나타내는 불변 dataclass입니다.
검출 결과가 시각 증거와 일치하는지는 아직 판단하지 않습니다. 객체 생성,
`from_dict`, `from_json` 모두 동일한 값 검증을 수행하고 잘못된 값은 ValueError를
발생시킵니다. 필수 생성자 인수 누락 등 Python 호출 오류는 TypeError입니다.

| 필드 | 의미와 규칙 |
|---|---|
| schema_version | JSON 입력 필수, 정수 1. Python 생성자 기본값 1 |
| event_id | 필수 비어 있지 않은 문자열, 이벤트 식별자 |
| timestamp | 필수 UTC ISO 8601, `YYYY-MM-DDTHH:MM:SS[.ffffff]Z` 또는 `+00:00` |
| camera_id | 필수 비어 있지 않은 문자열, 합성 카메라/채널 식별자 |
| class_label | 필수 비어 있지 않은 문자열, 검출 객체 클래스. 현재 가설 검토 대상 라벨 |
| confidence | 검출기 신뢰도, 유한한 수 0~1. VLM confidence와 별개 |
| bbox | 필수 `[x1, y1, x2, y2]`, 원본 이미지 기준 픽셀 좌표 |
| image_path | 선택적 상대 이미지 경로. 생략 또는 null 허용 |

문자열 ID와 라벨은 앞뒤 공백·제어문자를 거부합니다. 클래스 목록은 고정하지
않습니다. JSON의 누락·미지정 필드와 중복 키는 거부합니다. bool은 정수/실수로
인정하지 않으며 NaN·Infinity도 거부합니다. timestamp는 달력상 유효해야 하며
시간대 누락·비UTC 오프셋·윤초를 허용하지 않습니다. 출력은 UTC Z 형태로
정규화하며 소수 초는 최대 6자리입니다. 이벤트 간 중복 ID·시각 순서는 이
단일 이벤트 스키마에서 검사하지 않습니다.

bbox는 좌상단 포함·우하단 제외인 xyxy 좌표 계약입니다. 유한한 정수/실수를
허용하고 x1 < x2, y1 < y2여야 합니다. 정규화된 0~1 좌표가 아닙니다.
음수 및 이미지 경계 밖 좌표는 허용합니다. 이미지 크기를 알 수 없는 입력
단계에서는 클램프하지 않으며, PR-03에서 이미지 크기에 맞춰 처리할 예정입니다.
내부 bbox는 tuple로 복사되어 입력 리스트나 내보낸 dict 수정의 영향을 받지 않습니다.

image_path는 `/` 구분자의 상대경로만 허용하며 절대경로, URL, 드라이브 문자,
역슬래시, 빈 경로 요소, `.` 및 `..`를 거부합니다. 경로는 호출자가 명시하는
이미지 루트 기준으로 해석할 예정입니다. 이 모듈은 경로를 해석하거나 파일을
열지 않으며 존재 여부·이미지 형식·심볼릭 링크 안전성도 검사하지 않습니다.
이미지 로딩 단계에서 실제 루트 경계를 확인해야 합니다.

저장소 루트에서 실행할 수 있는 합성 입력 예제입니다.

```python
import json
from pathlib import Path
from examples.vlm_semantic_drift.event_schema import DetectionEvent

path = Path("examples/vlm_semantic_drift/sample_data/events.json")
with path.open(encoding="utf-8") as stream:
    events = [DetectionEvent.from_dict(item) for item in json.load(stream)]
for event in events:
    print(event.to_json())
    assert DetectionEvent.from_json(event.to_json()) == event
```

`sample_data/events.json`은 이 샘플을 위해 손으로 작성한 합성 메타데이터이며
실제 검출·카메라·현장 기록이 아닙니다. 첫 이벤트는 이미지 참조가 없고, 두 번째의
`synthetic/frame-002.png`는 실제 파일이 없는 설명용 참조입니다. 영상은 포함하지
않으며 이미지 없이도 검증과 JSON 변환을 실행할 수 있습니다.

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

현재 합성 이벤트 메타데이터만 포함하며 입력 영상은 없습니다. 후속 테스트는 합성 자료를
사용하며 실제 현장 영상, 개인정보, 내부 주소, 인증정보, 모델 파일을 포함하지
않습니다. Mock 결과를 실제 정확도나 성능 향상 수치로 보고하지 않습니다.

이 README는 PR-01에서 Git 추적에 포함되었습니다. 루트 .gitignore의 README.md
규칙은 이미 추적 중인 이 파일의 변경에는 적용되지 않습니다.
