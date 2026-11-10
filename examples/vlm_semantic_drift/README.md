# VLM semantic drift sample

이 디렉터리는 VLM을 활용한 **의미적 드리프트 후보 검토**를 설명하기 위한
독립 교육용 샘플의 골격입니다. 저장소의 Apache-2.0 라이선스를 따릅니다.

## 현재 구현 범위: PR-01 / PR-02 / PR-03 / PR-04 / PR-05

- 부작용 없는 Python 패키지 선언
- JSON 설정 예제 및 구조·기본값 검증 테스트
- 외부 의존성 없는 import 스모크 테스트
- 불변 DetectionEvent 입력 스키마, 값 검증 및 JSON 왕복 변환
- 합성 이벤트 메타데이터 2건과 유효·무효 입력 테스트
- 메모리 이미지와 DetectionEvent를 사용하는 경계 제한 ROI 추출
- 밝기·노출·경계 정보 기반 품질 후보 평가와 판단 유보
- 검출 라벨·선택적 품질 정보를 이용하는 H0/H1 검증 프롬프트 생성

Python 3.8 이상을 대상으로 합니다. 패키지와 이벤트 스키마 import는 표준
라이브러리만 사용하며, ROI 추출·품질 분석 및 전체 테스트에는 Pillow 설치가 필요합니다.
설치 후 실행에는 GPU, PostgreSQL, 카메라, 모델 가중치, API 키, 네트워크 연결이
필요하지 않습니다. 루트 requirements와 config.yaml을 사용하지 않습니다.
기존 vision_analysis, cron, retrain, dbmanager 모듈도 import하지 않습니다.

**MockVLM은 아직 구현되지 않았습니다.** PR-06에서 결정적인 모의 응답을 제공할
예정이며, 실제 VLM의 인식 성능을 나타내지 않습니다.
응답 정규화, 시간 누적, 드리프트 후보 판단, 피드백 및 리포트도
후속 작업입니다. 전체 실행 진입점 run_demo.py는 PR-12 범위입니다.

## 실행 및 검증

저장소 루트에서 별도 가상환경을 생성해 실행합니다. 아래는 Windows PowerShell
예제입니다. 기존 .venv가 있다면 덮어쓰지 말고 별도의 환경을 사용하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r examples/vlm_semantic_drift/requirements-demo.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s examples/vlm_semantic_drift/tests -p "test_*.py" -v
```

가상환경 활성화 후에는 다음 명령도 사용할 수 있습니다.

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
단계에서는 클램프하지 않으며, ROI 추출 함수가 이미지 크기에 맞춰 처리합니다.
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

## ROI 추출 (PR-03)

`extract_roi(image, event) -> ROIResult`는 메모리에 로드된 Pillow `Image.Image`와
PR-02의 `DetectionEvent`를 입력으로 받습니다. `event.image_path`는 읽거나
탐색하지 않습니다. 파일을 사용한다면 호출자가 미리 로드해야 합니다.
이미지의 현재 픽셀 좌표계와 bbox 좌표계가 같아야 하며 EXIF 회전, 색 변환,
리사이즈는 자동으로 수행하지 않습니다.

결과 필드는 다음과 같습니다.

| 필드 | 내용 |
|---|---|
| image | 원본과 픽셀 저장 공간을 공유하지 않는 Pillow ROI 이미지 |
| bbox | 실제 잘라낸 정수 `(left, top, right, bottom)`, 우하단 제외 |
| size | `(width, height)`, 반환 이미지의 현재 크기 |

좌표 규칙은 **좌상단 floor(내림), 우하단 ceil(올림)** 후 각각 `[0, width]`,
`[0, height]`로 제한하는 것입니다. 반올림 전에 연속 좌표의 교집합을 확인해
이미지 밖 bbox나 경계에 닿기만 하는 bbox가 픽셀로 부풀려지지 않도록 합니다.
예를 들어 `(1.9, 2.1, 6.2, 5.9)`는 `(1, 2, 7, 6)`, 크기 `(6, 4)`가 됩니다.
유효한 작은 소수점 영역은 최소 한 픽셀을 포함합니다. 원본과 이벤트는 변경하지
않으며 반환 이미지의 픽셀 수정도 원본에 영향을 주지 않습니다.

잘못된 이미지 또는 이벤트 타입은 TypeError, 빈 이미지·닫힌 이미지·양의 면적을
갖는 교집합이 없는 bbox는 ValueError입니다. 0폭·0높이 또는 역전된 bbox는
DetectionEvent 생성 단계에서 이미 거부됩니다. 결과 객체의 필드 대입은 막지만
내부 Pillow 이미지 자체는 수정 가능합니다.

아래 예제는 디스크 영상 없이 합성 이미지로 실행됩니다.

```python
from PIL import Image
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.roi_extractor import extract_roi

event = DetectionEvent(
    event_id="synthetic-roi-01",
    timestamp="2026-01-01T00:00:00Z",
    camera_id="synthetic-camera",
    class_label="synthetic-object",
    confidence=0.8,
    bbox=(-2.5, 1.2, 6.1, 7.8),
)
with Image.new("RGB", (10, 8), (20, 40, 60)) as image:
    result = extract_roi(image, event)
    print(result.bbox, result.size)  # (0, 1, 7, 8) (7, 7)
    result.image.close()
```

Pillow는 ROI 함수 호출 시에만 import됩니다. Pillow 없이도 패키지·이벤트·ROI
모듈의 import는 가능하지만 ROI 실행과 전체 테스트에는 설치가 필요합니다.
의존성 범위는 `Pillow>=10.4,<14`이며 pip가 Python 요구사항에 맞는 버전을 선택합니다.
[Pillow 공식 Python 지원표](https://pillow.readthedocs.io/en/stable/installation/python-support.html)에
따르면 Python 3.8은 Pillow 10.4까지, Python 3.9는 11 계열까지 지원합니다.
이 범위는 모든 Python/Pillow 조합을 실제 실행 검증했다는 의미가 아닙니다.
테스트 러너는 표준 라이브러리 unittest이며 추가 테스트 의존성은 없습니다.

ROI 추출은 후속 품질 점검과 VLM 검증에 전달할 영상 영역을 준비하는 단계입니다.
실제 VLM 호출, MockVLM, 드리프트 후보 판단 및 자동 재학습은 수행하지 않습니다.

## 이미지 품질 점검 (PR-04)

`analyze_image_quality(image, config=None) -> ImageQualityResult`는 ROI의
밝기와 경계 정보를 측정합니다. 입력은 메모리에 로드된 Pillow L/RGB 이미지입니다.
RGB는 별도의 L 이미지로 변환해 분석하며 원본 픽셀·모드·메타데이터는 변경하지
않습니다. RGBA·팔레트·고비트 영상은 ValueError로 거부하므로 호출자가 투명도와
명암 범위의 의미를 결정한 후 변환해야 합니다. 잘못된 입력 타입은 TypeError,
빈 이미지나 닫힌 이미지는 ValueError입니다. 파일 검색·저장·네트워크 호출은 없습니다.

결과와 설정은 frozen dataclass이고 `issues`는 tuple입니다.

| 결과 필드 | 계산 및 의미 |
|---|---|
| mean_brightness | 8-bit L 밝기 평균, 0~255 |
| dark_pixel_ratio | 밝기 <= dark_pixel_threshold인 픽셀 수 / 전체 픽셀 수 |
| bright_pixel_ratio | 밝기 >= bright_pixel_threshold인 픽셀 수 / 전체 픽셀 수 |
| luminance_stddev | 전체 픽셀 밝기의 모집단 표준편차 |
| edge_score | 수평·수직 인접 픽셀의 절대 밝기 차이 합 / (255 × 인접 쌍 수), 0~1 |
| issues | 노출·흐림 후보 또는 판단을 제한하는 사유 목록 |
| assessable | 최소 크기와 대비 조건을 통과하여 흐림 휴리스틱을 적용할 수 있는지 여부 |

edge_score는 높을수록 경계 변화가 큽니다. **흐림 확률이 아니며**, 1×1 이미지처럼
인접 쌍이 없으면 None입니다. 외곽 패딩·리사이즈 없이 모든 인접 쌍을 한 번씩
계산합니다. 픽셀 수에 비례하는 시간·메모리를 사용하므로 작은 ROI용 예제입니다.

`ImageQualityConfig`의 시연용 기본값과 판정 규칙:

| 설정 | 기본값 | 규칙 |
|---|---:|---|
| dark_pixel_threshold | 32 | 정수 0~255 |
| bright_pixel_threshold | 223 | 정수 0~255, dark보다 커야 함 |
| exposure_ratio_threshold | 0.8 | 0 초과 1 이하. 해당 픽셀 비율 >= 기준이면 노출 후보 |
| min_dimension | 8 | 정수 2 이상. 너비 또는 높이가 기준 미만이면 insufficient_size |
| min_luminance_stddev | 2.0 | 0~127.5. 표준편차 <= 기준이면 low_texture |
| blur_score_threshold | 0.03 | 0~1. assessable이고 edge_score < 기준이면 blur_candidate |

노출 후보 이름은 `underexposure_candidate`, `overexposure_candidate`입니다.
작거나 균일한 영상에서도 노출 지표는 제공하지만 `assessable=False`로 흐림 판정을
유보합니다. `low_texture`는 초점 흐림의 증명이 아닙니다. 사용자가 노출 비율을
낮추면 양쪽 노출 후보가 동시에 표시될 수 있습니다. 문제 목록 순서는 과소 노출,
과다 노출, 크기 부족, 저대비, 흐림 후보 순서로 고정됩니다.

모든 수치 설정은 bool·NaN·Infinity를 거부합니다. 다음처럼 설정 객체를 명시적으로
전달합니다. config.example.json 및 루트 설정은 이 함수에서 읽지 않습니다.

```python
from PIL import Image
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.roi_extractor import extract_roi
from examples.vlm_semantic_drift.image_quality import ImageQualityConfig, analyze_image_quality

event = DetectionEvent(
    event_id="synthetic-quality-01", timestamp="2026-01-01T00:00:00Z",
    camera_id="synthetic-camera", class_label="synthetic-object",
    confidence=0.8, bbox=(0, 0, 16, 16),
)
config = ImageQualityConfig(exposure_ratio_threshold=0.9)
with Image.new("RGB", (16, 16), (128, 128, 128)) as image:
    roi = extract_roi(image, event)
    try:
        result = analyze_image_quality(roi.image, config)
        print(result.mean_brightness, result.issues, result.assessable)
        # 128.0 ('low_texture',) False — 균일 영상이라 흐림 판단 유보
    finally:
        roi.image.close()
```

이 임계값은 **현장 검증을 거치지 않은 교육용 기본값**입니다. 부드러운 그라데이션도
blur_candidate가 될 수 있고, 노이즈는 경계 점수를 높일 수 있습니다. 실제 초점,
모션 블러, 역광·눈부심의 원인을 구분하지 않습니다. 이미지 해상도·콘텐츠에 따라
점수가 달라지며 정확도를 보장하지 않습니다. `assessable=True`나 빈 issues도
이미지 품질 또는 의미적 라벨 일치의 보장이 아닙니다.

품질 이상은 데이터 드리프트가 아닙니다. 이 모듈은 품질 문제를 별도로 기록할
근거만 제공하며 H0/H1, label_match 또는 drift_candidate를 생성하지 않습니다.
실제 VLM·MockVLM 추론, 시간 누적 및 드리프트 판단은 후속 PR 범위입니다.
Pillow는 함수 호출 시에만 import하며 신규 외부 의존성은 없습니다.

## H0/H1 가설 비교 프롬프트 (PR-05)

`build_hypothesis_prompts(event, quality=None, config=None) -> PromptResult`는
**텍스트 생성과 입력 검증만** 수행합니다. 실제 이미지가 함수에 전달되거나 VLM에
전송되는 것은 아닙니다. 후속 어댑터가 해당 이벤트에 맞는 ROI를 별도로 제공해야
하며, 이 함수는 이미지 존재 여부나 라벨의 시각적 타당성을 검사하지 않습니다.

- H0: ROI의 시각적 증거가 검출 라벨과 일치합니다.
- H1: ROI의 시각적 증거가 검출 라벨과 일치하지 않습니다.
- UNKNOWN: 가림·품질 한계·정보 부족 때문에 H0/H1 중 하나를 선택할 수 없습니다.

검출 라벨은 정답이 아닌 검증 대상입니다. 지시문에 물체의 외형·행동을 추가하지
않으며, NO를 선택하려면 불일치를 판단할 시각적 근거가 있어야 합니다. 증거 부족을
NO로 취급하지 않습니다. 이번 PR에서 H1과 UNKNOWN을 명시적으로 분리했습니다.

`PromptConfig`와 `PromptResult`는 frozen dataclass입니다.

| 설정 | 기본값 | 규칙 |
|---|---|---|
| max_label_length | 128 | 정수 1~256, 라벨 문자 수 상한 |
| include_quality | true | bool, false이면 제공된 품질 정보도 프롬프트에서 생략 |

`PromptResult`는 `h0`, `h1`, `question`, `response_contract`, `data_json`,
전체 결합 문자열 `prompt`를 제공합니다. 생성 언어는 영어로 고정합니다.
H0/H1과 질문은 고정된 `data.class_label` 참조를 사용하고, 실제 라벨은 JSON의
class_label 값에만 기록합니다. 이벤트 ID·카메라 ID·시각·신뢰도·경로는 프롬프트에
포함하지 않습니다. 원본 이벤트는 기존 DetectionEvent 계약으로 재검증합니다.

외부 문자열은 JSON 이스케이프하고 `<`, `>`, `&`도 이스케이프하여 데이터 구분을
유지합니다. 라벨의 Unicode 제어·서식 문자와 과도한 길이는 거부합니다. 임의의
라벨을 지시문으로 실행하지 않도록 명시하지만, **텍스트 구분만으로 실제 모델의
프롬프트 인젝션 저항성을 보장하지 않습니다.** 실제 VLM 보안·정확도 평가는 미수행입니다.

품질 결과는 타입, 지표 범위, 알려진 issues 이름, assessable의 bool 여부를
검증합니다. 지표에서 issues를 재계산하거나 품질 임계값을 바꾸지 않으며, 두 입력이
동일한 ROI에서 유래했는지는 호출자가 보장해야 합니다. quality=None은 품질 정상의
뜻이 아닙니다. include_quality=False여도 전달된 품질 객체의 유효성은 검사합니다.
타입 오류는 TypeError, 값·길이·지표 범위 오류는 ValueError입니다.

issues는 시각적 검증의 한계를 알리는 참고 정보이며 답변을 자동 결정하지 않습니다.
`assessable=False`는 흐림 휴리스틱을 적용하지 못했다는 뜻일 뿐, VLM의 모든 시각적
판단이 불가능하다는 뜻은 아닙니다. 어떤 품질 후보도 의미적 불일치나 드리프트로
변환하지 않습니다.

기본 프롬프트 생성은 Pillow 없이 표준 라이브러리만으로 실행할 수 있습니다.

```python
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts

event = DetectionEvent(
    event_id="synthetic-prompt", timestamp="2026-01-01T00:00:00Z",
    camera_id="synthetic-camera", class_label="person",
    confidence=0.8, bbox=(0, 0, 16, 16),
)
result = build_hypothesis_prompts(event)
print(result.h0)
print(result.h1)
print(result.question)
print(result.data_json)
```

출력 예시:

```text
H0: Visual evidence in the ROI agrees with the detector label in data.class_label.
H1: Visual evidence in the ROI does not agree with the detector label in data.class_label.
Does the visible evidence in the provided ROI agree with data.class_label? Compare H0 and H1 neutrally using only visible evidence.
{"class_label": "person", "quality": null}
```

품질 정보를 연결할 때는 앞의 event를 그대로 사용합니다.

```python
from PIL import Image
from examples.vlm_semantic_drift.roi_extractor import extract_roi
from examples.vlm_semantic_drift.image_quality import analyze_image_quality

with Image.new("RGB", (16, 16), (128, 128, 128)) as image:
    roi = extract_roi(image, event)
    try:
        quality = analyze_image_quality(roi.image)
        result = build_hypothesis_prompts(event, quality=quality)
        print(result.prompt)
    finally:
        roi.image.close()
```

전체 prompt에는 YES/NO/UNKNOWN 중 한 토큰만 반환하라는 규약과 아래 극성 정의가
포함됩니다. PR-05는 응답 파싱, 실제 VLM/MockVLM 호출, 시간 누적, 드리프트 판단을
구현하지 않습니다. 반환된 텍스트는 추론 결과가 아닙니다.

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

## 가설과 응답의 의미: 프롬프트 규약과 후속 정규화 계약

계획된 흐름은 이벤트 ROI와 검출 정보 → 품질 점검 → H0/H1 질문 → VLM 어댑터
→ 응답 정규화 → 시간 누적 → 후보 리포트 → 선택적 사용자 피드백입니다.

- H0: 검출기가 붙인 이벤트 라벨이 사용 가능한 시각 증거와 일치합니다.
- H1: 라벨이 시각 증거와 불일치합니다. 불확실성은 별도 UNKNOWN 상태입니다.

질문은 항상 **“이벤트 라벨이 시각 증거와 일치하는가?”**로 해석합니다.
다음 매핑은 프롬프트의 응답 규약에 포함됩니다. 실제 응답 파싱과 정규화 결과
생성은 후속 PR 범위이며, 지금 반환되는 것은 프롬프트 텍스트뿐입니다.

| 응답 | label_match | supports_h1 | 누적 처리 |
|---|---|---|---|
| YES | true | false | 유효한 일치 표본 |
| NO | false | true | 유효한 불일치 표본 |
| UNKNOWN | null | null | 판단 유보, 불일치 비율의 분자·분모에서 제외 |

UNKNOWN은 H0/H1 선택을 유보하는 상태이며 H1 지지 또는 불일치 증거가 아닙니다.
supports_h1의 null은 지지 여부를 결정할 수 없다는 뜻이며 false와 다릅니다.
Python에서는 JSON null을 None으로 표현합니다. 응답에는 향후 provider와
is_mock=true를 포함하여 모의 결과임을 구분할 예정입니다.

블러, 저노출, 눈부심·역광 등의 품질 문제는 별도 quality_issue로 표시하고
의미적 불일치로 집계하지 않습니다. PR-04는 노출·경계 정보의 간단한 휴리스틱을
제공하며, 역광·눈부심의 원인 판별과 전체 파이프라인의 품질 게이트 연결은 미구현입니다.
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
