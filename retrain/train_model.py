#!/usr/bin/env python3
"""
YOLO 모델 재학습 스크립트
- data/good, data/wrong_corrected 폴더 내 이미지를 기반으로 YOLO 모델 재학습
- yolo_producer_fastapi.py의 config.yaml과 동일한 설정을 사용
"""

import os
import sys
from glob import glob
from datetime import datetime
from ultralytics import YOLO
import shutil
import yaml
from pathlib import Path
# base_abspath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(base_abspath)

# YOLO80 클래스명 가져오기
from vision_analysis.class_names_yolo80n import TEXT_LABELS_80
class_names_list = None

def extract_class_names_from_labels(labels_dirs):
    """
    레이블 파일에서 사용된 클래스 ID를 추출하고 YOLO80 클래스명으로 매핑

    Args:
        labels_dirs: 레이블 디렉터리 리스트

    Returns:
        dict: {클래스 ID: 클래스명} 형태의 딕셔너리
    """
    # YOLO80 클래스명을 리스트로 변환 (인덱스 = 클래스 ID)
    yolo80_names = list(TEXT_LABELS_80.values())

    # 모든 레이블 파일에서 클래스 ID 수집
    class_ids = set()
    for labels_dir in labels_dirs:
        if not os.path.exists(labels_dir):
            continue

        label_files = glob(os.path.join(labels_dir, "*.txt"))
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_ids.add(class_id)
            except Exception as e:
                print(f"Warning: Failed to read {label_file}: {e}")
                continue

    # 클래스 ID를 정렬하고 매핑
    sorted_class_ids = sorted(class_ids)
    class_mapping = {}

    for class_id in sorted_class_ids:
        if class_id < len(yolo80_names):
            class_mapping[class_id] = yolo80_names[class_id]
        else:
            class_mapping[class_id] = f"class_{class_id}"

    return class_mapping, sorted_class_ids


def merge_and_split_datasets(source_dirs, output_dir="datasets/splitted",
                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             max_samples=None, random_seed=42):
    """
    여러 데이터셋을 머지하고 train/val/test로 분할

    Args:
        source_dirs: 소스 디렉토리 리스트
                    예: ['datasets/collected/good_images', 'datasets/collected/wronged_images']
        output_dir: 출력 디렉토리 (기본값: 'datasets/splitted')
        train_ratio: 학습 데이터 비율 (기본값: 0.7)
        val_ratio: 검증 데이터 비율 (기본값: 0.15)
        test_ratio: 테스트 데이터 비율 (기본값: 0.15)
        max_samples: 최대 샘플 수 제한 (None이면 전체 사용)
        random_seed: 랜덤 시드 (기본값: 42)

    Returns:
        dict: 분할된 데이터셋 정보
    """
    import random

    print(f"\n[{datetime.now()}] Starting dataset merge and split...")
    print(f"Split ratio - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    if max_samples:
        print(f"Max samples limit: {max_samples}")

    # 출력 디렉토리 구조 생성
    output_path = Path(base_abspath) / output_dir
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 모든 소스 디렉토리에서 이미지-레이블 쌍 수집
    all_pairs = []
    for src_dir in source_dirs:
        src_path = Path(base_abspath) / src_dir
        images_dir = src_path / 'images'
        labels_dir = src_path / 'labels'

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue

        # 이미지 파일 찾기 (jpg, png 지원)
        for img_file in list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_pairs.append({'image': img_file, 'label': label_file})
            else:
                print(f"Warning: Label not found for {img_file.name}, skipping...")

        print(f"  {src_dir}: {len(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))} images")

    total_files = len(all_pairs)
    if total_files == 0:
        print("Error: No valid image-label pairs found!")
        return None

    print(f"\nTotal collected: {total_files} image-label pairs")

    # 최대 샘플 수 제한 적용
    if max_samples and total_files > max_samples:
        random.seed(random_seed)
        all_pairs = random.sample(all_pairs, max_samples)
        total_files = max_samples
        print(f"Limited to {max_samples} samples")

    # 데이터 셔플
    random.seed(random_seed)
    random.shuffle(all_pairs)

    # 분할 인덱스 계산
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }

    # 파일 복사
    stats = {}
    for split_name, pairs in splits.items():
        print(f"\nCopying {len(pairs)} files to {split_name}...")

        for pair in pairs:
            # 이미지 복사
            dest_img = output_path / split_name / 'images' / pair['image'].name
            shutil.copy2(pair['image'], dest_img)

            # 레이블 복사
            dest_label = output_path / split_name / 'labels' / pair['label'].name
            shutil.copy2(pair['label'], dest_label)

        stats[split_name] = len(pairs)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Dataset merge and split completed!")
    print(f"  Train: {stats['train']} ({stats['train']/total_files*100:.1f}%)")
    print(f"  Val:   {stats['val']} ({stats['val']/total_files*100:.1f}%)")
    print(f"  Test:  {stats['test']} ({stats['test']/total_files*100:.1f}%)")
    print(f"  Total: {total_files}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")

    return {
        'output_dir': str(output_path),
        'stats': stats,
        'total': total_files
    }


def train_model():
    """
    YOLO 모델 재학습 함수
    - datasets/splitted/train, val, test 폴더의 이미지와 레이블을 사용하여 재학습
    - 학습된 모델은 runs/retrain/ 하위 폴더에 저장
    """

    print(f"[{datetime.now()}] Starting merge_and_split_datasets...")
    # 예시 1: 기본 사용 (70/15/15 비율)
    merge_and_split_datasets([
        'datasets/collected/good_images',
        'datasets/collected/wronged_images'
    ])

    # # 예시 2: 커스텀 비율 + 샘플 수 제한
    # merge_and_split_datasets(
    #     source_dirs=[
    #         'datasets/collected/good_images',
    #         'datasets/collected/wronged_images',
    #         'datasets/collected/other_images'
    #     ],
    #     train_ratio=0.8,
    #     val_ratio=0.1,
    #     test_ratio=0.1,
    #     max_samples=1000
    # )
    print(f"[{datetime.now()}] Starting YOLO model retraining...")

    # 설정 파일 로드
    config_path = os.path.join(base_abspath, "config.yaml")
    if not os.path.exists(config_path):
        print(f"config.yaml not found at {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # YOLO 모델 설정
    YOLO_MODEL = config["yolo_model"]["model_name"]
    CONF_THRESH = float(config["yolo_model"]["conf_thresh"])

    # 분할된 데이터셋 경로
    dataset_dir = os.path.join(base_abspath, "datasets", "splitted")

    # 데이터셋 존재 여부 확인
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_images_dir = os.path.join(dataset_dir, "val", "images")
    val_labels_dir = os.path.join(dataset_dir, "val", "labels")
    test_images_dir = os.path.join(dataset_dir, "test", "images")

    # 필수 디렉터리 확인
    for dir_path, dir_name in [
        (train_images_dir, "train/images"),
        (train_labels_dir, "train/labels"),
        (val_images_dir, "val/images"),
        (val_labels_dir, "val/labels")
    ]:
        if not os.path.exists(dir_path):
            print(f"Error: Required directory not found: {dir_name}")
            print(f"Expected path: {dir_path}")
            return

    # 각 세트의 이미지 수 확인
    train_images = glob(os.path.join(train_images_dir, "*.jpg"))
    val_images = glob(os.path.join(val_images_dir, "*.jpg"))
    test_images = glob(os.path.join(test_images_dir, "*.jpg"))

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} images")

    if len(train_images) == 0:
        print("Error: No training images found!")
        return
    if len(val_images) == 0:
        print("Error: No validation images found!")
        return

    # 레이블 파일에서 클래스 정보 추출
    labels_dirs = [train_labels_dir, val_labels_dir]
    if os.path.exists(os.path.join(dataset_dir, "test", "labels")):
        labels_dirs.append(os.path.join(dataset_dir, "test", "labels"))

    class_mapping, class_ids = extract_class_names_from_labels(labels_dirs)

    if not class_mapping:
        print("Error: No valid class labels found in dataset!")
        return

    # 클래스 정보 출력
    print(f"\nDetected {len(class_mapping)} classes:")
    for class_id, class_name in class_mapping.items():
        print(f"  ID {class_id}: {class_name}")

    # YOLO는 연속된 클래스 ID (0, 1, 2, ...)를 요구하므로 재매핑
    global class_names_list
    class_names_list = [class_mapping[class_id] for class_id in class_ids]

    # data.yaml 생성
    data_yaml = {
        "path": dataset_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names_list),
        "names": class_names_list
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    print(f"\ndata.yaml created at: {yaml_path}")
    print(f"Number of classes (nc): {len(class_names_list)}")

    # YOLO 모델 로드 및 재학습
    print(f"\n[{datetime.now()}] Loading model: {YOLO_MODEL}")
    # model = YOLO(YOLO_MODEL)

    # print(f"[{datetime.now()}] Training started...")
    # results = model.train(
    #     data=yaml_path,
    #     epochs=20,
    #     imgsz=640,
    #     batch=16,
    #     name="retrain",
    #     project=os.path.join(base_abspath, "runs"),
    #     device=0
    # )
    model = YOLO(YOLO_MODEL)

    print(f"[{datetime.now()}] Training started...")

    runs_dir = os.path.join(base_abspath, "runs")

    # # Check if runs folder exists and delete it
    # if os.path.exists(runs_dir):
    #     print(f"[{datetime.now()}] Deleting existing runs folder: {runs_dir}")
    #     shutil.rmtree(runs_dir)  # Recursively delete directory and all contents    
    # Adjust learning rate based on drift detection
    # Lower learning rate for fine-tuning to preserve learned features
    # while adapting to drift
    drift_detected = True
    if drift_detected:
        # Fine-tuning mode: use lower learning rate
        lr0 = 0.001  # Initial learning rate (reduced from default ~0.01)
        lrf = 0.01   # Final learning rate (as fraction of lr0)
        print(f"[{datetime.now()}] Drift detected - Using fine-tuning learning rate: {lr0}")
    else:
        # Full retraining mode: use default/higher learning rate
        lr0 = 0.01   # Standard initial learning rate
        lrf = 0.1    # Final learning rate
        print(f"[{datetime.now()}] Full retraining - Using standard learning rate: {lr0}")

    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        name="retrain",
        project=os.path.join(base_abspath, "runs"),
        device=0,
        lr0=lr0,      # Initial learning rate
        lrf=lrf,      # Final learning rate (lr0 * lrf)
        optimizer='AdamW',  # Consider AdamW for fine-tuning
        patience=10,  # Early stopping patience
        warmup_epochs=3  # Warmup epochs for stable training
    )
    print(f"\n[{datetime.now()}] Training completed!")
    print(f"Results saved to: {results.save_dir}")


    # After extracting class names, verify:
    print(f"\nClass mapping verification:")
    print(f"Old model classes: size={len(model.names)}, {model.names}")  # Your previous model's classes
    print(f"New dataset classes: size={len(model.names)}, {class_names_list}")  # Current dataset classes
    pass


def evaluate_model(model_path, test_data_yaml=None, test_images_dir=None):
    """
    학습된 모델을 테스트 세트로 평가

    Args:
        model_path: 학습된 모델 경로 (.pt 파일)
        test_data_yaml: data.yaml 파일 경로 (test 경로 포함)
        test_images_dir: 테스트 이미지 디렉토리 (yaml 없이 직접 지정 시)

    Returns:
        dict: 평가 결과
    """
    print(f"\n[{datetime.now()}] Starting model evaluation...")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    # YOLO 모델 로드
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 방법 1: data.yaml 사용 (test 경로 포함)
    if test_data_yaml:
        if not os.path.exists(test_data_yaml):
            print(f"Error: data.yaml not found at {test_data_yaml}")
            return None

        print(f"Evaluating with data.yaml: {test_data_yaml}")
        # val() 메서드는 data.yaml의 'test' 경로를 사용
        results = model.val(
            data=test_data_yaml,
            split='test',  # 'test' split 사용
            save_json=True,
            save_hybrid=True
        )

    # 방법 2: 테스트 이미지 디렉토리 직접 지정
    elif test_images_dir:
        if not os.path.exists(test_images_dir):
            print(f"Error: Test images directory not found at {test_images_dir}")
            return None

        print(f"Evaluating with test images: {test_images_dir}")

        # 임시 data.yaml 생성 (테스트용)
        import tempfile
        temp_yaml_path = os.path.join(tempfile.gettempdir(), "temp_eval_data.yaml")

        # 테스트 이미지 디렉토리의 부모 디렉토리 찾기
        test_parent_dir = os.path.dirname(os.path.dirname(test_images_dir))

        # 라벨 디렉토리 추론
        test_labels_dir = os.path.join(os.path.dirname(test_images_dir), "labels")

        # 클래스 정보 추출
        # class_names_list = list(TEXT_LABELS_80.values())
        # if os.path.exists(test_labels_dir):
        #     class_mapping, class_ids = extract_class_names_from_labels([test_labels_dir])
        #     class_names_list = [class_mapping[cid] for cid in class_ids] if class_mapping else ["person"]
        # else:
        #     # 기본 클래스 사용
        #     class_names_list = ["person"]

        # YOLO는 train과 val이 필수이므로 test를 train/val로도 사용
        test_rel_path = os.path.relpath(test_images_dir, test_parent_dir)
        temp_data_yaml = {
            "path": test_parent_dir,
            "train": test_rel_path,  # 필수: test와 동일하게 설정
            "val": test_rel_path,    # 필수: test와 동일하게 설정
            "test": test_rel_path,   # 실제 평가할 경로
            "nc": len(class_names_list),
            "names": class_names_list
        }

        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(temp_data_yaml, f, allow_unicode=True)

        print(f"Created temporary data.yaml at: {temp_yaml_path}")

        # 테스트 이미지에 대해 예측 수행
        results = model.val(
            data=temp_yaml_path,
            split='test',
            save_json=True,
            save_hybrid=True
        )

    else:
        print("Error: Either test_data_yaml or test_images_dir must be provided")
        return None

    # 결과 출력
    print(f"\n{'='*60}")
    print("Model Evaluation Results:")
    print(f"{'='*60}")

    # 주요 메트릭 출력
    if hasattr(results, 'box'):
        metrics = results.box
        print(f"mAP50: {metrics.map50:.4f}")
        print(f"mAP50-95: {metrics.map:.4f}")
        print(f"Precision: {metrics.mp:.4f}")
        print(f"Recall: {metrics.mr:.4f}")

        # 클래스별 성능
        if hasattr(metrics, 'ap_class_index'):
            print(f"\nPer-class mAP50:")
            for i, (cls_idx, ap) in enumerate(zip(metrics.ap_class_index, metrics.ap50)):
                print(f"  Class {cls_idx}: {ap:.4f}")

    print(f"{'='*60}\n")

    return results


# def evaluate_trained_model():
#     """
#     학습 완료 후 자동으로 테스트 세트로 평가하는 함수
#     """
#     # 가장 최근 학습된 모델 찾기
#     runs_base_dir = os.path.join(base_abspath, "runs")

#     if not os.path.exists(runs_base_dir):
#         print(f"Error: No runs directory found at {runs_base_dir}")
#         return

#     # retrain, retrain2, retrain3 등 폴더 찾기
#     subdirs = [d for d in os.listdir(runs_base_dir)
#                if os.path.isdir(os.path.join(runs_base_dir, d)) and d.startswith('retrain')]

#     if not subdirs:
#         print("Error: No retrain folders found in runs directory")
#         return

#     # 수정 시간 기준으로 가장 최근 폴더 찾기
#     latest_dir = max([os.path.join(runs_base_dir, d) for d in subdirs], key=os.path.getmtime)

#     print(f"Latest training folder: {latest_dir}")

#     # best.pt 파일 찾기 (weights 폴더 안 또는 직접)
#     possible_paths = [
#         os.path.join(latest_dir, "weights", "best.pt"),
#         os.path.join(latest_dir, "best.pt")
#     ]

#     model_path = None
#     for path in possible_paths:
#         if os.path.exists(path):
#             model_path = path
#             break

#     if not model_path:
#         print(f"Error: best.pt not found in {latest_dir}")
#         print(f"Checked paths: {possible_paths}")
#         return

#     # data.yaml 경로
#     dataset_dir = os.path.join(base_abspath, "datasets", "splitted")
#     data_yaml = os.path.join(dataset_dir, "data.yaml")

#     # 평가 실행
#     print(f"Evaluating model: {model_path}")
#     results = evaluate_model(
#         model_path=model_path,
#         test_data_yaml=data_yaml
#     )

#     return results


def evaluate_old_and_new_models():
    """
    구 모델과 신규 학습된 모델을 테스트 세트로 평가하여 비교

    Returns:
        dict: {'old_model': results, 'new_model': results}
    """
    test_images_dir = os.path.join(base_abspath, "datasets", "splitted", "test", "images")

    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        return None

    results = {}

    # Step 1: 구 모델 평가
    print(f"\n{'='*60}")
    print("Step 1: Evaluating OLD model on test set")
    print(f"{'='*60}")

    # config.yaml에서 구 모델 경로 가져오기
    config_path = os.path.join(base_abspath, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        old_model_name = config.get("yolo_model", {}).get("model_name", "./model/yolov8n_local.pt")
    else:
        old_model_name = "./model/yolov8n_local.pt"

    old_model_path = os.path.join(base_abspath, old_model_name)

    if os.path.exists(old_model_path):
        print(f"Old model: {old_model_path}")
        old_results = evaluate_model(
            model_path=old_model_path,
            test_images_dir=test_images_dir
        )
        results['old_model'] = old_results
    else:
        print(f"Warning: Old model not found at {old_model_path}")
        results['old_model'] = None

    # Step 2: 신규 모델 평가
    print(f"\n{'='*60}")
    print("Step 2: Evaluating NEW model on test set")
    print(f"{'='*60}")

    # 가장 최근 학습된 모델 찾기
    runs_base_dir = os.path.join(base_abspath, "runs")

    if os.path.exists(runs_base_dir):
        # retrain, retrain2, retrain3 등 폴더 찾기
        subdirs = [d for d in os.listdir(runs_base_dir)
                   if os.path.isdir(os.path.join(runs_base_dir, d)) and d.startswith('retrain')]

        if subdirs:
            # 수정 시간 기준으로 가장 최근 폴더 찾기
            latest_dir = max([os.path.join(runs_base_dir, d) for d in subdirs], key=os.path.getmtime)

            print(f"Latest training folder: {latest_dir}")

            # best.pt 파일 찾기 (weights 폴더 안 또는 직접)
            possible_paths = [
                os.path.join(latest_dir, "weights", "best.pt"),
                os.path.join(latest_dir, "best.pt")
            ]

            new_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    new_model_path = path
                    break

            if new_model_path:
                print(f"New model: {new_model_path}")
                new_results = evaluate_model(
                    model_path=new_model_path,
                    test_images_dir=test_images_dir
                )
                results['new_model'] = new_results
            else:
                print(f"Warning: best.pt not found in {latest_dir}")
                print(f"Checked paths: {possible_paths}")
                results['new_model'] = None
        else:
            print("Warning: No retrain folders found in runs directory")
            results['new_model'] = None
    else:
        print(f"Warning: Training directory not found at {runs_base_dir}")
        results['new_model'] = None

    # Step 3: 결과 비교
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")

    if results.get('old_model') and results.get('new_model'):
        old_metrics = results['old_model'].box if hasattr(results['old_model'], 'box') else None
        new_metrics = results['new_model'].box if hasattr(results['new_model'], 'box') else None

        if old_metrics and new_metrics:
            print(f"\nMetric Comparison:")
            print(f"{'Metric':<15} {'Old Model':<15} {'New Model':<15} {'Improvement':<15}")
            print(f"{'-'*60}")

            metrics_to_compare = [
                ('mAP50', 'map50'),
                ('mAP50-95', 'map'),
                ('Precision', 'mp'),
                ('Recall', 'mr')
            ]

            for metric_name, metric_attr in metrics_to_compare:
                old_val = getattr(old_metrics, metric_attr, 0)
                new_val = getattr(new_metrics, metric_attr, 0)
                improvement = new_val - old_val
                improvement_pct = (improvement / old_val * 100) if old_val > 0 else 0

                print(f"{metric_name:<15} {old_val:<15.4f} {new_val:<15.4f} {improvement_pct:+.2f}%")

    print(f"{'='*60}\n")

    return results


# -------------------- main --------------------
def main():
    """
    메인 엔트리 포인트
    학습 후 구 모델과 신규 모델을 테스트 세트로 평가
    """
    try:
        print(f"\n=== YOLO Model Retraining Process Started ===")

        # Step 1: 모델 학습
        train_model()

        # Step 2: 구 모델과 신규 모델 평가 및 비교
        print(f"\n=== Evaluating Old and New Models ===")
        evaluate_old_and_new_models()

        print(f"\n=== YOLO Model Retraining Finished ===\n")


    except Exception as e:
        import traceback
        print("[ERROR] Model retraining failed!")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
