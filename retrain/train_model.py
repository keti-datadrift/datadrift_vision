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
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import shutil
import yaml
base_abspath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_abspath)


def train_model():
    """
    YOLO 모델 재학습 함수
    - data/good, data/wrong_corrected 폴더 내 이미지를 로드하여 재학습
    - 학습된 모델은 runs/retrain/ 하위 폴더에 저장
    """

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

    # 학습용 데이터 디렉터리
    good_images = glob(f"{base_abspath}/data/good/*.jpg")
    wrong_images = glob(f"{base_abspath}/data/wrong_corrected/*.jpg")
    print(f"Loaded {len(good_images)} good samples")
    print(f"Loaded {len(wrong_images)} wrong samples")

    total = len(good_images) + len(wrong_images)
    if total == 0:
        print("No training data found!")
        return

    # 라벨 구성
    X = good_images + wrong_images
    y = [1] * len(good_images) + [0] * len(wrong_images)

    # 데이터 분할

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # YOLO 학습용 폴더 구조 생성
    dataset_dir = os.path.join(base_abspath, "datasets", "retrain_dataset")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)

    # good→1, wrong→0 YOLO 라벨 생성
    def write_label(image_path, label_value, subset):
        label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(dataset_dir, f"labels/{subset}", label_name)
        with open(label_path, "w") as f:
            # YOLO 형식: class_id x_center y_center width height (dummy)
            f.write(f"{label_value} 0.5 0.5 1.0 1.0\n")

        shutil.copy(image_path, os.path.join(dataset_dir, f"images/{subset}", os.path.basename(image_path)))

    for img, lbl in zip(X_train, y_train):
        write_label(img, lbl, "train")
    for img, lbl in zip(X_val, y_val):
        write_label(img, lbl, "val")

    # data.yaml 생성
    data_yaml = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["wrong", "good"]
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # YOLO 모델 로드 및 재학습
    print(f"\n[{datetime.now()}] Loading model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    print(f"[{datetime.now()}] Training started...")
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        name="retrain",
        project=os.path.join(base_abspath, "runs"),
        device=0
    )

    print(f"\n[{datetime.now()}] Training completed!")
    print(f"Results saved to: {results.save_dir}")


# -------------------- main --------------------
def main():
    """
    메인 엔트리 포인트
    """
    try:
        print(f"\n=== YOLO Model Retraining Process Started ===")
        train_model()
        print(f"=== YOLO Model Retraining Finished ===\n")
    except Exception as e:
        import traceback
        print("[ERROR] Model retraining failed!")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
