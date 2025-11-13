import os
import json
from collections import defaultdict

# COCO category_id → 0~79 YOLO class index 매핑
COCO_ID_TO_INDEX = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10,
    13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18,
    21:19, 22:20, 23:21, 24:22, 25:23, 27:24, 28:25, 31:26,
    32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34,
    40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 47:41, 48:42,
    49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50,
    57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58,
    65:59, 67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66,
    77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74,
    86:75, 87:76, 88:77, 89:78, 90:79
}

def convert_coco_to_yolo(
    img_dir="../../datasets/coco/images/val2017",
    label_path="../../datasets/coco/labels/val2017",
    save_label_dir="../../datasets/coco/labels_yolo/val2017"
):
    os.makedirs(save_label_dir, exist_ok=True)

    print("COCO annotation JSON 로딩:", json_path)
    with open(json_path, "r") as f:
        coco = json.load(f)

    # 이미지 정보 매핑
    image_info = {img["id"]: img for img in coco["images"]}

    # 이미지 ID별 라벨 목록 구성
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    print("총 이미지 수:", len(image_info))

    # 이미지별 YOLO 라벨 생성
    for image_id, anns in ann_by_image.items():
        img_file = image_info[image_id]["file_name"]
        img_path = os.path.join(img_dir, img_file)

        # 이미지 크기 로드
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print("이미지를 읽을 수 없음:", img_path)
            continue

        img_h, img_w = img.shape[:2]

        # YOLO 저장 파일 경로
        txt_name = os.path.splitext(img_file)[0] + ".txt"
        save_path = os.path.join(save_label_dir, txt_name)

        with open(save_path, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]

                if cat_id not in COCO_ID_TO_INDEX:
                    continue

                class_id = COCO_ID_TO_INDEX[cat_id]

                x_min, y_min, w, h = ann["bbox"]

                # x_center, y_center 계산
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h

                # w, h 정규화
                w /= img_w
                h /= img_h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("YOLO 레이블 변환 완료:", save_label_dir)


if __name__ == "__main__":
    convert_coco_to_yolo()
