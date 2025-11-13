import os
import json
import cv2
from collections import defaultdict

# COCO category_id → 0~79 매핑 테이블
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

# COCO 80 class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def visualize_coco_json(
    img_dir="datasets/coco/images/val2017",
    json_path="datasets/coco/annotations/instances_val2017.json",
    save_dir="datasets/images_marked"
):

    os.makedirs(save_dir, exist_ok=True)

    print("COCO annotation JSON 로딩 중:", json_path)
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # image id → filename 매핑
    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # image id → annotation 리스트 매핑
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    print("총 이미지 수:", len(coco["images"]))

    for image_id, filename in image_id_to_file.items():
        img_path = os.path.join(img_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            print("이미지를 읽을 수 없음:", img_path)
            continue

        anns = ann_by_image[image_id]

        for ann in anns:
            category_id = ann["category_id"]

            if category_id not in COCO_ID_TO_INDEX:
                print("Unknown category:", category_id)
                continue

            # category_id → 0~79 인덱스 변환
            class_index = COCO_ID_TO_INDEX[category_id]
            class_name = COCO_CLASSES[class_index]

            # bbox = [x_min, y_min, width, height]
            x_min, y_min, w, h = ann["bbox"]
            x1 = int(x_min)
            y1 = int(y_min)
            x2 = int(x_min + w)
            y2 = int(y_min + h)

            color = (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)

    print("저장 완료 →", save_dir)


if __name__ == "__main__":
    visualize_coco_json()
