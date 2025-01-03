import numpy as np
from albumentations import (
    Compose, GaussianBlur, MotionBlur, RandomBrightnessContrast, CoarseDropout
)

def transform_object(train_images, test_images, compare_images):
    gaussian_noise = GaussianBlur(blur_limit=7, p=1)  # 가우시안 노이즈
    motion_blur = MotionBlur(blur_limit=7, p=1)  # 모션 블러
    brightness = RandomBrightnessContrast(rightness_limit=0.2, p=1)  # 밝기 변형
    pixelate = CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=1)  # 픽셀화

    # 변형을 각각 적용하고 결과를 서로 다른 변수에 저장
    train_images_gaussian_noise = [gaussian_noise(image=image)["image"] for image in train_images]
    train_images_motion_blur = [motion_blur(image=image)["image"] for image in train_images]
    train_images_brightness = [brightness(image=image)["image"] for image in train_images]
    train_images_pixelate = [pixelate(image=image)["image"] for image in train_images]

    image_by_type = [train_images, test_images, train_images_gaussian_noise, train_images_motion_blur, train_images_brightness, train_images_pixelate, compare_images]
    type_names = ["train_images", "test_images", "train_images_gaussian_noise", "train_images_motion_blur", "train_images_brightness", "train_images_pixelate", "compare_images"]

    for idx, type in enumerate(image_by_type):
        type = np.array(type)
        type = np.stack(type)
        image_by_type[idx] = type
        print(f"{type_names[idx]} : {image_by_type[idx].shape}")

    X_c = [image_by_type[i] for i in range(2, len(image_by_type))]
    X_c_names = [type_names[i] for i in range(2, len(type_names))]

    return image_by_type[0], image_by_type[1], X_c, X_c_names