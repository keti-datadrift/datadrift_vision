import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import os
import cv2
from albumentations import (
    Compose, GaussianBlur, MotionBlur, RandomBrightnessContrast, CoarseDropout
)
from alibi_detect.cd import ClassifierDrift
import torch.nn as nn
import torchvision.models as models
from alibi_detect.saving import save_detector, load_detector
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
import random

def image_to_list(paths, list):
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        image = image.astype('float32')
        list.append(image)

def use_aliab(orig_file_path, compare_file_path):
    extensions = (".bmp", ".jpeg", ".jpg", ".png")

    orig_image_paths = [os.path.join(orig_file_path,i) for i in os.listdir(orig_file_path) if os.path.splitext(i)[1] in extensions]
    compare_image_paths = [os.path.join(compare_file_path,i) for i in os.listdir(compare_file_path) if os.path.splitext(i)[1] in extensions]

    random.shuffle(orig_image_paths)

    split_index = len(orig_image_paths) // 2
    train_image_paths = orig_image_paths[:split_index]
    test_image_paths = orig_image_paths[split_index:]

    train_images = []
    test_images = []
    compare_images = []

    image_to_list(train_image_paths, train_images)
    image_to_list(test_image_paths, test_images)
    image_to_list(compare_image_paths, compare_images)

    # 변형 객체 생성
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



    tf.random.set_seed(0)

    model = tf.keras.Sequential(
      [
          Input(shape=(32, 32, 3)),
          Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
          Flatten(),
          Dense(2, activation='softmax')
      ]
    )

    cd = ClassifierDrift(image_by_type[0], model, p_val=.05, train_size=.75, epochs=1)

    save=False

    if save:
        filepath = 'DataDrift_detector'
        save_detector(cd, filepath)

    # Load detector
    # cd = load_detector(filepath)

    labels = ['No!', 'Yes!']

    t = timer()
    preds = cd.predict(image_by_type[1])
    dt = timer() - t
    print('No corruption')
    print(f'Drift? {labels[preds["data"]["is_drift"]]}')
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print(f'Time (s) {dt:.3f}')

    if isinstance(X_c, list):
        for x, c in zip(X_c, X_c_names):
            t = timer()
            preds = cd.predict(x)
            dt = timer() - t
            print('')
            print(f'Corruption type : {c}')
            print(f'Drift? {labels[preds["data"]["is_drift"]]}')
            print(f'p-value? : {preds["data"]["p_val"]:.3f}')
            print(f'Time (s) : {dt:.3f}')





