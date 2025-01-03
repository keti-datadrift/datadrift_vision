import os
import random
from .image_to_list import image_to_list

def process_images(orig_file_path, compare_file_path):
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

    return train_images, test_images, compare_images