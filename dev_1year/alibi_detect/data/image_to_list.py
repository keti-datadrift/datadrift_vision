import cv2

def image_to_list(paths, list):
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        image = image.astype('float32')
        list.append(image)
