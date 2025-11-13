"""
Visualize YOLO labels on images
Draws bounding boxes and class labels on images based on YOLO format labels

Usage:
    python utils/visualize_labels.py
    python utils/visualize_labels.py --input E:/drift_proj/datasets/merged_data/val
    python utils/visualize_labels.py --input E:/drift_proj/datasets/merged_data/train --output-suffix _annotated
"""

import cv2
import yaml
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple
import numpy as np

# Color palette for different classes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Dark Blue
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Dark Red
    (128, 128, 0),    # Dark Cyan
    (128, 0, 128),    # Dark Magenta
    (0, 128, 128),    # Dark Yellow
    (255, 128, 0),    # Orange
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Light Green
    (0, 255, 128),    # Spring Green
]

def get_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for a class ID"""
    return COLORS[class_id % len(COLORS)]

def load_class_names(data_yaml_path: Path) -> Dict[int, str]:
    """Load class names from data.yaml file"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        names = data.get('names', {})

        # Convert to dict if it's a list
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}

        return names
    except Exception as e:
        print(f"⚠️ Warning: Could not load class names from {data_yaml_path}: {e}")
        return {}

def yolo_to_bbox(x_center: float, y_center: float, width: float, height: float,
                 img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (normalized) to pixel coordinates

    Args:
        x_center, y_center, width, height: Normalized YOLO format (0.0-1.0)
        img_width, img_height: Image dimensions in pixels

    Returns:
        (x1, y1, x2, y2): Top-left and bottom-right corners in pixels
    """
    # Convert normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    # Calculate corners
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    return x1, y1, x2, y2

def draw_boxes(image: np.ndarray, label_path: Path, class_names: Dict[int, str]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image

    Args:
        image: Input image (BGR format)
        label_path: Path to YOLO format label file
        class_names: Dictionary mapping class IDs to names

    Returns:
        Annotated image
    """
    if not label_path.exists():
        print(f"   ⚠️ Label file not found: {label_path}")
        return image

    img_height, img_width = image.shape[:2]
    img_annotated = image.copy()

    # Read label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"   ⚠️ Error reading {label_path}: {e}")
        return image

    # Draw each bounding box
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"   ⚠️ Invalid line in {label_path}: {line}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            print(f"   ⚠️ Invalid values in {label_path}: {line} - {e}")
            continue

        # Convert YOLO format to pixel coordinates
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)

        # Get color and class name
        color = get_color(class_id)
        class_name = class_names.get(class_id, f'Class {class_id}')

        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label_text = f'{class_name} ({class_id})'

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw filled rectangle for text background
        cv2.rectangle(
            img_annotated,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            img_annotated,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )

    return img_annotated

def visualize_dataset(dataset_path: Path, output_suffix: str = '_marked',
                     data_yaml_name: str = 'data.yaml'):
    """
    Visualize all images in a dataset with their labels

    Args:
        dataset_path: Path to dataset directory containing images/ and labels/
        output_suffix: Suffix for output directory name
        data_yaml_name: Name of the YAML file containing class names
    """
    print(f"\n{'='*60}")
    print(f"📊 YOLO Label Visualizer")
    print(f"{'='*60}")

    # Find images and labels directories
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    output_dir = dataset_path / f'images{output_suffix}'

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory: {output_dir}")

    # Load class names
    data_yaml_path = dataset_path.parent / data_yaml_name
    if not data_yaml_path.exists():
        # Try alternative locations
        data_yaml_path = dataset_path / data_yaml_name

    class_names = load_class_names(data_yaml_path)
    if class_names:
        print(f"✅ Loaded {len(class_names)} class names from {data_yaml_path}")
        for class_id, class_name in sorted(class_names.items()):
            print(f"   {class_id}: {class_name}")
    else:
        print(f"⚠️ Using default class names (Class 0, Class 1, ...)")

    print(f"\n{'='*60}")
    print(f"Processing images...")
    print(f"{'='*60}\n")

    # Process all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in images_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images\n")

    success_count = 0
    no_label_count = 0
    error_count = 0

    for idx, image_path in enumerate(sorted(image_files), 1):
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"

        # Read image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[{idx}/{len(image_files)}] ❌ Failed to read: {image_path.name}")
                error_count += 1
                continue
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ❌ Error reading {image_path.name}: {e}")
            error_count += 1
            continue

        # Check if label exists
        if not label_path.exists():
            print(f"[{idx}/{len(image_files)}] ⚠️ No label for: {image_path.name}")
            # Save original image without annotations
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), image)
            no_label_count += 1
            continue

        # Draw boxes
        try:
            annotated = draw_boxes(image, label_path, class_names)

            # Save annotated image
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), annotated)

            print(f"[{idx}/{len(image_files)}] ✅ {image_path.name}")
            success_count += 1

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ❌ Error processing {image_path.name}: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully annotated: {success_count}")
    print(f"⚠️ No labels found: {no_label_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Total images: {len(image_files)}")
    print(f"📂 Output directory: {output_dir}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO labels on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize validation set (default)
  python utils/visualize_labels.py

  # Visualize training set
  python utils/visualize_labels.py --input E:/drift_proj/datasets/merged_data/train

  # Custom output suffix
  python utils/visualize_labels.py --output-suffix _annotated

  # Custom data.yaml location
  python utils/visualize_labels.py --data-yaml custom_data.yaml
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default=r'E:/drift_proj/datasets/merged_data/val',
        help='Path to dataset directory (default: E:/drift_proj/datasets/merged_data/val)'
    )

    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_marked',
        help='Suffix for output directory name (default: _marked)'
    )

    parser.add_argument(
        '--data-yaml',
        type=str,
        default='data.yaml',
        help='Name of YAML file with class names (default: data.yaml)'
    )

    args = parser.parse_args()

    # Convert to Path
    dataset_path = Path(args.input)

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Run visualization
    visualize_dataset(
        dataset_path=dataset_path,
        output_suffix=args.output_suffix,
        data_yaml_name=args.data_yaml
    )

if __name__ == '__main__':
    main()
