#!/usr/bin/env python3
"""
FIXED: YOLO 모델 재학습 스크립트
Key fixes:
1. Load previous trained model for fine-tuning (not base model)
2. Ensure class consistency between old and new datasets
3. Proper model path handling
4. Backup previous model before training
"""
USE_PREV_MODEL = True
import os
import sys
from glob import glob
from datetime import datetime
from ultralytics import YOLO
import shutil
import yaml
from pathlib import Path
import time
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(base_abspath)

# YOLO80 클래스명 가져오기
from vision_analysis.class_names_yolo80n import TEXT_LABELS_80
class_names_list = None
class_names_list = list(TEXT_LABELS_80.values())

# Load datasets base path from config
def get_datasets_base_path():
    """
    Load datasets base path from config.yaml
    Falls back to old location if not specified in config
    """
    config_path = os.path.join(base_abspath, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        datasets_config = config.get("datasets", {})
        datasets_base = datasets_config.get("base_path", None)
        if datasets_base:
            return datasets_base
    # Fallback to old location inside project
    return os.path.join(base_abspath, "datasets")

# Global datasets base path
DATASETS_BASE_PATH = get_datasets_base_path()
print(f"Using datasets base path: {DATASETS_BASE_PATH}")

def extract_class_names_from_labels(labels_dirs):
    """
    레이블 파일에서 사용된 클래스 ID를 추출하고 YOLO80 클래스명으로 매핑
    """

    
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
    
    sorted_class_ids = sorted(class_ids)
    class_mapping = {}
    
    for class_id in sorted_class_ids:
        if class_id < len(class_names_list):
            class_mapping[class_id] = class_names_list[class_id]
        else:
            class_mapping[class_id] = f"class_{class_id}"
    
    return class_mapping, sorted_class_ids


def get_dataset_version():
    """
    DataOps: Generate dataset version based on existing versions
    Returns: version string (e.g., 'v1', 'v2', 'v3')
    """
    dataset_base = Path(DATASETS_BASE_PATH) / "splitted"

    if not dataset_base.exists():
        return "v1"

    # Find existing data*.yaml files
    existing_yamls = list(dataset_base.glob("data_*.yaml"))

    if not existing_yamls:
        return "v1"

    # Extract version numbers
    versions = []
    for yaml_file in existing_yamls:
        # Format: data_YYYYMMDD_vN.yaml
        name = yaml_file.stem
        parts = name.split('_')
        if len(parts) >= 3 and parts[-1].startswith('v'):
            try:
                version_num = int(parts[-1][1:])
                versions.append(version_num)
            except ValueError:
                continue

    if versions:
        return f"v{max(versions) + 1}"
    else:
        return "v1"


def get_previous_model_info():
    """
    Get information about the previously trained model
    Returns: (model_path, class_names) or (None, None) if not found
    """
    runs_base_dir = os.path.join(base_abspath, "runs")

    if not os.path.exists(runs_base_dir):
        print("No previous training found - will train from base model")
        return None, None

    # Find retrain, retrain2, retrain3 etc folders
    subdirs = [d for d in os.listdir(runs_base_dir)
               if os.path.isdir(os.path.join(runs_base_dir, d)) and d.startswith('retrain')]

    if not subdirs:
        print("No previous training results found")
        return None, None

    # Find most recent folder by modification time
    latest_dir = max([os.path.join(runs_base_dir, d) for d in subdirs], key=os.path.getmtime)

    print(f"Latest training folder: {latest_dir}")

    # Try both possible paths for best.pt
    possible_paths = [
        os.path.join(latest_dir, "weights", "best.pt"),
        os.path.join(latest_dir, "best.pt")
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break  # ✅ Use break to stop after finding first valid path

    if not model_path:
        print(f"Previous model not found in {latest_dir}")
        return None, None

    # Load model to get class names
    try:
        prev_model = YOLO(model_path)
        prev_classes = list(prev_model.names.values()) if hasattr(prev_model, 'names') else None
        print(f"Found previous model: {model_path}")
        print(f"Previous model classes: {prev_classes}")
        return model_path, prev_classes  # ✅ Return BOTH values
    except Exception as e:
        print(f"Error loading previous model: {e}")
        return None, None  # ✅ Return BOTH values even on error


def merge_and_split_datasets(source_dirs, output_dir=None,
                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             max_samples=None, random_seed=42):
    """여러 데이터셋을 머지하고 train/val/test로 분할"""
    import random

    print(f"\n[{datetime.now()}] Starting dataset merge and split...")
    print(f"Split ratio - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    if max_samples:
        print(f"Max samples limit: {max_samples}")

    # Use DATASETS_BASE_PATH for output
    if output_dir is None:
        output_path = Path(DATASETS_BASE_PATH) / "splitted"
    else:
        output_path = Path(output_dir)

    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for src_dir in source_dirs:
        # src_dir is now expected to be an absolute path
        src_path = Path(src_dir)
        images_dir = src_path / 'images'
        labels_dir = src_path / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
        
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
    
    if max_samples and total_files > max_samples:
        random.seed(random_seed)
        all_pairs = random.sample(all_pairs, max_samples)
        total_files = max_samples
        print(f"Limited to {max_samples} samples")
    
    random.seed(random_seed)
    random.shuffle(all_pairs)
    
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    stats = {}
    for split_name, pairs in splits.items():
        print(f"\nCopying {len(pairs)} files to {split_name}...")
        
        for pair in pairs:
            dest_img = output_path / split_name / 'images' / pair['image'].name
            src_img = str(pair['image'])

            # Skip if source doesn't exist
            if not os.path.exists(src_img):
                print(f"Warning: Source image not found: {src_img}")
                continue

            # Remove destination if exists (with retry logic for Windows file locks)
            if os.path.exists(dest_img):
                try:
                    os.remove(dest_img)
                    time.sleep(0.01)
                except PermissionError:
                    # If file is locked, skip it (it's already there from a previous run)
                    print(f"Warning: File is locked, skipping: {dest_img}")
                    continue

            shutil.copy2(src_img, dest_img)

            dest_label = output_path / split_name / 'labels' / pair['label'].name
            src_label = str(pair['label'])

            # Skip if label source doesn't exist
            if not os.path.exists(src_label):
                print(f"Warning: Source label not found: {src_label}")
                continue

            shutil.copy2(src_label, dest_label)
        
        stats[split_name] = len(pairs)
    
    # ===== DataOps: Create versioned data.yaml =====
    # Generate version and date for yaml filename
    dataset_version = get_dataset_version()
    date_str = datetime.now().strftime("%Y%m%d")
    yaml_filename = f"data_{date_str}_{dataset_version}.yaml"
    yaml_path = output_path / yaml_filename

    # Create data.yaml with metadata
    data_yaml = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names_list),
        'names': class_names_list,
        # DataOps metadata
        'metadata': {
            'version': dataset_version,
            'created_date': datetime.now().isoformat(),
            'source_dirs': source_dirs,
            'split_ratio': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'total_samples': total_files,
            'samples_per_split': stats,
            'random_seed': random_seed
        }
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    # Also create a symlink/copy as data.yaml for compatibility
    data_yaml_link = output_path / "data.yaml"
    shutil.copy2(yaml_path, data_yaml_link)

    print(f"\n{'='*60}")
    print(f"Dataset merge and split completed!")
    print(f"  Train: {stats['train']} ({stats['train']/total_files*100:.1f}%)")
    print(f"  Val:   {stats['val']} ({stats['val']/total_files*100:.1f}%)")
    print(f"  Test:  {stats['test']} ({stats['test']/total_files*100:.1f}%)")
    print(f"  Total: {total_files}")
    print(f"Output directory: {output_path}")
    print(f"\nDataOps - Versioned YAML created:")
    print(f"  Version: {dataset_version}")
    print(f"  File: {yaml_filename}")
    print(f"  Symlink: data.yaml -> {yaml_filename}")
    print(f"{'='*60}\n")

    return {
        'output_dir': str(output_path),
        'yaml_path': str(yaml_path),
        'yaml_filename': yaml_filename,
        'version': dataset_version,
        'stats': stats,
        'total': total_files
    }


def get_latest_merged_dataset():
    """
    DataOps: Find the latest merged dataset YAML file
    Returns: (yaml_path, version) or (None, None) if not found
    """
    merged_base = Path(DATASETS_BASE_PATH) / "merged_data"

    if not merged_base.exists():
        return None, None

    # Find existing data_merged*.yaml files
    existing_yamls = list(merged_base.glob("data_merged_*.yaml"))

    if not existing_yamls:
        return None, None

    # Find the most recent merged dataset by modification time
    latest_yaml = max(existing_yamls, key=lambda p: p.stat().st_mtime)

    # Extract version from filename
    name = latest_yaml.stem
    parts = name.split('_')
    version = parts[-1] if parts[-1].startswith('v') else "unknown"

    print(f"Found latest merged dataset: {latest_yaml.name} ({version})")

    return str(latest_yaml), version


def clean_merged_data_directory(output_dir):
    """
    Clean up the merged_data directory to prevent dataset accumulation
    Deletes the entire merged_data directory completely
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"\n{'='*60}")
        print("MERGED DATA - Directory does not exist")
        print(f"{'='*60}")
        print(f"  Will be created fresh")
        print(f"{'='*60}\n")
        return

    print(f"\n{'='*60}")
    print("DELETING ENTIRE MERGED_DATA DIRECTORY")
    print(f"{'='*60}")

    # Count total files before deletion
    total_files = 0
    for split in ['train', 'val', 'test']:
        images_dir = output_path / split / 'images'
        labels_dir = output_path / split / 'labels'
        img_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        lbl_count = len(list(labels_dir.glob('*'))) if labels_dir.exists() else 0
        total_files += img_count + lbl_count
        if img_count > 0 or lbl_count > 0:
            print(f"  {split}: {img_count} images, {lbl_count} labels")

    print(f"  Total files: {total_files}")
    print(f"  Deleting entire directory: {output_path}")

    # Delete entire directory with error handler for Windows
    def handle_remove_readonly(func, path, _exc):
        """Error handler for Windows readonly files"""
        import stat
        try:
            os.chmod(path, stat.S_IWUSR | stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"    Warning: Could not delete {path}: {e}")

    try:
        shutil.rmtree(output_path, onerror=handle_remove_readonly)
        print(f"✅ Entire directory deleted successfully")
    except Exception as e:
        print(f"❌ Error deleting directory: {e}")
        print(f"   Some files may be locked by another process")
        print(f"   Attempting alternative cleanup method...")

        # Alternative: Try to delete as many files as possible
        import stat
        deleted_count = 0
        failed_count = 0

        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                dir_path = output_path / split / subdir
                if dir_path.exists():
                    for file in dir_path.glob('*'):
                        try:
                            # Force remove readonly attribute
                            os.chmod(file, stat.S_IWUSR | stat.S_IWRITE)
                            file.unlink()
                            deleted_count += 1
                        except Exception:
                            failed_count += 1

        if deleted_count > 0:
            print(f"   Deleted {deleted_count} files, {failed_count} failed")

        # Try to remove empty directories
        try:
            for split in ['train', 'val', 'test']:
                for subdir in ['images', 'labels']:
                    dir_path = output_path / split / subdir
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        dir_path.rmdir()

                split_path = output_path / split
                if split_path.exists() and not any(split_path.iterdir()):
                    split_path.rmdir()

            if output_path.exists() and not any(output_path.iterdir()):
                output_path.rmdir()
                print(f"✅ Cleaned up directory structure")
        except Exception:
            pass

    print(f"{'='*60}\n")


def merge_new_with_coco(new_data_yaml_path, coco_yaml_path=None, output_dir="datasets/merged_data",
                         new_ratio=1, old_ratio=3, random_seed=42, use_latest_merged=False):
    """
    새로 생성된 데이터셋과 기존 데이터셋(COCO 또는 이전 merged)을 new:old 비율로 병합

    Args:
        new_data_yaml_path: 새로 생성된 data.yaml 경로 (datasets/splitted/data.yaml)
        coco_yaml_path: COCO data.yaml 경로 (첫 실행 시)
        output_dir: 병합된 데이터셋 출력 디렉토리
        new_ratio: 새 데이터 비율 (기본값: 1)
        old_ratio: 기존 데이터 비율 (기본값: 3)
        random_seed: 랜덤 시드
        use_latest_merged: True면 최신 merged dataset 사용, False면 coco_yaml_path 사용

    Returns:
        dict: 병합된 데이터셋 정보
    """
    import random

    # Clean up old merged data to prevent accumulation
    clean_merged_data_directory(output_dir)

    # Determine which base dataset to use - ALWAYS USE COCO, NOT PREVIOUS MERGED
    base_type = "coco"
    use_latest_merged = False  # Force to always use COCO to prevent accumulation
    if use_latest_merged:
        latest_merged, _ = get_latest_merged_dataset()
        if latest_merged:
            coco_yaml_path = latest_merged
            base_type = "previous_merged"
            print(f"\n{'='*60}")
            print(f"Merging NEW dataset with LATEST MERGED dataset")
            print(f"Base dataset: {Path(coco_yaml_path).name}")
            print(f"Ratio - New:Old = {new_ratio}:{old_ratio}")
            print(f"{'='*60}\n")
        else:
            print("\nWarning: No previous merged dataset found")
            if not coco_yaml_path:
                print("Error: No COCO dataset path provided and no previous merged dataset found")
                return None
            print("Falling back to COCO dataset\n")
            base_type = "coco"
            print(f"\n{'='*60}")
            print(f"Merging NEW dataset with COCO dataset (first time)")
            print(f"Ratio - New:Old = {new_ratio}:{old_ratio}")
            print(f"{'='*60}\n")
    else:
        if not coco_yaml_path:
            print("Error: coco_yaml_path must be provided when use_latest_merged=False")
            return None
        print(f"\n{'='*60}")
        print(f"Merging NEW dataset with COCO dataset")
        print(f"Ratio - New:Old = {new_ratio}:{old_ratio}")
        print(f"{'='*60}\n")

    # Load YAML files
    with open(new_data_yaml_path, 'r', encoding='utf-8') as f:
        new_data = yaml.safe_load(f)

    with open(coco_yaml_path, 'r', encoding='utf-8') as f:
        coco_data = yaml.safe_load(f)

    # Create output directory structure
    # output_dir is now expected to be an absolute path
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {}

    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"\n--- Processing {split} split ---")

        # Get paths from YAML
        new_base_path = Path(base_abspath) / new_data['path']
        new_images_dir = new_base_path / new_data[split]
        new_labels_dir = new_base_path / new_data[split].replace('images', 'labels')

        # COCO paths (adjust based on your COCO dataset structure)
        coco_base_path = Path(coco_data['path'])
        if not coco_base_path.is_absolute():
            coco_base_path = Path(base_abspath) / coco_data['path']

        coco_images_dir = coco_base_path / coco_data[split]
        coco_labels_dir = coco_base_path / coco_data[split].replace('images', 'labels')

        # Collect image-label pairs from NEW dataset
        new_pairs = []
        if new_images_dir.exists():
            for img_file in list(new_images_dir.glob('*.jpg')) + list(new_images_dir.glob('*.png')):
                label_file = new_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    new_pairs.append({
                        'image': img_file,
                        'label': label_file,
                        'source': 'new'
                    })

        print(f"  NEW dataset: {len(new_pairs)} samples")

        # Collect image-label pairs from COCO dataset
        coco_pairs = []
        if coco_images_dir.exists():
            for img_file in list(coco_images_dir.glob('*.jpg')) + list(coco_images_dir.glob('*.png')):
                label_file = coco_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    coco_pairs.append({
                        'image': img_file,
                        'label': label_file,
                        'source': 'coco'
                    })

        print(f"  COCO dataset: {len(coco_pairs)} samples")

        # Calculate target counts based on ratio
        if len(new_pairs) == 0:
            print(f"  Warning: No new samples found for {split}, using only COCO data")
            selected_new = []
            selected_coco = coco_pairs
        elif len(coco_pairs) == 0:
            print(f"  Warning: No COCO samples found for {split}, using only new data")
            selected_new = new_pairs
            selected_coco = []
        else:
            # Calculate how many samples to take from each dataset
            # If we have N new samples, we want N*old_ratio/new_ratio COCO samples
            target_new_count = len(new_pairs)
            target_coco_count = int(target_new_count * old_ratio / new_ratio)

            # Use all new samples
            selected_new = new_pairs

            # Sample from COCO to match the ratio
            if len(coco_pairs) >= target_coco_count:
                random.seed(random_seed)
                selected_coco = random.sample(coco_pairs, target_coco_count)
            else:
                print(f"  Warning: Not enough COCO samples ({len(coco_pairs)} < {target_coco_count})")
                print(f"  Using all available COCO samples")
                selected_coco = coco_pairs

        print(f"  Selected - NEW: {len(selected_new)}, COCO: {len(selected_coco)}")
        print(f"  Actual ratio - NEW:COCO = {len(selected_new)}:{len(selected_coco)}")

        # Combine and shuffle
        all_pairs = selected_new + selected_coco
        random.seed(random_seed)
        random.shuffle(all_pairs)

        # Copy files to output directory
        print(f"  Copying {len(all_pairs)} files...")
        copied_count = 0
        seen_filenames = set()  # Track unique filenames to prevent duplicates

        for idx, pair in enumerate(all_pairs):
            # Skip if source files don't exist
            src_img = str(pair['image'])
            src_label = str(pair['label'])

            if not os.path.exists(src_img):
                print(f"  Warning: Source image not found: {src_img}")
                continue

            if not os.path.exists(src_label):
                print(f"  Warning: Source label not found: {src_label}")
                continue

            # Generate unique filename to avoid conflicts
            source_prefix = 'new_' if pair['source'] == 'new' else 'coco_'
            base_name = pair['image'].stem
            ext = pair['image'].suffix

            # Use index to ensure uniqueness instead of copied_count
            unique_name = f"{source_prefix}{base_name}"

            # If filename already seen, append unique index
            if unique_name in seen_filenames:
                unique_name = f"{source_prefix}{base_name}_{idx}"

            seen_filenames.add(unique_name)

            dest_img = output_path / split / 'images' / f"{unique_name}{ext}"
            dest_label = output_path / split / 'labels' / f"{unique_name}.txt"

            try:
                shutil.copy2(src_img, dest_img)
                shutil.copy2(src_label, dest_label)
                copied_count += 1
            except Exception as e:
                print(f"  Warning: Failed to copy {src_img}: {e}")
                continue

        stats[split] = {
            'new': len(selected_new),
            'coco': len(selected_coco),
            'total': len(all_pairs)
        }

    # ===== DataOps: Create versioned merged data.yaml =====
    # Generate version and date for yaml filename
    dataset_version = get_dataset_version()
    date_str = datetime.now().strftime("%Y%m%d")
    yaml_filename = f"data_merged_{date_str}_{dataset_version}.yaml"
    merged_yaml_path = output_path / yaml_filename

    # Create merged data.yaml with metadata
    merged_yaml = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': new_data['nc'],
        'names': new_data['names'],
        # DataOps metadata
        'metadata': {
            'version': dataset_version,
            'created_date': datetime.now().isoformat(),
            'merge_type': base_type,  # 'coco' or 'previous_merged'
            'base_dataset': Path(coco_yaml_path).name if coco_yaml_path else 'unknown',
            'merge_ratio': {
                'new': new_ratio,
                'base': old_ratio
            },
            'source_yamls': {
                'new': new_data_yaml_path,
                'base': coco_yaml_path
            },
            'samples_per_split': stats,
            'random_seed': random_seed
        }
    }

    with open(merged_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_yaml, f, allow_unicode=True, sort_keys=False)

    # Also create a symlink/copy as data.yaml for compatibility
    data_yaml_link = output_path / "data.yaml"
    shutil.copy2(merged_yaml_path, data_yaml_link)

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset Merge Summary:")
    print(f"{'='*60}")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        print(f"  NEW samples:  {stats[split]['new']}")
        print(f"  COCO samples: {stats[split]['coco']}")
        print(f"  Total:        {stats[split]['total']}")
        if stats[split]['total'] > 0:
            new_pct = stats[split]['new'] / stats[split]['total'] * 100
            coco_pct = stats[split]['coco'] / stats[split]['total'] * 100
            print(f"  Ratio:        NEW {new_pct:.1f}% : COCO {coco_pct:.1f}%")

    print(f"\nDataOps - Versioned merged YAML created:")
    print(f"  Version: {dataset_version}")
    print(f"  File: {yaml_filename}")
    print(f"  Symlink: data.yaml -> {yaml_filename}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")

    return {
        'output_dir': str(output_path),
        'yaml_path': str(merged_yaml_path),
        'yaml_filename': yaml_filename,
        'version': dataset_version,
        'stats': stats
    }


def train_model():
    """
    YOLO 모델 재학습 함수 - FIXED VERSION
    """

    print(f"[{datetime.now()}] Starting merge_and_split_datasets...")
    merge_and_split_datasets([
        os.path.join(DATASETS_BASE_PATH, 'collected', 'good_images'),
        os.path.join(DATASETS_BASE_PATH, 'collected', 'wronged_images')
    ])

    # ===== OPTION 1: Merge with COCO dataset (First Time - 1:3 ratio) =====
    # Uncomment below for FIRST RUN to merge with COCO dataset
    #
    # merge_result = merge_new_with_coco(
    #     new_data_yaml_path=os.path.join(DATASETS_BASE_PATH, "splitted/data.yaml"),
    #     coco_yaml_path=os.path.join(DATASETS_BASE_PATH, "coco/coco.yaml"),  # Update path
    #     output_dir=os.path.join(DATASETS_BASE_PATH, "merged_data"),
    #     new_ratio=1,
    #     old_ratio=3,
    #     random_seed=42,
    #     use_latest_merged=False  # First time: use COCO
    # )
    # dataset_dir = merge_result['output_dir']  # Use merged dataset for training
    # =======================================================================

    # ===== OPTION 2: Merge with Latest Merged Dataset (Subsequent Runs) =====
    # Uncomment below for SUBSEQUENT RUNS to merge with latest merged dataset

    merge_result = merge_new_with_coco(
        new_data_yaml_path=os.path.join(DATASETS_BASE_PATH, "splitted", "data.yaml"),
        coco_yaml_path=os.path.join(DATASETS_BASE_PATH, "cocox", "coco.yaml"),  # Fallback
        output_dir=os.path.join(DATASETS_BASE_PATH, "merged_data"),
        new_ratio=1,
        old_ratio=3,
        random_seed=42,
        use_latest_merged=True  # Use latest merged dataset
    )
    dataset_dir = merge_result['output_dir']  # Use merged dataset for training
    yaml_filename = merge_result['yaml_filename']
    # ===========================================================================
    yaml_path = os.path.join(dataset_dir, yaml_filename) 
    print(f"[{datetime.now()}] Starting YOLO model retraining...")
    
    # Load config
    config_path = os.path.join(base_abspath, "config.yaml")
    if not os.path.exists(config_path):
        print(f"config.yaml not found at {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    YOLO_MODEL = config["yolo_model"]["model_name"]
    CONF_THRESH = float(config["yolo_model"]["conf_thresh"])
    
    # Dataset paths
    # dataset_dir = os.path.join(base_abspath, "datasets", "splitted")
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_images_dir = os.path.join(dataset_dir, "val", "images")
    val_labels_dir = os.path.join(dataset_dir, "val", "labels")
    test_images_dir = os.path.join(dataset_dir, "test", "images")
    
    # Verify directories
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
    
    # Count images
    train_images = glob(os.path.join(train_images_dir, "*.jpg"))
    val_images = glob(os.path.join(val_images_dir, "*.jpg"))
    test_images = glob(os.path.join(test_images_dir, "*.jpg"))
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} images")
    
    if len(train_images) == 0 or len(val_images) == 0:
        print("Error: No training or validation images found!")
        return
    
    # Extract class info from NEW dataset
    labels_dirs = [train_labels_dir, val_labels_dir]
    if os.path.exists(os.path.join(dataset_dir, "test", "labels")):
        labels_dirs.append(os.path.join(dataset_dir, "test", "labels"))
    
    class_mapping, class_ids = extract_class_names_from_labels(labels_dirs)
    
    if not class_mapping:
        print("Error: No valid class labels found in dataset!")
        return
    
    print(f"\nDetected {len(class_mapping)} classes in NEW dataset:")
    for class_id, class_name in class_mapping.items():
        print(f"  ID {class_id}: {class_name}")
    
    # global class_names_list
    # class_names_list = [class_mapping[class_id] for class_id in class_ids]
    
    # ===== CRITICAL: Check previous model and class consistency =====
    prev_model_path, prev_classes = get_previous_model_info()
    
    # Determine if this is fine-tuning or fresh training
    drift_detected = False
    use_previous_model = False
    
    if prev_model_path and prev_classes:
        # Check class consistency
        if prev_classes == class_names_list:
            print("\n✅ Class names MATCH - Can safely fine-tune from previous model")
            drift_detected = True
            use_previous_model = True
        else:
            print("\n⚠️  WARNING: Class mismatch detected!")
            print(f"   Previous classes: {prev_classes}")
            print(f"   New classes: {class_names_list}")
            print("   Will train from BASE model (not fine-tuning)")
            drift_detected = False
            use_previous_model = False
    else:
        print("\n📝 No previous model found - Training from base model")
        drift_detected = False
        use_previous_model = False
    
    # # Create data.yaml
    # data_yaml = {
    #     "path": dataset_dir,
    #     "train": "train/images",
    #     "val": "val/images",
    #     "test": "test/images",
    #     "nc": len(class_names_list),
    #     "names": class_names_list
    # }
    # yaml_path = os.path.join(dataset_dir, "data.yaml")
    # with open(yaml_path, "w") as f:
    #     yaml.dump(data_yaml, f, allow_unicode=True)

    print(f"\ndata.yaml created at: {yaml_path}")
    print(f"Number of classes (nc): {len(class_names_list)}")
    
    # ===== CRITICAL FIX: Load correct model =====
    runs_dir = os.path.join(base_abspath, "runs")
    
    # Backup previous model if exists
    if use_previous_model and prev_model_path:
        backup_path = os.path.join(base_abspath, "backup_best_model.pt")
        shutil.copy2(prev_model_path, backup_path)
        print(f"\n✅ Backed up previous model to: {backup_path}")
    
    # DON'T delete runs directory - we need it for comparison!
    # Instead, YOLO will create a new numbered subfolder

    # Load appropriate model
    # Only use previous model if USE_PREV_MODEL is True AND prev_model_path exists
    use_previous_model_final = USE_PREV_MODEL and prev_model_path is not None and os.path.exists(prev_model_path)

    if use_previous_model_final:
        print(f"\n🔄 Fine-tuning mode: Loading previous trained model")
        model = YOLO(prev_model_path)
        lr0 = 0.001  # Low LR for fine-tuning
        lrf = 0.01
        print(f"   Model: {prev_model_path}")
        print(f"   Learning rate: {lr0}")
    else:
        print(f"\n🆕 Fresh training mode: Loading base model")
        if USE_PREV_MODEL and prev_model_path is None:
            print(f"   Note: USE_PREV_MODEL=True but no previous model found")
        model = YOLO(YOLO_MODEL)
        lr0 = 0.001  # Higher LR for fresh training
        lrf = 0.01
        print(f"   Model: {YOLO_MODEL}")
        print(f"   Learning rate: {lr0}")
    
    print(f"\n[{datetime.now()}] Training started...")
    
    # Train model
    # results = model.train(
    #     data=yaml_path,
    #     epochs=100,
    #     imgsz=640,
    #     batch=16,
    #     name="retrain",
    #     project=runs_dir,
    #     device=0,
    #     lr0=lr0,
    #     lrf=lrf,
    #     optimizer='AdamW',
    #     patience=10,
    #     warmup_epochs=3
    # )
    results = model.train(
        # 데이터
        data=yaml_path,  # ⭐ 병합된 데이터 사용
        cache=False,  # Cache images for faster training (first epoch will be slower)

        # 기본 파라미터
        # epochs=20,
        # patience=10,             # 조기 종료
        epochs=5,
        patience=3,             # 조기 종료
        imgsz=640,
        batch=16,
        
        # 학습률 (기존 지식 보존용 낮은 LR)
        lr0=lr0,              # ⭐ 신규만 쓸 때보다 낮춤
        lrf=lrf,
        optimizer='AdamW',
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # 정규화 (과적합 방지)
        weight_decay=0.0005,
        dropout=0.0,
        
        # 레이어 동결 (선택적)
        freeze=10,               # 백본 일부 동결로 재앙적 망각 방지
        
        # 데이터 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,              # 모자이크 증강
        mixup=0.1,               # 믹스업
        copy_paste=0.1,          # 복사-붙여넣기 (신규 객체에 효과적)
        
        # 저장 및 로깅
        save=True,
        save_period=10,
        plots=True,
        
        # 하드웨어
        device=0,
        workers=8,
        
        # 프로젝트
        # project='runs/train',
        project=runs_dir,
        name='retrain'
    )
    print(f"\n[{datetime.now()}] Training completed!")
    print(f"Results saved to: {results.save_dir}")


def evaluate_model(model_path, test_data_yaml=None, test_images_dir=None):
    """모델 평가"""
    print(f"\nEvaluating model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = YOLO(model_path)
    
    if test_data_yaml and os.path.exists(test_data_yaml):
        results = model.val(
            data=test_data_yaml,
            split='test',
            save_json=True,
            save_hybrid=True
        )
    
    elif test_images_dir and os.path.exists(test_images_dir):
        test_images = glob(os.path.join(test_images_dir, "*.jpg"))
        if len(test_images) == 0:
            print(f"Error: No test images found in {test_images_dir}")
            return None
        
        temp_yaml_path = os.path.join(os.path.dirname(test_images_dir), "temp_test.yaml")
        test_parent_dir = os.path.dirname(os.path.dirname(test_images_dir))
        test_labels_dir = os.path.join(os.path.dirname(test_images_dir), "labels")
        
        test_rel_path = os.path.relpath(test_images_dir, test_parent_dir)
        temp_data_yaml = {
            "path": test_parent_dir,
            "train": test_rel_path,
            "val": test_rel_path,
            "test": test_rel_path,
            "nc": len(class_names_list),
            "names": class_names_list
        }
        
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(temp_data_yaml, f, allow_unicode=True)
        
        print(f"Created temporary data.yaml at: {temp_yaml_path}")
        
        results = model.val(
            data=temp_yaml_path,
            split='test',
            save_json=True,
            save_hybrid=True
        )
    
    else:
        print("Error: Either test_data_yaml or test_images_dir must be provided")
        return None
    
    # Print results
    print(f"\n{'='*60}")
    print("Model Evaluation Results:")
    print(f"{'='*60}")
    
    if hasattr(results, 'box'):
        metrics = results.box
        print(f"mAP50: {metrics.map50:.4f}")
        print(f"mAP50-95: {metrics.map:.4f}")
        print(f"Precision: {metrics.mp:.4f}")
        print(f"Recall: {metrics.mr:.4f}")
        
        if hasattr(metrics, 'ap_class_index'):
            print(f"\nPer-class mAP50:")
            for i, (cls_idx, ap) in enumerate(zip(metrics.ap_class_index, metrics.ap50)):
                print(f"  Class {cls_idx}: {ap:.4f}")
    
    print(f"{'='*60}\n")
    
    return results


def evaluate_old_and_new_models():
    """구 모델과 신규 학습된 모델을 테스트 세트로 평가하여 비교"""
    test_images_dir = os.path.join(DATASETS_BASE_PATH, "splitted", "test", "images")

    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        return None
    
    results = {}
    
    # Step 1: OLD model
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
    
    # Step 2: NEW model
    print(f"\n{'='*60}")
    print("Step 2: Evaluating NEW model on test set")
    print(f"{'='*60}")
    
    # runs_dir = os.path.join(base_abspath, "runs", "retrain")
        # 가장 최근 학습된 모델 찾기
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
    
    # Step 3: Compare
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    
    print(f"old_model_name = {old_model_name}")
    print(f"new_model_path = {new_model_path}")

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


def count_images_and_instances_from_labels(test_labels_dir):
    """
    Count number of images and instances per class from test labels directory

    Args:
        test_labels_dir: Path to test labels directory

    Returns:
        tuple: (image_counts_dict, instance_counts_dict)
    """
    image_counts = {}  # Number of images containing each class
    instance_counts = {}  # Total number of instances per class

    if not os.path.exists(test_labels_dir):
        return image_counts, instance_counts

    label_files = glob(os.path.join(test_labels_dir, "*.txt"))

    for label_file in label_files:
        classes_in_image = set()
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
                        # Count instances
                        instance_counts[class_id] = instance_counts.get(class_id, 0) + 1

            # Count images
            for class_id in classes_in_image:
                image_counts[class_id] = image_counts.get(class_id, 0) + 1

        except Exception as e:
            print(f"Warning: Failed to read {label_file}: {e}")
            continue

    return image_counts, instance_counts


def check_model_update_criteria(eval_results, per_class_comparison):
    """
    Check if new model meets the update criteria

    Args:
        eval_results: Dictionary containing old and new model evaluation results
        per_class_comparison: Dictionary containing per-class comparison statistics

    Returns:
        tuple: (should_update: bool, reason: str)
    """
    # Load config
    config_path = os.path.join(base_abspath, "config.yaml")
    if not os.path.exists(config_path):
        return False, "Config file not found"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get update configuration
    model_update_config = config.get("model_update", {})
    overall_threshold = float(model_update_config.get("overall_map_threshold", 0.03))
    criteria_class_threshold = float(model_update_config.get("criteria_class_map_threshold", 0.03))
    auto_update = model_update_config.get("auto_update", True)

    # Get criteria class
    criteria_class = config.get("yolo_model", {}).get("criteria_classes", "person")

    if not auto_update:
        return False, "Auto-update is disabled in config"

    if not eval_results or not eval_results.get('old_model') or not eval_results.get('new_model'):
        return False, "Evaluation results not available"

    # Check overall mAP50-95 improvement
    old_metrics = eval_results['old_model'].box if hasattr(eval_results['old_model'], 'box') else None
    new_metrics = eval_results['new_model'].box if hasattr(eval_results['new_model'], 'box') else None

    if not old_metrics or not new_metrics:
        return False, "Model metrics not available"

    overall_map_improvement = new_metrics.map - old_metrics.map

    print(f"\n{'='*80}")
    print("MODEL UPDATE CRITERIA CHECK")
    print(f"{'='*80}")
    print(f"Overall mAP50-95 improvement: {overall_map_improvement:+.4f} (threshold: {overall_threshold:.4f})")

    if overall_map_improvement < overall_threshold:
        reason = f"Overall mAP50-95 improvement ({overall_map_improvement:.4f}) below threshold ({overall_threshold:.4f})"
        print(f"❌ {reason}")
        print(f"{'='*80}\n")
        return False, reason

    print(f"✅ Overall mAP50-95 improvement meets threshold")

    # Check criteria class improvement
    if not per_class_comparison:
        return False, "Per-class comparison not available"

    # Find criteria class in comparison
    criteria_class_data = None
    criteria_class_idx = None

    for cls_idx, data in per_class_comparison.items():
        if data['class_name'].lower() == criteria_class.lower():
            criteria_class_data = data
            criteria_class_idx = cls_idx
            break

    if not criteria_class_data:
        reason = f"Criteria class '{criteria_class}' not found in comparison"
        print(f"❌ {reason}")
        print(f"{'='*80}\n")
        return False, reason

    criteria_class_improvement = criteria_class_data['ap_improvement']

    print(f"Criteria class '{criteria_class}' mAP50-95 improvement: {criteria_class_improvement:+.4f} (threshold: {criteria_class_threshold:.4f})")

    if criteria_class_improvement < criteria_class_threshold:
        reason = f"Criteria class '{criteria_class}' improvement ({criteria_class_improvement:.4f}) below threshold ({criteria_class_threshold:.4f})"
        print(f"❌ {reason}")
        print(f"{'='*80}\n")
        return False, reason

    print(f"✅ Criteria class improvement meets threshold")
    print(f"\n{'='*80}")
    print("🎉 ALL CRITERIA MET - Model will be updated!")
    print(f"{'='*80}\n")

    reason = f"Overall improvement: {overall_map_improvement:.4f}, Criteria class improvement: {criteria_class_improvement:.4f}"
    return True, reason


def update_production_model():
    """
    Update the production model with the newly trained model

    Steps:
    1. Find the latest trained model
    2. Backup current production model
    3. Generate versioned filename for new model
    4. Copy new model to production location with versioned name
    5. Update config.yaml to point to new versioned model

    Returns:
        bool: True if update successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("UPDATING PRODUCTION MODEL")
    print(f"{'='*80}\n")

    # Load config
    config_path = os.path.join(base_abspath, "config.yaml")
    if not os.path.exists(config_path):
        print("Error: Config file not found")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get backup setting
    backup_old_model = config.get("model_update", {}).get("backup_old_model", True)

    # Find latest trained model
    runs_base_dir = os.path.join(base_abspath, "runs")
    if not os.path.exists(runs_base_dir):
        print("Error: No training runs found")
        return False

    subdirs = [d for d in os.listdir(runs_base_dir)
               if os.path.isdir(os.path.join(runs_base_dir, d)) and d.startswith('retrain')]

    if not subdirs:
        print("Error: No retrain folders found")
        return False

    latest_dir = max([os.path.join(runs_base_dir, d) for d in subdirs], key=os.path.getmtime)

    possible_paths = [
        os.path.join(latest_dir, "weights", "best.pt"),
        os.path.join(latest_dir, "best.pt")
    ]

    new_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            new_model_path = path
            break

    if not new_model_path:
        print(f"Error: No model found in {latest_dir}")
        return False

    print(f"New trained model: {new_model_path}")

    # Get current production model path
    current_model_name = config["yolo_model"]["model_name"]
    current_model_path = os.path.join(base_abspath, current_model_name)

    print(f"Current production model: {current_model_path}")

    # Move current model to backups if it exists
    backup_dir = os.path.join(base_abspath, "model", "backups")
    os.makedirs(backup_dir, exist_ok=True)

    if backup_old_model and os.path.exists(current_model_path):
        model_basename = os.path.basename(current_model_path)

        # Check if current model is a versioned model (contains timestamp and version)
        # Format: yolov8nGender_20250107_v1.pt
        if '_v' in model_basename and len(model_basename.split('_')) >= 3:
            # It's a versioned model, move it to backups with original name
            backup_path = os.path.join(backup_dir, model_basename)
            shutil.move(current_model_path, backup_path)
            print(f"✅ Moved old versioned model to backups: {backup_path}")
        else:
            # It's a non-versioned model (e.g., yolov8nGender.pt), back it up with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{os.path.splitext(model_basename)[0]}_{timestamp}.pt")
            shutil.move(current_model_path, backup_path)
            print(f"✅ Moved old model to backups: {backup_path}")

    # Generate versioned filename for new model
    timestamp = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.dirname(current_model_path)
    current_basename = os.path.basename(current_model_path)

    # Extract base model name without version/timestamp if present
    # Handle both "yolov8nGender.pt" and "yolov8nGender_20250107_v1.pt"
    if '_v' in current_basename and len(current_basename.split('_')) >= 3:
        # Extract base name from versioned model: yolov8nGender_20250107_v1.pt -> yolov8nGender
        parts = current_basename.split('_')
        # Find the index where timestamp starts (8 digit number)
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                model_name_without_ext = '_'.join(parts[:i])
                break
        else:
            # Fallback if pattern not found
            model_name_without_ext = os.path.splitext(current_basename)[0]
    else:
        # Non-versioned model
        model_name_without_ext = os.path.splitext(current_basename)[0]

    # Find existing versioned models to determine next version number
    model_pattern = f"{model_name_without_ext}_{timestamp}_v*.pt"
    existing_versions = glob(os.path.join(model_dir, model_pattern))

    if existing_versions:
        # Extract version numbers and find the max
        version_numbers = []
        for ver_path in existing_versions:
            basename = os.path.basename(ver_path)
            # Extract version number from filename like "yolov8nGender_20250107_v1.pt"
            try:
                version_str = basename.split('_v')[-1].replace('.pt', '')
                version_numbers.append(int(version_str))
            except:
                pass

        if version_numbers:
            next_version = max(version_numbers) + 1
        else:
            next_version = 1
    else:
        next_version = 1

    # Generate new versioned model filename
    versioned_model_name = f"{model_name_without_ext}_{timestamp}_v{next_version}.pt"
    versioned_model_path = os.path.join(model_dir, versioned_model_name)

    print(f"New versioned model name: {versioned_model_name}")

    # Copy new model to production location with versioned name
    try:
        shutil.copy2(new_model_path, versioned_model_path)
        print(f"✅ Copied new model to: {versioned_model_path}")
    except Exception as e:
        print(f"❌ Error copying model: {e}")
        return False

    # Update config.yaml to point to new versioned model and record timestamp
    try:
        # Update the model_name in config
        relative_model_path = os.path.join(".", "model", versioned_model_name)
        config["yolo_model"]["model_name"] = relative_model_path

        # Record the update timestamp in ISO 8601 format
        update_timestamp = datetime.now().isoformat()
        config["yolo_model"]["last_model_update"] = update_timestamp

        # Write updated config back to file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"✅ Updated config.yaml with new model path: {relative_model_path}")
        print(f"✅ Recorded model update timestamp: {update_timestamp}")
    except Exception as e:
        print(f"❌ Error updating config.yaml: {e}")
        return False

    print(f"\n{'='*80}")
    print("✅ PRODUCTION MODEL UPDATE COMPLETED!")
    print(f"{'='*80}")
    print(f"Production model updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: {versioned_model_path}")
    print(f"Config updated to: {relative_model_path}")
    print(f"Update timestamp: {update_timestamp}")
    print(f"{'='*80}\n")

    return True


def compare_per_class_performance(old_results, new_results, class_names=None):
    """
    클래스별 성능 비교 (Old vs New 모델)

    Args:
        old_results: Old model evaluation results
        new_results: New model evaluation results
        class_names: List of class names (optional, uses global class_names_list if not provided)

    Returns:
        dict: Per-class comparison statistics
    """
    if not old_results or not new_results:
        print("Error: Both old and new results are required for comparison")
        return None

    if not hasattr(old_results, 'box') or not hasattr(new_results, 'box'):
        print("Error: Results do not contain box metrics")
        return None

    old_metrics = old_results.box
    new_metrics = new_results.box

    # Use provided class names or fall back to global
    if class_names is None:
        class_names = class_names_list

    print(f"\n{'='*100}")
    print("PER-CLASS PERFORMANCE COMPARISON (Old vs New Model)")
    print(f"{'='*100}\n")

    # Check if per-class metrics are available
    if not hasattr(old_metrics, 'ap_class_index') or not hasattr(new_metrics, 'ap_class_index'):
        print("Warning: Per-class metrics not available in results")
        return None

    # Get class indices and AP values
    old_class_indices = old_metrics.ap_class_index
    old_ap50 = old_metrics.ap50
    old_ap = old_metrics.ap

    new_class_indices = new_metrics.ap_class_index
    new_ap50 = new_metrics.ap50
    new_ap = new_metrics.ap

    # Count images and instances from test dataset labels
    test_labels_dir = os.path.join(DATASETS_BASE_PATH, "splitted", "test", "labels")
    image_counts, instance_counts = count_images_and_instances_from_labels(test_labels_dir)

    print(f"Loaded from test set: {len(image_counts)} classes with data")
    print(f"  Total images in test set: {len(glob(os.path.join(test_labels_dir, '*.txt')))}")
    print(f"  Total instances: {sum(instance_counts.values())}")

    # Create comparison data structure
    comparison = {}

    # Collect all unique class indices
    all_class_indices = set(list(old_class_indices) + list(new_class_indices))

    for cls_idx in sorted(all_class_indices):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"

        # Find metrics for this class in old model
        old_idx = None
        if cls_idx in old_class_indices:
            old_idx = list(old_class_indices).index(cls_idx)

        # Find metrics for this class in new model
        new_idx = None
        if cls_idx in new_class_indices:
            new_idx = list(new_class_indices).index(cls_idx)

        # Get AP values
        old_ap50_val = float(old_ap50[old_idx]) if old_idx is not None else 0.0
        old_ap_val = float(old_ap[old_idx]) if old_idx is not None else 0.0

        new_ap50_val = float(new_ap50[new_idx]) if new_idx is not None else 0.0
        new_ap_val = float(new_ap[new_idx]) if new_idx is not None else 0.0

        # Get image and instance counts from test set
        num_images = image_counts.get(cls_idx, 0)
        num_instances = instance_counts.get(cls_idx, 0)

        # Calculate improvements
        ap50_improvement = new_ap50_val - old_ap50_val
        ap_improvement = new_ap_val - old_ap_val

        ap50_improvement_pct = (ap50_improvement / old_ap50_val * 100) if old_ap50_val > 0 else 0
        ap_improvement_pct = (ap_improvement / old_ap_val * 100) if old_ap_val > 0 else 0

        comparison[cls_idx] = {
            'class_name': cls_name,
            'old_ap50': old_ap50_val,
            'new_ap50': new_ap50_val,
            'ap50_improvement': ap50_improvement,
            'ap50_improvement_pct': ap50_improvement_pct,
            'old_ap': old_ap_val,
            'new_ap': new_ap_val,
            'ap_improvement': ap_improvement,
            'ap_improvement_pct': ap_improvement_pct,
            'images': num_images,
            'instances': num_instances
        }

    # Print detailed comparison table with Images and Instances columns
    print(f"\n{'Class':<15} {'Images':<8} {'Instances':<10} {'Metric':<12} {'Old Model':<12} {'New Model':<12} {'Δ Absolute':<12} {'Δ %':<10}")
    print(f"{'-'*105}")

    for cls_idx in sorted(comparison.keys()):
        data = comparison[cls_idx]
        cls_name = data['class_name']
        num_images = data['images']
        num_instances = data['instances']

        # mAP50
        print(f"{cls_name:<15} {num_images:<8} {num_instances:<10} {'mAP50':<12} {data['old_ap50']:<12.4f} {data['new_ap50']:<12.4f} "
              f"{data['ap50_improvement']:<+12.4f} {data['ap50_improvement_pct']:+10.2f}%")

        # mAP50-95
        print(f"{'':<15} {'':<8} {'':<10} {'mAP50-95':<12} {data['old_ap']:<12.4f} {data['new_ap']:<12.4f} "
              f"{data['ap_improvement']:<+12.4f} {data['ap_improvement_pct']:+10.2f}%")

        print()  # Empty line between classes

    print(f"{'='*90}\n")

    # Print summary statistics
    print("SUMMARY STATISTICS:")
    print(f"{'-'*80}")

    # Classes with improvement
    improved_classes = [cls_idx for cls_idx, data in comparison.items()
                       if data['ap50_improvement'] > 0]
    degraded_classes = [cls_idx for cls_idx, data in comparison.items()
                       if data['ap50_improvement'] < 0]
    unchanged_classes = [cls_idx for cls_idx, data in comparison.items()
                        if data['ap50_improvement'] == 0]

    print(f"Classes improved:   {len(improved_classes)}/{len(comparison)} "
          f"({len(improved_classes)/len(comparison)*100:.1f}%)")
    print(f"Classes degraded:   {len(degraded_classes)}/{len(comparison)} "
          f"({len(degraded_classes)/len(comparison)*100:.1f}%)")
    print(f"Classes unchanged:  {len(unchanged_classes)}/{len(comparison)} "
          f"({len(unchanged_classes)/len(comparison)*100:.1f}%)")

    # Average improvement
    avg_ap50_improvement = sum(data['ap50_improvement'] for data in comparison.values()) / len(comparison)
    avg_ap_improvement = sum(data['ap_improvement'] for data in comparison.values()) / len(comparison)

    print(f"\nAverage mAP50 improvement:    {avg_ap50_improvement:+.4f}")
    print(f"Average mAP50-95 improvement: {avg_ap_improvement:+.4f}")

    # Best and worst improvements
    best_improved = max(comparison.items(), key=lambda x: x[1]['ap50_improvement'])
    worst_degraded = min(comparison.items(), key=lambda x: x[1]['ap50_improvement'])

    print(f"\nBest improved class:  {best_improved[1]['class_name']} "
          f"(+{best_improved[1]['ap50_improvement']:.4f}, {best_improved[1]['ap50_improvement_pct']:+.2f}%)")
    print(f"Worst degraded class: {worst_degraded[1]['class_name']} "
          f"({worst_degraded[1]['ap50_improvement']:.4f}, {worst_degraded[1]['ap50_improvement_pct']:+.2f}%)")

    print(f"{'='*80}\n")

    return comparison


def main():
    """메인 엔트리 포인트"""
    try:
        print(f"\n=== YOLO Model Retraining Process Started ===")

        # Step 1: Train model
        train_model()

        # Step 2: Evaluate and compare overall metrics
        print(f"\n=== Evaluating Old and New Models ===")
        eval_results = evaluate_old_and_new_models()

        # Step 3: Compare per-class performance
        per_class_comparison = None
        if eval_results and eval_results.get('old_model') and eval_results.get('new_model'):
            print(f"\n=== Per-Class Performance Comparison ===")
            per_class_comparison = compare_per_class_performance(
                old_results=eval_results['old_model'],
                new_results=eval_results['new_model']
            )
        else:
            print("\nSkipping per-class comparison - evaluation results not available")

        # Step 4: Check model update criteria and update if met
        if eval_results and per_class_comparison:
            should_update, reason = check_model_update_criteria(eval_results, per_class_comparison)

            if should_update:
                print(f"\n=== Updating Production Model ===")
                print(f"Reason: {reason}")
                update_success = update_production_model()

                if update_success:
                    print("\n✅ Production model has been successfully updated!")
                    print("   The new model will be used in production (yolo_producer_fastapi.py)")
                    print("   NOTE: You may need to restart the yolo_producer_fastapi service for changes to take effect.")
                else:
                    print("\n❌ Failed to update production model")
            else:
                print(f"\n=== Model Update Skipped ===")
                print(f"Reason: {reason}")
                print("The current production model will continue to be used.")

        print(f"\n=== YOLO Model Retraining Finished ===\n")

    except Exception as e:
        import traceback
        print("[ERROR] Model retraining failed!")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
