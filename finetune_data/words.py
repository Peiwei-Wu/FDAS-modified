"""WORD数据集：使用官方100/20/30病例划分。"""

import json
from pathlib import Path

from .btcv import strip_nii_suffix, write_case_lists


def load_words_split(data_root, output_dir, seed):
    """读取WORD训练、验证和测试集的固定划分。"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    split_dirs = {
        "train": ("imagesTr", "labelsTr", 100),
        "valid": ("imagesVal", "labelsVal", 20),
        "test": ("imagesTs", "labelsTs", 30),
    }

    split_info = {"dataset": "WORD", "seed": seed}
    for split, (image_dir, label_dir, expected_count) in split_dirs.items():
        image_paths = sorted((data_root / image_dir).glob("*.nii.gz"))
        label_names = {path.name for path in (data_root / label_dir).glob("*.nii.gz")}
        if len(image_paths) != expected_count or {path.name for path in image_paths} != label_names:
            raise ValueError(f"WORD {split}图像和标签数量或名称不对应")

        split_info[split] = [
            {
                "case": strip_nii_suffix(image_path),
                "image": str(Path(image_dir) / image_path.name),
                "label": str(Path(label_dir) / image_path.name),
            }
            for image_path in image_paths
        ]

    with open(data_root / "dataset.json", "r", encoding="utf-8") as file:
        split_info["labels"] = json.load(file)["labels"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "splits.json", "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2, ensure_ascii=False)
    write_case_lists(split_info, output_dir)
    print("Saved WORD split: train=100, valid=20, test=30")
    return split_info
