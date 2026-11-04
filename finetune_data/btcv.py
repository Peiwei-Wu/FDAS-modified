import json
import random
from pathlib import Path

from .btcv_transunet import load_transunet_split


def write_case_lists(split_info, output_dir):
    """保存各集合病例名称。"""
    for split in ("train", "valid", "test"):
        with open(output_dir / f"{split}_cases.txt", "w", encoding="utf-8") as file:
            for item in split_info[split]:
                file.write(f"{item['case']}\t{item['image']}\t{item['label']}\n")


def strip_nii_suffix(path):
    """提取病例名称。"""
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def load_btcv_split(data_root, output_dir, seed, btcv_split="original"):
    """复用当前划分；切换20/4/6和18/12时按原始病例信息重新生成。"""
    split_path = output_dir / "splits.json"
    protocol = "transunet_18_12" if btcv_split == "transunet" else "original"

    # 无protocol字段的已有划分默认为原20/4/6。
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as file:
            split_info = json.load(file)
        if split_info.get("protocol", "original") == protocol:
            write_case_lists(split_info, output_dir)
            print(f"Reuse split: {split_path}")
            return split_info # 保存的是数据划分信息不是图像内容

    # 如果没有则先创建病例划分
    dataset_json = data_root / "dataset_0.json"
    if not dataset_json.exists():
        dataset_json = data_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"BTCV配置文件不存在: {dataset_json}")

    with open(dataset_json, "r", encoding="utf-8") as file:
        dataset_info = json.load(file) # 下载时候自带的json表格

    def make_record(item):
        return {
            "case": strip_nii_suffix(item["image"]),
            "image": item["image"],
            "label": item["label"],
        }

    candidates = [make_record(item) for item in dataset_info.get("training", [])] # 训练验证集
    test_cases = [make_record(item) for item in dataset_info.get("validation", [])] # 测试集
    if len(candidates) != 24 or len(test_cases) != 6:
        raise ValueError(f"BTCV应为24个候选病例和6个测试病例，实际为{len(candidates)}和{len(test_cases)}")

    random.Random(seed).shuffle(candidates)
    valid_cases = candidates[:4]
    train_cases = candidates[4:]
    split_info = {
        "dataset": "BTCV",
        "protocol": "original",
        "seed": seed,
        "labels": dataset_info["labels"],
        "train": train_cases,
        "valid": valid_cases,
        "test": test_cases,
    }
    if btcv_split == "transunet":
        split_info = load_transunet_split(split_info)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2, ensure_ascii=False)
    write_case_lists(split_info, output_dir)
    print(f"Saved split: {split_path}")

    return split_info
