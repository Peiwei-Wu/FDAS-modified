"""BTCV训练切片统计与小器官重采样。"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def count_slice_labels(data_dir, files, num_classes):
    """统计每张切片各类别的像素数。"""
    counts = []
    for name in files:
        with h5py.File(data_dir / name, "r") as file:
            counts.append(np.bincount(file["masks"][:].ravel(), minlength=num_classes))
    return np.asarray(counts)


def build_btcv_sampler(train_set, class_ids, weight, seed, report_dir=None):
    """提高指定器官切片的抽样权重，保持背景的预期占比。"""
    output_dir = train_set.data_dir.parent
    with open(output_dir / "splits.json", "r", encoding="utf-8") as file:
        labels = json.load(file)["labels"]
    num_classes = len(labels)
    if not class_ids or min(class_ids) < 1 or max(class_ids) >= num_classes or not np.isfinite(weight) or weight < 1:
        raise ValueError("resample_classes应为前景类别编号，resample_weight应为不小于1的有限数值")
    with open(output_dir / "train_slices.list", "r", encoding="utf-8") as file:
        all_files = [line.strip() for line in file if line.strip()]

    counts = count_slice_labels(train_set.data_dir, all_files, num_classes)
    presence = counts > 0
    presence[:, 0] = ~presence[:, 1:].any(axis=1)  # 第0列统计纯背景切片
    file_indices = {name: index for index, name in enumerate(all_files)}
    selected = [file_indices[name] for name in train_set.files]
    active_presence = presence[selected]  # 顺序必须与背景过滤后的Dataset一致

    weights = np.ones(len(train_set), dtype=np.float64)
    boosted = active_presence[:, class_ids].any(axis=1)
    weights[boosted] = weight  # 多个目标器官共存时不叠加权重
    foreground = ~active_presence[:, 0]
    if foreground.any():
        weights[foreground] *= foreground.sum() / weights[foreground].sum()
    probabilities = weights / weights.sum()

    # 保持每轮抽样次数；有放回抽样允许重点切片重复出现，各轮会重新抽样。
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_set),
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )

    pixel_totals = counts.sum(axis=0)
    rows = []
    for class_id in range(num_classes):
        rows.append({
            "class_id": class_id,
            "class_name": labels[str(class_id)] if class_id else "pure_background",
            "all_train_slices": len(all_files),
            "original_count": int(presence[:, class_id].sum()),
            "original_percent": 100 * presence[:, class_id].mean(),
            "active_train_slices": len(train_set),
            "active_count": int(active_presence[:, class_id].sum()),
            "active_percent": 100 * active_presence[:, class_id].mean(),
            "expected_sample_percent": 100 * (probabilities @ active_presence[:, class_id]),
            "foreground_pixel_percent": 100 * pixel_totals[class_id] / max(pixel_totals[1:].sum(), 1) if class_id else np.nan,
        })
    report_dir = Path(report_dir) if report_dir is not None else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "train_organ_sampling.csv", index=False)
    pd.DataFrame({
        "file": train_set.files, "boosted": boosted, "weight": weights,
        "sample_probability": probabilities,
    }).to_csv(report_dir / "train_sampling_weights.csv", index=False)
    logging.info(
        "BTCV resampling: classes=%s, relative_weight=%.2f, boosted=%d/%d, expected_background=%.2f%%, draws_per_epoch=%d",
        class_ids, weight, boosted.sum(), len(train_set),
        100 * probabilities[~foreground].sum(), len(train_set),
    )
    logging.info("Sampling reports: %s", report_dir)
    return sampler
