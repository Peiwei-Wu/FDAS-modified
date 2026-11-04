"""CHAOS CT：肝脏分割，按病例固定划分12/4/4，暂不做五折。"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

from .btcv import write_case_lists


def convert_ct_case(case_dir, image_path, label_path):
    """将DICOM和PNG标签配对保存为三维NIfTI。"""
    if image_path.exists() and label_path.exists():
        return

    dicom_dir = str(case_dir / "DICOM_anon")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if len(series_ids) != 1:
        raise ValueError(f"{case_dir.name}应包含一个CT序列，实际为{len(series_ids)}")
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(dicom_files)  # 按空间位置排列，不按文件名排列
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()  # 保留原始尺寸、HU值和物理间距

    masks = []
    for slice_idx in range(image.GetDepth()):
        # InstanceNumber从1开始，PNG编号从0开始；随DICOM顺序配对。
        mask_idx = int(reader.GetMetaData(slice_idx, "0020|0013")) - 1
        mask_path = case_dir / "Ground" / f"liver_GT_{mask_idx:03d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"标签读取失败: {mask_path}")
        masks.append((mask > 0).astype(np.uint8))

    label = sitk.GetImageFromArray(np.stack(masks))
    label.CopyInformation(image)
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))
    print(f"CHAOS {case_dir.name}: {image.GetDepth()} slices")


def load_chaos_split(data_root, output_dir, seed):
    """转换20个CT病例，创建或读取12/4/4划分。"""
    data_root, output_dir = Path(data_root), Path(output_dir).resolve()
    ct_root = data_root / "Train_Sets" / "CT"
    if not ct_root.is_dir():
        ct_root = data_root  # 也支持直接传入Train_Sets/CT
    case_dirs = sorted(
        [path for path in ct_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if len(case_dirs) != 20:
        raise ValueError(f"CHAOS应有20个CT病例，实际为{len(case_dirs)}: {ct_root}")

    # 原始DICOM/PNG不改动，转换结果单独保存，供现有H5及评估代码读取。
    nifti_dir = output_dir / "nifti"
    (nifti_dir / "images").mkdir(parents=True, exist_ok=True)
    (nifti_dir / "labels").mkdir(parents=True, exist_ok=True)
    records = []
    for case_dir in case_dirs:
        case_name = f"chaos_{case_dir.name}"
        image_path = nifti_dir / "images" / f"{case_name}.nii.gz"
        label_path = nifti_dir / "labels" / f"{case_name}.nii.gz"
        convert_ct_case(case_dir, image_path, label_path)
        records.append({
            "case": case_name,
            # 绝对路径兼容现有data_root / item["image"]的读取方式。
            "image": str(image_path),
            "label": str(label_path),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "splits.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as file:
            split_info = json.load(file)
        # 保留已有病例分组，更新转换文件的保存路径。
        record_map = {item["case"]: item for item in records}
        for split in ("train", "valid", "test"):
            split_info[split] = [record_map[item["case"]] for item in split_info[split]]
        print(f"Reuse split: {split_path}")
    else:
        random.Random(seed).shuffle(records)
        split_info = {
            "dataset": "CHAOS",
            "seed": seed,
            "labels": {"0": "background", "1": "liver"},
            "train": records[:12],
            "valid": records[12:16],
            "test": records[16:],
        }
        print("Saved split: train=12, valid=4, test=4")

    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2, ensure_ascii=False)
    write_case_lists(split_info, output_dir)
    return split_info
