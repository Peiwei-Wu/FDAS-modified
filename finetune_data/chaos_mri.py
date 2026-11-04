"""CHAOS MRI：四器官分割，按患者划分12/4/4，包含T1两相和T2SPIR。"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

from .btcv import write_case_lists


def convert_mri_series(dicom_dir, ground_dir, image_path, label_path):
    """将单个MRI序列与四器官标签配对保存为NIfTI。"""
    if image_path.exists() and label_path.exists():
        return

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if len(series_ids) != 1:
        raise ValueError(f"MRI目录应包含一个序列: {dicom_dir}")
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(dicom_files)  # 按空间位置排序
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()

    mask_paths = {int(path.stem.split("-")[-1]): path for path in ground_dir.glob("*.png")}
    if len(mask_paths) != image.GetDepth():
        raise ValueError(f"MRI图像与标签切片数不一致: {dicom_dir}")
    masks = []
    for slice_idx in range(image.GetDepth()):
        mask_idx = int(reader.GetMetaData(slice_idx, "0020|0013"))
        if dicom_dir.name == "OutPhase":
            mask_idx += 1  # T1反相为奇数编号，共用相邻偶数编号的同相标签
        mask_path = mask_paths[mask_idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"标签读取失败: {mask_path}")
        label_slice = np.zeros(mask.shape, dtype=np.uint8)
        # 官方灰度范围：肝脏、右肾、左肾、脾脏，背景为0。
        for class_id, (low, high) in enumerate(((55, 70), (110, 135), (175, 200), (240, 255)), 1):
            label_slice[(mask > low) & (mask <= high)] = class_id
        masks.append(label_slice)

    label = sitk.GetImageFromArray(np.stack(masks))
    label.CopyInformation(image)
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))
    print(f"{image_path.stem}: {image.GetDepth()} slices")


def load_chaos_mri_split(data_root, output_dir, seed):
    """转换MRI序列，创建或读取患者级12/4/4划分。"""
    data_root, output_dir = Path(data_root), Path(output_dir).resolve()
    mr_root = data_root / "Train_Sets" / "MR"
    if not mr_root.is_dir():
        mr_root = data_root  # 支持直接传入Train_Sets/MR
    case_dirs = sorted(
        [path for path in mr_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if len(case_dirs) != 20:
        raise ValueError(f"CHAOS MRI应有20个患者，实际为{len(case_dirs)}")

    nifti_dir = output_dir / "nifti"
    (nifti_dir / "images").mkdir(parents=True, exist_ok=True)
    (nifti_dir / "labels").mkdir(parents=True, exist_ok=True)
    sequences = (("T1DUAL", "InPhase"), ("T1DUAL", "OutPhase"), ("T2SPIR", ""))
    patient_records = {}
    for case_dir in case_dirs:
        records = []
        for sequence, phase in sequences:
            sequence_name = f"{sequence}_{phase}" if phase else sequence
            case_name = f"chaos_mri_{case_dir.name}_{sequence_name}"
            image_path = nifti_dir / "images" / f"{case_name}.nii.gz"
            label_path = nifti_dir / "labels" / f"{case_name}.nii.gz"
            convert_mri_series(
                case_dir / sequence / "DICOM_anon" / phase,
                case_dir / sequence / "Ground", image_path, label_path,
            )
            records.append({
                "case": case_name,
                "patient": case_dir.name,
                "sequence": sequence_name,
                "image": str(image_path),
                "label": str(label_path),
            })
        patient_records[case_dir.name] = records

    split_path = output_dir / "splits.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as file:
            split_info = json.load(file)
        record_map = {item["case"]: item for records in patient_records.values() for item in records}
        for split in ("train", "valid", "test"):
            split_info[split] = [record_map[item["case"]] for item in split_info[split]]
        print(f"Reuse split: {split_path}")
    else:
        patients = list(patient_records)
        random.Random(seed).shuffle(patients)
        patient_splits = {"train": patients[:12], "valid": patients[12:16], "test": patients[16:]}
        split_info = {
            "dataset": "CHAOS_MRI",
            "seed": seed,
            "labels": {"0": "background", "1": "liver", "2": "rkid", "3": "lkid", "4": "spleen"},
            "patient_splits": patient_splits,
        }
        # 先划分患者，再展开序列，保证同一患者的三套图像不会跨集合。
        for split, patient_ids in patient_splits.items():
            split_info[split] = [item for patient in patient_ids for item in patient_records[patient]]
        print("Saved split: patients=12/4/4, series=36/12/12")

    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2, ensure_ascii=False)
    write_case_lists(split_info, output_dir)
    return split_info
