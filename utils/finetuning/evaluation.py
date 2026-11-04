"""原始三维网格上的逐器官Dice/ASSD与CSV、NIfTI输出。"""

import logging
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm

from .inference import predict_case
from .postprocessing import apply_postprocessing


def calculate_metrics(prediction, label, num_classes, spacing, with_assd):
    dice, assd = [], []
    for c in range(1, num_classes):
        p, t = prediction == c, label == c
        denominator = p.sum() + t.sum()
        dice.append(200.0 * np.count_nonzero(p & t) / denominator if denominator else np.nan)
        if with_assd:
            assd.append(metric.binary.assd(p, t, voxelspacing=spacing) if p.any() and t.any() else np.nan)
    return dice, assd


def make_summary(case_log, class_names, with_assd):
    rows = []
    for name in class_names[1:]:
        dice = case_log[f"dice_{name}"].dropna().to_numpy()
        row = {"class": name, "dice_mean": float(dice.mean()) if dice.size else np.nan,
               "dice_std": float(dice.std()) if dice.size else np.nan}
        if with_assd:
            assd = case_log[f"assd_{name}"].dropna().to_numpy()
            row.update(assd_mean_mm=float(assd.mean()) if assd.size else np.nan,
                       assd_std_mm=float(assd.std()) if assd.size else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(model, data_loader, records, data_root, class_names, device,
             with_assd=False, prediction_dir=None, inference=None, rules=None,
             cache=None, cached=None, fixed_postprocess=None):
    model.eval()
    settings, rows = inference or {}, []
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(tqdm(records, desc="3D evaluation")):
        if cached is None:
            reference = sitk.ReadImage(str(data_root / item["label"]))
            label = sitk.GetArrayFromImage(reference)
            if data_loader.dataset.label_map is not None:
                label = data_loader.dataset.label_map[label]
            prediction = predict_case(model, data_loader.dataset, item["case"], device,
                                      settings.get("step_size", 0.5), settings.get("mirror_tta", False),
                                      z_smooth_sigma=float(settings.get("z_smooth_sigma", 0.0)))
        else:
            _, prediction, label, reference = cached[index]
        if cache is not None:
            cache.append((item["case"], prediction, label, reference))
        if fixed_postprocess: # 无验证集则采用不同数据集固定的后处理策略
            prediction = fixed_postprocess(prediction, reference.GetDirection())
        elif rules: # 基于验证集选择的最优规则做后处理
            prediction = apply_postprocessing(prediction, rules, direction=reference.GetDirection())
        dice, assd = calculate_metrics(prediction, label, len(class_names), tuple(reversed(reference.GetSpacing())), with_assd)
        row = {"case": item["case"], **{f"dice_{name}": score for name, score in zip(class_names[1:], dice)}}
        row.update({f"assd_{name}": score for name, score in zip(class_names[1:], assd)})
        rows.append(row)
        if prediction_dir:
            image = sitk.GetImageFromArray(prediction)
            image.CopyInformation(reference)
            sitk.WriteImage(image, str(prediction_dir / f"{item['case']}.nii.gz"))
    case_log = pd.DataFrame(rows)
    values = case_log[[f"dice_{name}" for name in class_names[1:]]].to_numpy(dtype=float)
    mean = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    return mean, case_log, make_summary(case_log, class_names, with_assd)


def save_evaluation(name, model, loader, records, data_root, class_names, device, result_dir,
                    save_result, reuse_raw=False, **kwargs):
    csv_dir = result_dir / "csv"
    # 没有后处理时，直接复用raw结果。
    if ((reuse_raw or kwargs.get("cached") is not None) and not kwargs.get("rules")
            and not kwargs.get("fixed_postprocess")):
        for suffix in ("cases", "summary"):
            shutil.copyfile(csv_dir / f"{name}_raw_{suffix}.csv", csv_dir / f"{name}_{suffix}.csv")
        if save_result:
            shutil.copytree(result_dir / "nii" / f"{name}_raw", result_dir / "nii" / name, dirs_exist_ok=True)
        cases = pd.read_csv(csv_dir / f"{name}_cases.csv")
        values = cases[[f"dice_{c}" for c in class_names[1:]]].to_numpy()
        mean = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        logging.info("%s mean Dice: %.4f (no postprocessing; reused raw results)", name.capitalize(), mean)
        return mean
    prediction_dir = result_dir / "nii" / name if save_result else None
    mean, cases, summary = evaluate(model, loader, records, data_root, class_names, device,
                                    with_assd=True, prediction_dir=prediction_dir, **kwargs)
    csv_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(csv_dir / f"{name}_cases.csv", index=False)
    summary.to_csv(csv_dir / f"{name}_summary.csv", index=False)
    logging.info("%s mean Dice: %.4f", name.capitalize(), mean)
    return mean
