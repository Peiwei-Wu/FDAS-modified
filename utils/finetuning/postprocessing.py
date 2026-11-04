"""统一管理验证集规则和各数据集的固定后处理。"""

import numpy as np
from scipy import ndimage


def apply_postprocessing(prediction, rules, kidney_labels=None, direction=None):
    result = prediction.copy()

    # 针对BTCV，划分左右肾
    if kidney_labels:
        # 完整三维图像按左右半幅重分肾标签；SimpleITK物理X正方向为患者左侧。
        right_label, left_label = kidney_labels
        direction = np.asarray(direction if direction is not None else np.eye(3)).reshape(3, 3)
        axis = int(np.abs(direction[0]).argmax())
        array_axis = 2 - axis  # ITK的XYZ对应numpy的ZYX。
        shape = [1, 1, 1]
        shape[array_axis] = result.shape[array_axis]
        left_side = np.arange(shape[array_axis]) >= shape[array_axis] / 2
        if direction[0, axis] < 0:
            left_side = ~left_side
        kidneys = np.isin(result, kidney_labels)
        labels = np.where(left_side.reshape(shape), left_label, right_label)
        result[kidneys] = np.broadcast_to(labels, result.shape)[kidneys]

    # 不同数据集，不同器官进行三维连通域处理
    for classes in rules:
        mask = np.isin(result, classes)
        components, count = ndimage.label(mask)  # 3D六邻域，和nnU-Net默认一致。
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            result[mask & (components != sizes.argmax())] = 0

    return result


def btcv_postprocessing(prediction, direction=None):
    """BTCV和BTCV8：纠正左右肾，再过滤脾、肝的最大三维连通域。"""
    return apply_postprocessing(prediction, [[1], [6]], [2, 3], direction)


def get_fixed_postprocessing(dataset):
    if dataset in ("BTCV", "BTCV8"):
        return btcv_postprocessing
    return None


def select_postprocessing(cases, num_classes):
    """cases来自最佳权重的验证预测；绝不通过测试标签选择规则。"""
    predictions = [item[1] for item in cases]
    labels = [item[2] for item in cases]
    def mean_dice(volumes):
        scores = np.full((len(volumes), num_classes-1), np.nan)
        for i, (prediction, label) in enumerate(zip(volumes, labels)):
            for c in range(1, num_classes):
                p, t = prediction == c, label == c
                denominator = p.sum() + t.sum()
                if denominator:
                    scores[i, c-1] = 2 * np.count_nonzero(p & t) / denominator
        return np.array([column[np.isfinite(column)].mean() if np.isfinite(column).any() else np.nan
                         for column in scores.T])
    baseline = mean_dice(predictions)
    rules, report = [], []
    groups = [list(range(1, num_classes))] + [[c] for c in range(1, num_classes)]
    for classes in groups:
        candidate = [apply_postprocessing(p, [classes]) for p in predictions]
        scores = mean_dice(candidate)
        if len(classes) > 1:
            accepted = np.nanmean(scores) > np.nanmean(baseline) and not np.any(scores < baseline)
        else:
            accepted = scores[classes[0]-1] > baseline[classes[0]-1]
        report.append(dict(classes=classes, accepted=bool(accepted),
                           mean_before=float(np.nanmean(baseline)), mean_after=float(np.nanmean(scores))))
        if accepted:
            rules.append(classes)
            predictions, baseline = candidate, scores
    return {"rules": rules, "validation_cases": [item[0] for item in cases], "comparisons": report}
