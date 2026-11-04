"""各器官独立的前景Batch Dice + 全类别CE。"""

import torch.nn.functional as F


def dice_ce_loss(logits, labels):
    # ground truth
    target = labels[:, 0].long()
    if target.ndim == 4:  # 兼容体数据标签：[B,D,H,W]展开为[B*D,H,W]，与逐切片预测对应
        target = target.flatten(0, 1)
    if target.shape[-2:] != logits.shape[-2:]: # 标签与当前预测尺寸不一致时，用最近邻缩放标签
        target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
    # prediction
    probabilities = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    probabilities, onehot = probabilities[:, 1:], onehot[:, 1:]
    axes = (0, 2, 3)  # 合并batch和空间维，保留器官类别维。
    intersection = (probabilities * onehot).sum(axes)
    dice = (2 * intersection + 1e-5) / (probabilities.sum(axes) + onehot.sum(axes) + 1e-5)

    w_dice, w_ce = 1.4, 0.6
    total_loss = w_dice * (1 - dice.mean()) + w_ce * F.cross_entropy(logits, target)

    return total_loss
