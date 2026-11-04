"""二维滑窗融合logits并恢复原始三维网格。"""

import json
from functools import lru_cache
from itertools import product
from tempfile import TemporaryFile

import h5py
import numpy as np
from scipy import ndimage
from skimage.transform import resize
import torch

# 兼容旧代码的导入路径；具体实现放在postprocessing.py。
from .postprocessing import apply_postprocessing, select_postprocessing


@lru_cache(maxsize=8)
def gaussian_weights(patch_size):
    impulse = np.zeros(patch_size, dtype=np.float32)
    impulse[tuple(size // 2 for size in patch_size)] = 1
    weights = ndimage.gaussian_filter(impulse, sigma=np.asarray(patch_size) / 8, mode="constant")
    weights /= weights.max()
    return np.maximum(weights, 1e-7)


@torch.no_grad()
def sliding_window_logits(model, image, patch_size, device, step_size=0.5, mirror_tta=False):
    """窗口间距不超过patch的一半，重叠处对原始logits加权平均。"""
    h, w = image.shape
    ph, pw = patch_size
    dh, dw = max(0, ph-h), max(0, pw-w)
    image = np.pad(image, ((dh//2, dh-dh//2), (dw//2, dw-dw//2)))
    positions = [np.round(np.linspace(0, n-p, int(np.ceil((n-p)/(p*step_size)))+1)).astype(int)
                 for n, p in zip(image.shape, patch_size)]
    weights = torch.as_tensor(gaussian_weights(tuple(patch_size)), device=device)
    accumulated, normalization = None, torch.zeros(image.shape, device=device)
    flips = ((), (2,), (3,), (2, 3)) if mirror_tta else ((),)
    for y, x in product(*positions):
        patch = torch.from_numpy(image[y:y+ph, x:x+pw].copy())[None, None].to(device)
        logits = None
        for axes in flips:
            prediction = model(torch.flip(patch, axes) if axes else patch, return_logits=True).float()
            prediction = torch.flip(prediction, axes) if axes else prediction
            logits = prediction if logits is None else logits + prediction
        logits = logits[0] / len(flips)
        if accumulated is None:
            accumulated = torch.zeros((logits.shape[0], *image.shape), device=device)
        accumulated[:, y:y+ph, x:x+pw] += logits * weights
        normalization[y:y+ph, x:x+pw] += weights
    accumulated /= normalization
    return accumulated[:, dh//2:dh//2+h, dw//2:dw//2+w].cpu().numpy()


def predict_case(model, dataset, case, device, step_size=0.5, mirror_tta=False, z_smooth_sigma=0.0):
    """Restore slice logits, optionally smooth along Z, then take argmax."""
    metadata = json.loads((dataset.data_dir.parent / "cases" / f"{case}.json").read_text())
    prediction = np.zeros(metadata["original_shape"], dtype=np.uint8)
    y0, y1, x0, x1 = metadata["bbox"]
    with TemporaryFile() as temporary, h5py.File(temporary, "w") as volume:
        for z in range(prediction.shape[0]):
            with h5py.File(dataset.data_dir / f"{case}_slice{z}.h5", "r") as file:
                image = file["image"][:]
            logits = sliding_window_logits(model, image, tuple(dataset.patch_size), device, step_size, mirror_tta)
            # 等价于Plans的is_seg=False/order=1：先插值各类logits，再argmax。
            restored = np.stack([resize(channel, (y1-y0, x1-x0), order=1, mode="edge",
                                        anti_aliasing=False, preserve_range=True) for channel in logits])
            if z_smooth_sigma > 0:
                if z == 0:
                    scores = volume.create_dataset("logits", shape=(prediction.shape[0], *restored.shape), dtype="float32")
                scores[z] = restored
            else:
                prediction[z, y0:y1, x0:x1] = restored.argmax(0).astype(np.uint8)
        if z_smooth_sigma > 0:
            # Bound RAM by filtering thin XY bands of the disk-backed Z/C/Y/X logits.
            for y in range(0, y1-y0, 16):
                block = scores[:, :, y:y+16, :]
                ndimage.gaussian_filter1d(block, z_smooth_sigma, axis=0, output=block, mode="reflect")
                prediction[:, y0+y:y0+y+block.shape[2], x0:x1] = block.argmax(1).astype(np.uint8)
    return prediction

