"""每个batch约33%按病例、器官、切片定向采样，其余随机抽取切片。"""

import math

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler

from utils.dataloader import FineTuned_Dataset
from .augmentation import training_transforms


class PatchDataset(FineTuned_Dataset):
    def __init__(self, output_dir, split, patch_size, background_ratio=None, seed=1337, label_map=None):
        # 直接复用原来的背景过滤函数，过滤口径不变。
        super().__init__(output_dir, split, background_ratio=background_ratio, seed=seed, label_map=label_map)
        self.training = split == "train"
        self.patch_size = np.asarray(patch_size)
        # 先取更大的区域再做旋转/缩放，避免先切512造成旋转后丢失周边信息。
        self.initial_size = np.ceil(np.linalg.norm(self.patch_size) / 0.7).astype(int)
        self.transforms = training_transforms(patch_size) if self.training else None
        self.foreground = {}
        if self.training:
            for index, name in enumerate(self.files):
                with h5py.File(self.data_dir / name, "r") as file:
                    case = str(file.attrs["case"])
                    counts = file.attrs["class_counts"]
                for class_id in np.flatnonzero(counts[1:]) + 1:
                    if label_map is not None and label_map[class_id] == 0:
                        continue
                    self.foreground.setdefault(case, {}).setdefault(int(class_id), []).append((index, int(counts[class_id])))

    def __getitem__(self, index):
        class_id = 0
        if isinstance(index, tuple):
            index, class_id = index
        with h5py.File(self.data_dir / self.files[index], "r") as file:
            image, label = file["image"][:], file["masks"][:]
            case, z = str(file.attrs["case"]), int(file.attrs["slice_index"])
            if class_id:
                coords = file["foreground"][str(class_id)][:]
                center = coords[np.random.randint(len(coords))]
        if self.label_map is not None:
            label = self.label_map[label]
        if self.training:
            if not class_id:
                # 普通随机crop；图像不足patch大小时围绕图像中心padding。
                center = np.array([np.random.randint(p // 2, n - p + p // 2 + 1) if n > p else n // 2
                                   for n, p in zip(image.shape, self.patch_size)])
            # 按最终patch约束中心：靠边就移框，只有图像不足patch的方向才补齐。
            shape, half = np.asarray(image.shape), self.patch_size // 2
            center = np.where(
                shape >= self.patch_size,
                np.clip(center, half, shape - self.patch_size + half),
                shape // 2,
            )
            size = int(self.initial_size)
            lower = center - size // 2
            upper = lower + size
            source = tuple(slice(max(0, int(a)), min(n, int(b))) for a, b, n in zip(lower, upper, image.shape))
            destination = tuple(slice(max(0, int(-a)), max(0, int(-a)) + s.stop-s.start) for a, s in zip(lower, source))
            patch_image = np.zeros((size, size), dtype=np.float32)
            patch_label = np.full((size, size), -1, dtype=np.int16)
            patch_image[destination], patch_label[destination] = image[source], label[source]
            transformed = self.transforms(image=torch.from_numpy(patch_image[None]),
                                          segmentation=torch.from_numpy(patch_label[None]))
            return transformed["image"].float(), transformed["segmentation"].long(), case, z
        return torch.from_numpy(image[None]).float(), torch.from_numpy(label[None]).long(), case, z


class ForegroundBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, foreground_fraction=0.33, seed=1337):
        self.dataset, self.batch_size = dataset, batch_size
        self.normal_count = round(batch_size * (1 - foreground_fraction))
        self.seed, self.epoch = seed, 0
        self.cases = list(dataset.foreground)

    def __len__(self):
        return max(1, math.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            batch = []
            for position in range(self.batch_size):
                if position < self.normal_count or not self.cases:
                    batch.append((int(rng.integers(len(self.dataset))), 0))
                    continue
                organs = self.dataset.foreground[rng.choice(self.cases)]
                class_id = int(rng.choice(list(organs)))
                candidates = organs[class_id]
                counts = np.array([count for _, count in candidates], dtype=float)
                index = candidates[rng.choice(len(candidates), p=counts / counts.sum())][0]
                batch.append((index, class_id))
            yield batch
