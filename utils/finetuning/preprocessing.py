"""按物理spacing生成H5切片；保留原始Z轴和病例几何信息。"""

import json

import h5py
import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from tqdm import tqdm


def normalize_image(image, settings):
    image = image.astype(np.float32)
    if settings["normalization"] == "CT":
        # 方案1: 参考nnU-net建议配置标准化
        # image = (np.clip(image, *settings["clip"]) - settings["mean"]) / settings["std"]
        
        # 方案2: 更符合上游预训练逐切片归一化到[-1,1]
        image = np.clip(image, -125, 225) # 博客推荐截断
        # image = np.clip(image, *settings["clip"]) # nnU-net推荐截断（无论选哪个反正都要截断，不然BTCV的图片值太大）
        vmin = image.min(axis=(1, 2), keepdims=True)
        vmax = image.max(axis=(1, 2), keepdims=True) 
        image = 2 * (image - vmin) / np.maximum(vmax - vmin, 1e-8) - 1
        return image

    if settings["normalization"] == "zscore":
        # MRI逐病例z-score，不套用CT的HU参数。
        return (image - image.mean()) / max(float(image.std()), 1e-8)
    raise ValueError("normalization仅支持CT或zscore")


def prepare_slices(split_info, data_root, output_dir, settings):
    """重采样XY并保存前景坐标；训练patch在Dataset中在线裁剪。"""
    # 复用H5前也检查，避免已有缓存掩盖错误的归一化名称。
    if settings["normalization"] not in ("CT", "zscore"):
        raise ValueError("normalization仅支持CT或zscore")
    data_dir, metadata_dir = output_dir / "data", output_dir / "cases"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "preprocessing.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    unchanged = previous == settings
    rng = np.random.default_rng(1337)
    num_classes = len(split_info["labels"])
    for split in ("train", "valid", "test"):
        names = []
        for item in tqdm(split_info[split], desc=f"Prepare {split}"):
            metadata_path = metadata_dir / f"{item['case']}.json"
            if unchanged and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                case_names = [f"{item['case']}_slice{z}.h5" for z in range(metadata["original_shape"][0])]
                if all((data_dir / name).exists() for name in case_names):
                    names.extend(case_names)
                    continue
            image_itk = sitk.ReadImage(str(data_root / item["image"]))
            image = sitk.GetArrayFromImage(image_itk)
            label = sitk.GetArrayFromImage(sitk.ReadImage(str(data_root / item["label"])))
            spacing = list(reversed(image_itk.GetSpacing()))
            # 图像非零包围框，只裁XY；不根据器官Label裁剪，不删除Z切片。
            nonzero_y, nonzero_x = np.where(np.any(image != 0, axis=0))
            if nonzero_y.size:
                y0, y1, x0, x1 = int(nonzero_y.min()), int(nonzero_y.max()) + 1, int(nonzero_x.min()), int(nonzero_x.max()) + 1
            else:
                y0, y1, x0, x1 = 0, image.shape[1], 0, image.shape[2]
            shape = np.maximum(1, np.round(np.array([y1-y0, x1-x0]) * spacing[1:] / np.array(settings["spacing"]))).astype(int)
            metadata = dict(original_shape=list(image.shape), spacing=spacing,
                            origin=list(image_itk.GetOrigin()), direction=list(image_itk.GetDirection()),
                            bbox=[y0, y1, x0, x1], resampled_shape=shape.tolist())
            image = normalize_image(image[:, y0:y1, x0:x1], settings)
            label = label[:, y0:y1, x0:x1]
            for z in range(image.shape[0]):
                name = f"{item['case']}_slice{z}.h5"
                names.append(name)
                image_slice = resize(image[z], shape, order=3, mode="edge", anti_aliasing=False, preserve_range=True).astype(np.float32)
                # 每个类别的0/1掩码线性插值后取最大响应，输出仍为整数标签。
                label_slice = resize_segmentation(label[z], shape, order=1).astype(np.uint8)
                with h5py.File(data_dir / name, "w") as file:
                    file.create_dataset("image", data=image_slice)
                    file.create_dataset("masks", data=label_slice)
                    file.attrs.update(case=item["case"], slice_index=z,
                                      class_counts=np.bincount(label_slice.ravel(), minlength=num_classes))
                    foreground = file.create_group("foreground")
                    for class_id in np.unique(label_slice):
                        if class_id == 0:
                            continue
                        coords = np.argwhere(label_slice == class_id)
                        coords = coords[rng.choice(len(coords), min(512, len(coords)), replace=False)]
                        foreground.create_dataset(str(class_id), data=coords.astype(np.int32))
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output_dir / f"{split}_slices.list").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    config_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "splits.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")
