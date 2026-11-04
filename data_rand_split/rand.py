import os
import random
import shutil
from pathlib import Path


def copy_subset(img_full_dir, label_full_dir, img_subset_dir, label_subset_dir, selected_imgs):
    os.makedirs(img_subset_dir, exist_ok=True)
    os.makedirs(label_subset_dir, exist_ok=True)

    for img_path in selected_imgs:
        img_name = img_path.name

        # 例如 A4J4S4_sa.nii.gz -> A4J4S4_sa_gt.nii.gz
        label_name = img_name.replace(".nii.gz", "_gt.nii.gz")

        label_path = label_full_dir / label_name

        if not label_path.exists():
            raise FileNotFoundError(f"找不到对应标签: {label_path}")

        shutil.copy2(img_path, img_subset_dir / img_name)
        shutil.copy2(label_path, label_subset_dir / label_name)


def main():
    seed = 2025
    random.seed(seed)

    img_full_dir = Path("/data/WPW/datasets/mms/train/img/full")
    label_full_dir = Path("/data/WPW/datasets/mms/train/label/full")

    img_subset10_dir = Path("/data/WPW/datasets/mms/train/img/subset10")
    label_subset10_dir = Path("/data/WPW/datasets/mms/train/label/subset10")

    img_subset20_dir = Path("/data/WPW/datasets/mms/train/img/subset20")
    label_subset20_dir = Path("/data/WPW/datasets/mms/train/label/subset20")

    all_imgs = sorted(img_full_dir.glob("*.nii.gz"))

    total_num = len(all_imgs)
    subset10_num = round(total_num * 0.10)
    subset20_num = round(total_num * 0.20)

    print(f"总图像数量: {total_num}")
    print(f"subset10 数量: {subset10_num}")
    print(f"subset20 数量: {subset20_num}")

    # 随机打乱
    random.shuffle(all_imgs)

    # 让 subset10 包含在 subset20 里面，更公平
    selected_subset20 = all_imgs[:subset20_num]
    selected_subset10 = selected_subset20[:subset10_num]

    copy_subset(
        img_full_dir,
        label_full_dir,
        img_subset10_dir,
        label_subset10_dir,
        selected_subset10,
    )

    copy_subset(
        img_full_dir,
        label_full_dir,
        img_subset20_dir,
        label_subset20_dir,
        selected_subset20,
    )

    print("复制完成！")
    print(f"subset10 图像: {len(list(img_subset10_dir.glob('*.nii.gz')))}")
    print(f"subset10 标签: {len(list(label_subset10_dir.glob('*.nii.gz')))}")
    print(f"subset20 图像: {len(list(img_subset20_dir.glob('*.nii.gz')))}")
    print(f"subset20 标签: {len(list(label_subset20_dir.glob('*.nii.gz')))}")


if __name__ == "__main__":
    main()