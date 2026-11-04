"""
Process Inference Masks
-----------------------
This script processes the masks generated from SAM inference

步骤2：
    对于train集中的数据，保留能满足覆盖95%区域的K值作为锚点，其他较小区域临近插值到这K+1个区域中
    pre_processed 之后的图像对应的各区域 label，本质就是筛选完 K+1 个区域后再按面积重新编号
    注：---如果按照这个代码逻辑，则 K+1 个区域中可能包括未被SAM分割的区域，
        ---如果这个代码是错的按照generate_masks得到的label=0区域是未被SAM分割的区域，则 K+1 个区域中一定会包含这部分区域

"""

import numpy as np
import os
import h5py
import cv2
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt


def normalize(x):
    return x / 255.0 * 2 - 1


def caculate_anchor_submasks_num(input_folder):
    count_values = []
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.h5'):
            file_path = os.path.join(input_folder, file)
            with h5py.File(file_path, 'r') as f:
                masks = f['masks'][:]
            labeled_masks = label(masks)
            props = regionprops(labeled_masks)
            sorted_regions = sorted(props, key=lambda x: x.area, reverse=True) # SAM分割子区域按面积大小排序
            total_foreground_area = np.sum(masks > 0)
            threshold_area = 0.95 * total_foreground_area if total_foreground_area > 0 else 0 # 95%面积阈值
            accumulated_area = 0
            count = 0
            for region in sorted_regions:
                accumulated_area += region.area
                count += 1
                if accumulated_area >= threshold_area:
                    break
            count_values.append(count)
    area_num = int(np.ceil(np.mean(count_values))) if count_values else 0

    return area_num


def reorder_masks_by_area(new_masks):
    unique_labels, counts = np.unique(new_masks, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]
    label_mapping = {label: new_idx for new_idx, label in enumerate(sorted_labels)}
    remapped_masks = np.vectorize(label_mapping.get)(new_masks)
    remapped_masks += 1

    return remapped_masks


def process_masks(input_folder, output_folder, area_num, target_size):
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.h5'):
            file_path = os.path.join(input_folder, file)
            with h5py.File(file_path, 'r') as f:
                masks = f['masks'][:]
                images = f['image'][:]
            labeled_masks = masks
            props = regionprops(labeled_masks)
            sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
            largest_regions = sorted_regions[:area_num]
            largest_labels = [region.label for region in largest_regions]

            new_masks = np.full_like(labeled_masks, fill_value=-1, dtype=np.int32)
            for idx, label_value in enumerate(largest_labels, start=1):
                new_masks[labeled_masks == label_value] = idx

            new_masks[labeled_masks == 0] = 0
            remaining_mask = (new_masks == -1)
            large_region_mask = (new_masks > -1)

            distance_map, nearest_index = distance_transform_edt(~large_region_mask, return_indices=True)
            nearest_large_region = new_masks[nearest_index[0], nearest_index[1]]
            new_masks[remaining_mask] = nearest_large_region[remaining_mask]
            new_masks = reorder_masks_by_area(new_masks)
            unique_labels = np.unique(new_masks)

            resized_image = cv2.resize(images, target_size, interpolation=cv2.INTER_LINEAR)
            normalized_image = normalize(resized_image)
            normalized_masks = cv2.resize(new_masks, target_size, interpolation=cv2.INTER_NEAREST)

            output_h5_path = os.path.join(output_folder, file)
            with h5py.File(output_h5_path, 'w') as out_f:
                out_f.create_dataset('image', data=normalized_image, compression='gzip')
                out_f.create_dataset('masks', data=normalized_masks, compression='gzip')
                # 最终的保存数据包括：
                #   image: 归一化后的图像
                #   masks: 归一化后的掩码 K+1个子区域(包括未被分割的区域) 由面积从大到小编号-对应预训练的最终label标签

    print("Number of classes in new_masks:", len(unique_labels)) # FLARE:10


def write_list(data_dir, list_file):
    file_names = os.listdir(data_dir)
    file_names.sort()
    with open(list_file, 'w') as f:
        for name in file_names:
            f.write(name + '\n')


if __name__ == '__main__':
    dataset = "FLARE" # "mms" / "FLARE"（腹部数据集）
    input_folder = f"./outputs/{dataset}_masks"
    output_folder = f"./outputs/{dataset}_pre_processed_masks/data"
    os.makedirs(output_folder, exist_ok=True)
    list_file = f"./outputs/{dataset}_pre_processed_masks/train_slices.list"
    target_size = (256, 256)
    area_num = caculate_anchor_submasks_num(input_folder) # area_num: FLARE:K=9
    process_masks(input_folder, output_folder, area_num, target_size)
    write_list(output_folder, list_file)
