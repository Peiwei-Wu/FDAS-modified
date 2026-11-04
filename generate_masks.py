"""
Inference script using SAM to generate segmentation masks

步骤1：
    对于train集中的数据(大量做预训练的数据)生成SAM分割的掩码结果，从而用于预训练

# 潜在问题（已解决）： #! 但这也并不是问题，因为预训练的目的只是让模型做前期理解
    同一个器官，在不同切片里，被映射成了不同的数字编号

"""

from pathlib import Path
import torch
import os
import numpy as np
import cv2
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import h5py
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import faulthandler
faulthandler.enable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def convert_gray(rgb_img_array):
    unique_colors = np.unique(rgb_img_array.reshape(-1, rgb_img_array.shape[2]), axis=0) # 所有可能的RGB从小到大排序
    color_to_gray_map = {tuple(color): i for i, color in enumerate(unique_colors)} # RGB从小到大依次对应0/1/2...
    gray_img = np.zeros((rgb_img_array.shape[0], rgb_img_array.shape[1]), dtype=np.uint8)
    for i in range(rgb_img_array.shape[0]):
        for j in range(rgb_img_array.shape[1]):
            rgb_value = tuple(rgb_img_array[i, j, :])
            gray_img[i, j] = color_to_gray_map[rgb_value]
    return gray_img


def save_anns(anns):
    # anns是一个列表，每个元素是一个字典（对应SAM分割出来的某一个子区域），包含segmentation和area
    #   segmentation就是H*W的一个bool张量，里面为0/1，1代表SAM分割的某个子区域
    #   area就是对应的面积大小
    if len(anns) == 0: # 如果SAM没有分割出任何子区域
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True) # 按照每个掩码的面积从大到小进行排序
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    # 遍历排序后的每个掩码实例（从大面积到小面积，确保小面积叠加在大面积之上）
    for ann in sorted_anns:
        m = ann['segmentation'] # 提取当前掩码的二值布尔矩阵 (H, W)，0/1表示当前像素是否属于该分割物体
        color_mask = np.concatenate([np.random.random(3), [0.35]]) # 生成随机的 RGB 颜色
        img[m] = color_mask # 将当前掩码区域（True位置）赋值为该随机颜色

    return img


def main(dataset, input_fold, save_data_root, sam_path, target_size):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path).to(device)
    sam.eval() # 只做预测
    mask_generator = SamAutomaticMaskGenerator(sam)

    if dataset == "mms":
        all_niis = [f for f in os.listdir(input_fold) if f.endswith("nii.gz")] # 直接读取文件名
    elif dataset == "FLARE":
        input_fold = Path(input_fold)
        all_niis = list(input_fold.rglob("*.nii.gz"))
        all_niis = [p for p in all_niis if p.name.endswith("_0000.nii.gz")] # 读取的是路径
    # random.shuffle(all_niis) # 不用随机打乱按顺序读取

    for nii_file in all_niis:
        if dataset == "mms":
            nii_path = os.path.join(input_fold, nii_file)
            case_name = nii_file.split('.')[0]
        elif dataset == "FLARE":
            nii_path = str(nii_file) 
            file_name = nii_file.name # 只取文件名
            split_name = nii_file.parent.name
            case_name = f"{split_name}_{file_name.replace('_0000.nii.gz', '')}"
        print(case_name)

        img_sitk = sitk.ReadImage(nii_path)
        img_arrays = sitk.GetArrayFromImage(img_sitk)
        img_arrays = (img_arrays - img_arrays.min()) / (img_arrays.max() - img_arrays.min())
        # print('Shape:', img_arrays.shape) # mms: (N, 256, 256)  FLARE: (N, 512, 512)
        
        for slice_idx in range(img_arrays.shape[0]):
            if not os.path.exists(save_data_root + f'/{case_name}_slice{slice_idx}.h5'):
                image = img_arrays[slice_idx, :, :]
                img_max, img_min = image.max(), image.min()

                den = img_max - img_min
                if den > 0:
                    image = (image - img_min) / den  # 标准归一化 #todo 线性压缩 （可以尝试进行增强对比度的操作，用非线性分段映射）
                else:
                    print(f"skip empty slice: {case_name}_slice{slice_idx}", flush=True)
                    continue

                if image.shape != target_size:
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR) # 统一为(256, 256)
                image = (image * 255).astype(np.uint8) # 消除不同切片间的绝对亮度差异
                # print('Shape:', image.shape) # (256, 256)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                with torch.no_grad():
                    masks = mask_generator.generate(image_rgb) #* 生成掩码字典: n个{'segmentation', 'area', ...}, 一张图分割多个子区域, 每个子区域对应的segmentation就是一个bool矩阵掩码0/1，area就是对应的面积大小
                mask_ = save_anns(masks) # 所有掩码区域分别涂上了RGB颜色 (H, W, 4)

                if mask_ is not None:
                    mask_ = (mask_ * 255).astype(np.uint8)
                    mask_ = mask_[:, :, :3] # 只保存RGB三颜色 (H, W, 3)
                    gray_mask = convert_gray(mask_) # 转换为灰度标签(H, W), 每个像素对应一个掩码标签0/1/2/3...
                    with h5py.File(save_data_root + f'/{case_name}_slice{slice_idx}.h5', 'w') as h5f:
                        h5f.create_dataset('masks', data=gray_mask) # 掩码标签
                        h5f.create_dataset('image', data=image)     # 归一化后的原始图像
                    
                    #* H5文件是一个数据仓库，把图像和对应的分割结果打包存储在一个文件里
                    # 例：h5文件(case_name_slice_slice_idx.h5)
                    #     ---> "image" 原始医学图像数据    (H, W) 0-255灰度值
                    #     ---> "masks" SAM分割掩码标签数据 (H, W) 每个像素值为1/2...表示属于哪个器官, 0表示未被SAM分割出来的区域(very small)


if __name__ == "__main__":
    dataset = "FLARE" # "mms" / "FLARE"
    target_size = (256, 256) # SAM分割预训练时就转为(256, 256)

    if dataset == "mms": # 心脏 会议方法
        input_fold = "/data/WPW/datasets/mms/train/img/full/"
    elif dataset == "FLARE": # 腹部 扩刊方法
        input_fold = "/mnt/e/PeiweiWu/datasets/FLARE/Training"

    save_data_root = f"./outputs/{dataset}_masks" # 保存SAM分割掩码的图像
    os.makedirs(save_data_root, exist_ok=True)

    sam_path = "/mnt/e/PeiweiWu/SSL/pretrained_model/sam_vit_h.pth"
    main(dataset, input_fold, save_data_root, sam_path, target_size)
