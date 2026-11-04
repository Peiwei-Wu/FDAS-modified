import json
from .btcv import load_btcv_split
from .chaos import load_chaos_split
from .chaos_mri import load_chaos_mri_split
from .words import load_words_split


def get_dataset_split(dataset, data_root, output_dir, seed, btcv_split="original"):
    """在当前预处理目录读取或生成病例划分；H5由prepare_slices生成。"""
    if dataset == "BTCV":
        return load_btcv_split(data_root, output_dir, seed, btcv_split)
    if dataset == "CHAOS":
        return load_chaos_split(data_root, output_dir, seed)
    if dataset == "CHAOS_MRI":
        return load_chaos_mri_split(data_root, output_dir, seed)
    if dataset == "WORD":
        return load_words_split(data_root, output_dir, seed)
    if dataset == "AMOS":
        # 在AMOS_nnunet2d中提供显式splits.json，不擅自划分CT/MRI病例。
        with open(output_dir / "splits.json", encoding="utf-8") as file:
            return json.load(file)
    raise NotImplementedError(f"{dataset}的目录和配对规则尚未添加")
