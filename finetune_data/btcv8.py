"""BTCV原始13器官标签转Synapse八器官；原始H5不改动。"""
import numpy as np

# 新编号0~8对应的原始编号：背景、脾、右肾、左肾、胆囊、胰腺、肝、胃、主动脉。
BTCV8_CLASS_IDS = (0, 1, 2, 3, 4, 11, 6, 7, 8)
BTCV8_LABEL_MAP = np.zeros(14, dtype=np.uint8)
BTCV8_LABEL_MAP[list(BTCV8_CLASS_IDS)] = np.arange(9, dtype=np.uint8)
