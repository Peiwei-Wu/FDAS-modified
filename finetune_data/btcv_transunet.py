"""TransUNet官方Synapse病例划分；仅按病例编号分组，不读取旧切片列表。"""

# 来源：https://github.com/Beckschen/TransUNet/tree/main/lists/lists_Synapse
TRAIN_IDS = (5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40)
TEST_IDS = (8, 22, 38, 36, 32, 2, 29, 3, 1, 4, 25, 35)


def load_transunet_split(split_info):
    """按官方病例编号重组18/12划分，切片名单由预处理统一生成。"""
    records = {item["case"]: item for s in ("train", "valid", "test") for item in split_info[s]}
    result = {**split_info, "protocol": "transunet_18_12"}
    result["train"] = [records[f"img{i:04d}"] for i in TRAIN_IDS]
    result["valid"] = []
    result["test"] = [records[f"img{i:04d}"] for i in TEST_IDS]
    return result
