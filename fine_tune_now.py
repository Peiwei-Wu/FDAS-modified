import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from network.unet import UNet
from utils.parse_config import parse_config, parse_bool, set_random
from finetune_data import get_dataset_split
from finetune_data.btcv8 import BTCV8_CLASS_IDS, BTCV8_LABEL_MAP
from utils.finetuning.preprocessing import prepare_slices
from utils.finetuning.dataset import PatchDataset, ForegroundBatchSampler
from utils.finetuning.evaluation import evaluate, save_evaluation
from utils.finetuning.postprocessing import get_fixed_postprocessing, select_postprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FDAS 2D UNet")
    parser.add_argument("--dataset", type=str.upper, default="BTCV8",
                        choices=["BTCV", "BTCV8", "CHAOS", "CHAOS_MRI", "WORD", "AMOS"])
    parser.add_argument("--config", default="./config/config.cfg")
    parser.add_argument("--seed", type=int, default=1337)
    for name, default in [("prepare_only", False), ("save_result", True),
                          ("train", True), ("test", True), ("pre_weights", True)]:
        parser.add_argument(f"--{name}", type=parse_bool, nargs="?", const=True, default=default)
    parser.add_argument("--checkpoint", default="./results/BTCV8/2026-11-03_22-00-40/model/model-final.pth")
    return parser.parse_args()


def create_loaders(output_dir, train_config, patch_size, seed, label_map=None):
    background_ratio = train_config["background_ratio"] if train_config["filter_background"] else None
    train_set = PatchDataset(output_dir, "train", patch_size, background_ratio, seed, label_map)
    valid_set = PatchDataset(output_dir, "valid", patch_size, label_map=label_map)
    test_set = PatchDataset(output_dir, "test", patch_size, label_map=label_map)
    sampler = ForegroundBatchSampler(train_set, int(train_config["batch_size"]),
                                     float(train_config.get("foreground_oversample", 0.33)), seed)
    common = dict(num_workers=int(train_config["num_workers"]), pin_memory=torch.cuda.is_available())
    return (DataLoader(train_set, batch_sampler=sampler, **common),
            DataLoader(valid_set, batch_size=1, **common),
            DataLoader(test_set, batch_size=1, **common))


def load_pretrained_model(num_classes, device, checkpoint_path, lr_encoder, lr_decoder, max_epochs=500, lr_head=None):
    model = UNet(lr_encoder=lr_encoder, lr_decoder=lr_decoder, in_chns=1,
                 class_num=num_classes, pre_train=False, max_epochs=max_epochs, lr_head=lr_head).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    current = model.state_dict()
    # 保持原来的Encoder + Decoder主体加载方式，shape不匹配的分割头不加载。
    pretrained_dict = {
        k: v for k, v in checkpoint.items() 
        if k in current and v.shape == current[k].shape   
    }
    current.update(pretrained_dict)
    model.load_state_dict(current)
    logging.info("Pretrained weights: %s; matched tensors: %d/%d", checkpoint_path, len(pretrained_dict), len(current))
    return model


def train(config, model, train_loader, valid_loader, split_info,
          data_root, class_names, device, result_dir):
    settings = config["train"]
    num_epochs, valid_epoch = int(settings["num_epochs"]), int(settings["valid_epoch"])
    model_dir = result_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "model-latest-best.pth"
    best_dice, no_improvement = -1.0, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, sample_count = 0.0, 0
        used_lr = model.enc_opt.param_groups[0]["lr"]
        decoder_lr, head_lr = [group["lr"] for group in model.dec_opt.param_groups]
        progress = tqdm(train_loader, ncols=90, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for images, labels, _, _ in progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            loss = model.train_(images, labels)
            total_loss += loss * images.size(0)
            sample_count += images.size(0)
            progress.set_postfix(loss=f"{loss:.4f}")
        model.update_lr()
        logging.info("Epoch %d/%d - loss: %.4f - lr: %.8f - decoder_lr: %.8f - head_lr: %.8f",
                     epoch + 1, num_epochs, total_loss / sample_count, used_lr, decoder_lr, head_lr)
        if epoch + 1 >= 400 and (epoch + 1) % 10 == 0: # 每隔10个epoch保存一次权重
            torch.save(model.state_dict(), model_dir / f"model-epoch-{epoch + 1}.pth")
        torch.save(model.state_dict(), model_dir / "model-latest.pth")
        
        if split_info["valid"] and ((epoch + 1) % valid_epoch == 0 or epoch + 1 == num_epochs):
            valid_dice, _, _ = evaluate(model, valid_loader, split_info["valid"], data_root,
                                         class_names, device, inference=config.get("inference", {}))
            logging.info("Epoch %d - validation Dice: %.4f", epoch + 1, valid_dice)
            if valid_dice > best_dice:
                best_dice, no_improvement = valid_dice, 0
                torch.save(model.state_dict(), best_path)
                # 只保留最优的一个checkpoint，覆盖上一次的结果
                # torch.save(model.state_dict(), model_dir / f"model-best-{epoch + 1}-{valid_dice:.3f}.pth")
            else:
                no_improvement += 1
            if no_improvement >= int(settings["patience"]):
                logging.info("Early stopping at epoch %d", epoch + 1)
                break

    if not split_info["valid"]: # 没有验证集的情况，保留最后一次的模型权重
        best_path = model_dir / "model-final.pth"
        torch.save(model.state_dict(), best_path)

    return best_path


def main():
    args = parse_args()
    set_random(args.seed)
    config = parse_config(args.config)
    evaluation_suffix = "_zsmooth" if config.get("inference", {}).get("z_smooth_sigma", 0) > 0 else ""
    fine, training = config["finetune"], config["train"]
    dataset = args.dataset
    source_dataset = "BTCV" if dataset == "BTCV8" else dataset

    # prepare .h5 slices in ./outputs/xxx
    # 只用经nnU-net得到的各数据集明确配置的Plans参数；缺少配置直接报错
    settings = config[f"preprocess_{source_dataset.lower()}"]
    data_root = Path(fine[f"data_root_{source_dataset.lower()}"])
    # 病例划分、切片列表、H5及空间信息统一放在新目录
    output_dir = Path(fine["output_root"]) / f"{source_dataset}_nnunet2d"
    split_info = get_dataset_split(source_dataset, data_root, output_dir, args.seed,
                                  btcv_split=fine.get("btcv_split", "original"))
    prepare_slices(split_info, data_root, output_dir, settings) # 检查.h5切片数据是否准备好
    if args.prepare_only:
        print(f"H5 data saved to: {output_dir}")
        return

    # prepare logging
    label_map = BTCV8_LABEL_MAP if dataset == "BTCV8" else None # BTCV13器官label映射为8器官label
    if label_map is not None:
        labels = {str(i): split_info["labels"][str(old)] for i, old in enumerate(BTCV8_CLASS_IDS)}
        split_info = {**split_info, "dataset": dataset, "labels": labels,
                      "original_to_target": label_map.tolist()}
    num_classes = int(config["network"][f"n_classes_{dataset.lower()}"])
    class_names = [split_info["labels"][str(i)] for i in range(num_classes)]
    num_epochs = int(training["num_epochs"])
    device = torch.device(f"cuda:{int(training['gpu'])}" if torch.cuda.is_available() else "cpu")
    result_dir = Path("results") / dataset / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(result_dir / f"train{evaluation_suffix}.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    for name, content in [("run_config", config), ("splits", split_info), ("preprocessing", settings)]:
        (result_dir / f"{name}{evaluation_suffix}.json").write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Dataset: %s; device: %s; train/valid/test: %d/%d/%d", dataset, device,
                 len(split_info["train"]), len(split_info["valid"]), len(split_info["test"]))
    logging.info("Final-output Dice (foreground) + CE (all classes), AdamW + PolyLR(power=0.9)")

    # load train/valid/test datasets
    train_loader, valid_loader, test_loader = create_loaders(
        output_dir, training, fine["target_size"], args.seed, label_map)
    lr_encoder, lr_decoder, lr_head = float(training["lr_encoder"]), float(training["lr_decoder"]), float(training["lr_head"])

    # load pre-trained parameters or not
    if args.train and args.pre_weights:
        checkpoint = Path("models") / str(fine["pretrain_dataset"]) / str(fine["pretrain_exp"]) / str(fine["pretrain_checkpoint"])
        model = load_pretrained_model(num_classes, device, checkpoint, lr_encoder, lr_decoder, num_epochs, lr_head)
    else:
        model = UNet(lr_encoder=lr_encoder, lr_decoder=lr_decoder, in_chns=1,
                     class_num=num_classes, pre_train=False, max_epochs=num_epochs, lr_head=lr_head).to(device)

    # step1: training and validing
    if args.train:
        best_path = train(config, model, train_loader, valid_loader, split_info, data_root,
                      class_names, device, result_dir)
    else:
        best_path = Path(args.checkpoint)
    
    checkpoint = torch.load(best_path, map_location=device) # last or best weight path
    model.load_state_dict(checkpoint)
    logging.info("Evaluation checkpoint: %s", best_path.resolve())

    # step2: 有验证集先用验证集选规则；没有验证集才使用数据集固定方案。
    inference = config.get("inference", {})
    selection = {"rules": []}
    fixed_postprocess = None
    if inference.get("postprocess", True):
        if split_info["valid"]: 
            # 有验证集先用验证集选规则
            validation_cache = []
            save_evaluation(f"valid{evaluation_suffix}_raw", model, valid_loader, split_info["valid"], data_root,
                            class_names, device, result_dir, args.save_result,
                            inference=inference, cache=validation_cache)
            selection = select_postprocessing(validation_cache, num_classes) # 根据验证集选择最优后处理策略
            save_evaluation(f"valid{evaluation_suffix}", model, valid_loader, split_info["valid"], data_root,
                            class_names, device, result_dir, args.save_result,
                            cached=validation_cache, rules=selection["rules"],
                            reuse_raw=not selection["rules"])
            del validation_cache
        else: 
            # 无验证集则采用不同数据集固定的后处理策略
            fixed_postprocess = get_fixed_postprocessing(dataset)
            if fixed_postprocess:
                logging.info("No validation set: use %s fixed postprocessing.", dataset)
            else:
                logging.info("No fixed postprocessing for %s; use raw predictions.", dataset)
    
    # step3: testing
    if args.test and split_info["test"]:
        test_postprocess = bool(selection["rules"] or fixed_postprocess)
        test_cache = [] if test_postprocess else None
        save_evaluation(f"test{evaluation_suffix}_raw", model, test_loader, split_info["test"], data_root,
                        class_names, device, result_dir, args.save_result,
                        inference=inference, cache=test_cache) # 保留纯模型预测，便于与后处理结果对比。
        save_evaluation(f"test{evaluation_suffix}", model, test_loader, split_info["test"], data_root,
                        class_names, device, result_dir, args.save_result,
                        cached=test_cache, rules=selection["rules"],
                        fixed_postprocess=fixed_postprocess,
                        reuse_raw=not test_postprocess) # 没有规则时复用原始结果。
    print(f"Finished. Results: {result_dir}")


if __name__ == "__main__":
    main()
