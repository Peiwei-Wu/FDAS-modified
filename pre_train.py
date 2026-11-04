"""
Script for self-supervised pretraining on the full training dataset.

步骤3：
    在大量的腹部数据集上做预训练

"""

import argparse
import csv
from datetime import datetime
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import losses
from network.unet import UNet
from utils.dataloader import Pretrain_Datasets
import faulthandler
from utils.parse_config import parse_config

# 在import之后直接添加以下启用代码即可
faulthandler.enable()

# Disable cuDNN to avoid segmentation fault with PyTorch 2.3.1
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='default', help='name of the experiment')
parser.add_argument('--datasets', type=str, default='FLARE', # "mms" / "FLARE"（腹部数据集）
                    help='name of training datasets')
parser.add_argument('--channel_num', type=int, default=1, help='input channel of the network')
parser.add_argument('--mask_ratio', type=float, default=0.7, help='mask ratio')
parser.add_argument('--alpha', type=float, default=0.5, help='loss weight')
parser.add_argument('--max_iterations', type=int, default=870800, help='maximum number of training iterations') # 之前一部分数据时是100000
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--root_path', type=str, default="./outputs", help='root path of data')
parser.add_argument('--sam_seg_class', type=int, default=10, help='number of new_masks classes obtained from pre_process.py')
parser.add_argument('--recon_class_num', type=int, default=1, help='reconstruction class num in pretrain')
args = parser.parse_args()


# 获取预处理完成的SAM分割掩码数据集.h5文件
def get_h5_datasets(dataset):
    print(dataset)
    h5_data = []
    h5_data.append(f"{dataset}_pre_processed_masks")
    return h5_data


def contrastive_loss(anchor_embedding, positive_embedding, negative_embeddings, temperature=0.1):
    """
    Compute InfoNCE loss
    """
    pos_sim = F.cosine_similarity(anchor_embedding.unsqueeze(0), positive_embedding.unsqueeze(0), dim=1) / temperature
    neg_sim = F.cosine_similarity(anchor_embedding.unsqueeze(0), negative_embeddings, dim=1) / temperature

    pos_sim = pos_sim.squeeze(0)
    pos_exp = torch.exp(pos_sim)
    neg_exp = torch.exp(neg_sim)
    neg_sim_sum = torch.sum(neg_exp, dim=0)

    numerator = pos_exp
    denominator = pos_exp + neg_sim_sum
    loss = -torch.log(numerator / denominator)

    return loss.mean()


def compute_contrastive_loss(A_embeddings, A1_embeddings):
    """
    Compute contrastive loss over the entire batch.

    Args:
        A_embeddings: Feature embeddings of original images
        A1_embeddings: Feature embeddings of fused images
    """
    batch_size = A_embeddings.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        anchor_embedding = A_embeddings[i]
        positive_embedding = A1_embeddings[i]
        negative_indices = [j for j in range(batch_size) if j != i]
        negative_embeddings = A1_embeddings[negative_indices]

        loss = contrastive_loss(anchor_embedding, positive_embedding, negative_embeddings)
        total_loss += loss

    contrastive_loss_ = total_loss / batch_size

    return contrastive_loss_


def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    alpha = args.alpha
    mask_num = round(args.sam_seg_class * args.mask_ratio)
    remain_num = args.sam_seg_class - mask_num

    model = UNet(in_chns=args.channel_num, num_class_seg=args.sam_seg_class, 
                 num_class_recon=args.recon_class_num, pre_train=True).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    h5_datasets = get_h5_datasets(args.datasets)
    train_set = Pretrain_Datasets(base_dir=args.root_path, datasets=h5_datasets, transform=None)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    dice_loss = losses.DiceLoss(args.sam_seg_class)
    mse_loss = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") # FP16梯度缩放，防止梯度过小导致数值下溢

    # 每步、每轮分别记录，追加保存并用run区分不同启动。
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    step_file = open(os.path.join(snapshot_path, 'iteration_loss.csv'), 'a', newline='', buffering=1)
    epoch_file = open(os.path.join(snapshot_path, 'epoch_loss.csv'), 'a', newline='', buffering=1)
    step_writer, epoch_writer = csv.writer(step_file), csv.writer(epoch_file)
    if step_file.tell() == 0:
        step_writer.writerow(['epoch', 'iteration', 'loss', 'dice_loss', 'recon_loss', 'contrast_loss', 'lr'])
    if epoch_file.tell() == 0:
        epoch_writer.writerow(['run', 'epoch', 'iteration', 'samples', 'loss', 'dice_loss', 'recon_loss', 'contrast_loss'])
    logging.info('Loss CSV run: %s, directory: %s', run_id, snapshot_path)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    display_epochs = (max_iterations + len(trainloader) - 1) // len(trainloader)
    for epoch_num in range(max_epoch):
        epoch_losses, epoch_samples = np.zeros(4), 0
        progress = tqdm(total=min(len(trainloader), max_iterations - iter_num),
                        desc=f'Epoch {epoch_num + 1}/{display_epochs}', dynamic_ncols=True,
                        disable=not sys.stderr.isatty())
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, name_batch = sampled_batch['image'], sampled_batch['name']
            mask_gt = sampled_batch['mask_gt'].cuda() # [B, 1, H, W]
            image_batch = image_batch.cuda() # [B, 1, H, W]

            unique_regions = torch.unique(mask_gt)
            num_regions = len(unique_regions)
            selected_regions = unique_regions[torch.randperm(num_regions)[:remain_num]] # 选择图像不变的SAM分割区域编号 占比mask_ratio
            region_mask = torch.isin(mask_gt, selected_regions).unsqueeze(1)

            mask_gt = mask_gt - 1 # 让背景区域编号从-1开始
            mask_gt = mask_gt.unsqueeze(1) # [B, 1, H, W]
            masked_image = torch.full_like(image_batch, -2)
            masked_image[region_mask] = image_batch[region_mask] # 图像中保持不变的区域

            batch_size = image_batch.shape[0]
            random_indices = torch.randint(0, batch_size, (batch_size,), device=image_batch.device)
            background_mask = (masked_image == -2)
            masked_image[background_mask] = image_batch[random_indices][:, :, :, :][background_mask] # 融合同一批batch中其他图像的区域

            A = image_batch   # 原始图像 [B, H ,W]
            A1 = masked_image # 融合图像 [B, H ,W]
            A1_embeddings, recon_img, seg_logits = model(A1) # 融合后图像输入模型, 输出分别为: 深层特征、重建图像、预测分割结果
            # print(A1_embeddings.shape) # *****记得查看一下, 应该是[B, C, H, W]***** [24, 512, 16, 16], 预估正确
            _, A_embeddings = model.encoder(A)               # 融合前图像输入模型

            # calculate training loss
            fcl_loss = compute_contrastive_loss(A_embeddings, A1_embeddings) # 对比损失
            recon_outputs = torch.tanh(recon_img)
            ifr_loss = mse_loss(recon_outputs, image_batch) # 重建损失
            skd_loss = dice_loss(inputs=seg_logits, target=mask_gt, onehot=True, softmax=True) # 分割dice损失
            loss = skd_loss + ifr_loss + alpha * fcl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            used_lr = optimizer.param_groups[0]['lr']
            # optimizer.zero_grad()
            # scaler.scale(loss).backward() # 缩放loss后反向传播，避免FP16反向传播过程中（在卷积、激活等中间层）的梯度下溢（FP16最小表示有限）
            # scaler.step(optimizer) # 梯度有效时更新模型参数
            # scaler.update() # 动态调整下一轮的梯度缩放倍数
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            values = [loss.item(), skd_loss.item(), ifr_loss.item(), fcl_loss.item()]
            epoch_losses += np.asarray(values) * image_batch.shape[0]
            epoch_samples += image_batch.shape[0]
            step_writer.writerow([epoch_num + 1, iter_num, *values, used_lr])
            progress.set_postfix(loss=f'{values[0]:.4f}', mean=f'{epoch_losses[0] / epoch_samples:.4f}',
                                 dice=f'{values[1]:.4f}', refresh=False)
            progress.update(1)
            # 后台每100步输出一行；CSV仍保存每一步。
            if not sys.stderr.isatty() and (iter_num % 100 == 0 or iter_num == 1 or iter_num >= max_iterations):
                logging.info('Epoch %d/%d iteration %d/%d loss: %.4f Dice: %.4f mean: %.4f',
                             epoch_num + 1, display_epochs, iter_num, max_iterations,
                             values[0], values[1], epoch_losses[0] / epoch_samples)

            if iter_num % 20000 == 0 or iter_num >= max_iterations: # 之前是每隔10000保存一次，并且还需要保存最后一次参数
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        progress.close()
        means = epoch_losses / epoch_samples
        epoch_writer.writerow([run_id, epoch_num + 1, iter_num, epoch_samples, *means])
        logging.info('Epoch %d/%d mean loss: %.4f Dice: %.4f recon: %.4f contrast: %.4f',
                     epoch_num + 1, display_epochs, *means)
        if iter_num >= max_iterations:
            break

    step_file.close()
    epoch_file.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./models/{}/{}".format(args.datasets, args.exp_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
