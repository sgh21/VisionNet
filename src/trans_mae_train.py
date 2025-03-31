import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import random
import math
import yaml
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


import timm.optim.optim_factory as optim_factory
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import models.TransMAE as transmae
from utils.custome_datasets import TransMAEDataset


def get_default_args():
    """获取默认参数"""
    parser = get_args_parser()
    default_args = parser.parse_args([])
    return default_args

def load_yaml_config(yaml_path):
    """加载yaml配置并与命令行参数合并"""
    # 获取默认参数
    args = get_default_args()
    
    # 读取yaml配置
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    # 更新参数
    if yaml_cfg:
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('CrossMAE training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--use_fix_template', action='store_true')
    # Model parameters
    parser.add_argument('--model', default='crossmae', type=str, metavar='MODEL')
    parser.add_argument('--embed_dim', default=1024, type=int) 
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--encoder_num_heads', default=16, type=int)
    parser.add_argument('--cross_num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--feature_dim', default=3, type=int)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--sigma', default=0.5, type=float)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--pair_downsample', type=float, default=1.0)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--log_dir', default='./output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    
    # 预训练权重
    parser.add_argument('--mae_pretrained', default='', type=str)
    parser.add_argument('--resume', default='', type=str)

    # Use yaml config
    parser.add_argument('--config', type=str, default='',
                        help='path to yaml config file')
    return parser
loss_norm = [2.5, 2.5, 1] # x,y,rz的归一化权重
def label_normalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label * weight
    return label

def label_denormalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label / weight
    return label

def create_transform_matrix(vector, intrinsic, img_size, scale=1.0):
    """
    从物理坐标系的平移和旋转参数生成图像变换矩阵
    
    Args:
        vector (torch.Tensor): 形状为[B, 3]的张量，包含[tx, ty, theta]
            - tx, ty: 平移量，单位为mm
            - theta: 旋转角度，单位为deg
        intrinsic (torch.Tensor): 形状为[B, 2]的张量，包含[fx, fy]
            - fx, fy: 内参矩阵的焦距参数，单位为mm/pixel
        scale (float): 缩放因子，用于调整变换幅度
            
    Returns:
        torch.Tensor: 形状为[B, 5]的变换矩阵参数 [theta, cx, cy, tx, ty]
    """
    B = vector.shape[0]
    device = vector.device
    
    # 提取参数
    tx_mm = vector[:, 0]  # x方向平移(mm)
    ty_mm = vector[:, 1]  # y方向平移(mm)
    theta_deg = vector[:, 2]  # 旋转角度(deg)
    
    # 提取内参
    fx , fy = intrinsic
    
    # 提取图片尺寸
    H, W = img_size

    # 将角度转换为弧度
    theta_rad = theta_deg * (torch.pi / 180.0)
    
    # 计算旋转中心(默认为图像中心)
    cx = torch.zeros_like(tx_mm)  # 默认为0，即图像中心
    cy = torch.zeros_like(ty_mm)  # 默认为0，即图像中心
    
    # 将物理平移量(mm)转换为归一化图像坐标
    # 归一化坐标范围为[-1, 1]，对应实际像素坐标[-W/2, W/2]和[-H/2, H/2]
    # 计算公式: tx_norm = tx_mm / (W/2 * fx)，其中W是图像宽度(像素)
    # 由于我们使用归一化坐标，W/2对应于1.0，所以公式简化为 tx_norm = tx_mm / fx
    tx_norm = tx_mm / (fx * W/2) # 图像x方向的归一化平移量
    ty_norm = ty_mm / (fy * H/2) # 图像y方向的归一化平移量
    
    # 构建完整的变换矩阵参数 [theta, cx, cy, tx, ty]
    transform_matrix = torch.stack([theta_rad, cx, cy, tx_norm, ty_norm], dim=1)
    
    return transform_matrix.to(device=device)

def create_pred_vector(T, intrinsic, img_size):
    """
    从变换矩阵参数中提取物理坐标系的平移和旋转参数
    
    Args:
        T (torch.Tensor): 形状为[B, 5]的变换矩阵参数 [theta, cx, cy, tx, ty]
        intrinsic (torch.Tensor): 形状为[B, 2]的张量，包含[fx, fy]
            - fx, fy: 内参矩阵的焦距参数，单位为mm/pixel
            
    Returns:
        torch.Tensor: 形状为[B, 3]的张量，包含[tx, ty, theta]
            - tx, ty: 平移量，单位为mm
            - theta: 旋转角度，单位为deg
    """
    B = T.shape[0]
    device = T.device
    
    # 提取变换矩阵参数
    theta_rad = T[:,0]
    tx_norm = T[:, 3]
    ty_norm = T[:, 4]
    
    # 提取内参
    fx , fy = intrinsic
    
    # 提取图片尺寸
    H , W = img_size
    
    # 转换为角度
    theta_deg = theta_rad * (180.0 / torch.pi)
    
    # 将归一化平移量转换回物理平移量(mm)
    tx_mm = tx_norm * fx * W/2
    ty_mm = ty_norm * fy * H/2
    
    # 构建最终向量 [tx, ty, theta]
    pred_vector = torch.stack([tx_mm, ty_mm, theta_deg], dim=1)
    
    return pred_vector.to(device=device)

def calculate_dim_mae(pred, target):
    mae_x = torch.mean(torch.abs(pred[:,0] - target[:,0]))
    mae_y = torch.mean(torch.abs(pred[:,1] - target[:,1]))
    mae_rz = torch.mean(torch.abs(pred[:,2] - target[:,2]))
    return mae_x, mae_y, mae_rz

def train_one_epoch(model: torch.nn.Module, data_loader, optimizer: torch.optim.Optimizer, criterion, 
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train()  # 设置模型为训练模式
    metric_logger = misc.MetricLogger(delimiter="  ")  # 用于记录和打印指标
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 学习率
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # 打印频率

    accum_iter = args.accum_iter  # 梯度累积

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (img1, img2, label1, label2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 每累积一定步数调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label1 = label1.to(device, non_blocking=True)
        label2 = label2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 混合精度训练
            T, trans_diff_loss, img2_trans = model(img1, img2, args.mask_ratio, sigma = args.sigma)  # 模型输出
            
            loss = trans_diff_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter  # 梯度累积
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  # 梯度归零

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)  # 更新损失

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)  # 更新学习率

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # 计算全局损失
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()  # 跨进程同步
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, data_loader, criterion, device, epoch, log_writer=None, args=None):
    model.eval()  # 设置模型为评估模式
    metric_logger = misc.MetricLogger(delimiter="  ")  # 用于记录和打印指标
    header = 'Test:'

    total_loss = 0
    total_x_mae = 0
    total_y_mae = 0
    total_rz_mae = 0
    # * :图像的反归一化和可视化
    # 图像反归一化的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def denormalize_image(img):
        """将归一化的图像反归一化回[0, 1]范围"""
        img = img * std + mean  # 反归一化
        img = torch.clamp(img, 0, 1)  # 裁剪到[0, 1]范围
        return img
    
    with torch.no_grad():  # 在验证过程中不计算梯度
        for batch_idx, (img1, img2, label1, label2) in enumerate(metric_logger.log_every(data_loader, 20, header)):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            batch_size = img1.size(0)
            
            with torch.cuda.amp.autocast():  # 混合精度验证
                T, trans_diff_loss, img2_trans = model(img1, img2, args.mask_ratio, sigma = args.sigma)
                delta_label = label2 - label1
                
               
                loss = trans_diff_loss
                pred = create_pred_vector(T, intrinsic=[-0.0206*2.5,-0.0207*2.5],img_size=[224, 224])
                mae_x, mae_y, mae_rz = calculate_dim_mae(pred, delta_label)
                
                total_x_mae += mae_x.item() * batch_size
                total_y_mae += mae_y.item() * batch_size
                total_rz_mae += mae_rz.item() * batch_size
                total_loss += loss.item() * batch_size

            metric_logger.update(loss=loss.item())  # 更新损失
            # 每个epoch只可视化一个批次的图像
            if log_writer is not None and batch_idx == 0:
                # 选择批次中的前min(8, batch_size)个样本进行可视化
                n_vis = min(3, batch_size)
                
                # 反归一化图像
                img1_vis = denormalize_image(img1[:n_vis])
                img2_vis = denormalize_image(img2[:n_vis])
                img2_trans_vis = denormalize_image(img2_trans[:n_vis])
                
                # 计算差异图像
                diff_vis = torch.abs(img2_trans_vis - img1_vis)  # 绝对差异
                
                # 创建图像网格
                grid_img1 = torchvision.utils.make_grid(img1_vis, nrow=n_vis, padding=2)
                grid_img2 = torchvision.utils.make_grid(img2_vis, nrow=n_vis, padding=2)
                grid_trans = torchvision.utils.make_grid(img2_trans_vis, nrow=n_vis, padding=2)
                grid_diff = torchvision.utils.make_grid(diff_vis, nrow=n_vis, padding=2)
                
                # 添加到tensorboard
                log_writer.add_image('val/img1_source', grid_img1, epoch)
                log_writer.add_image('val/img2_target', grid_img2, epoch)
                log_writer.add_image('val/img2_trans', grid_trans, epoch)
                log_writer.add_image('val/trans_diff', grid_diff, epoch)
                
                # 额外添加一个合并的可视化图像，便于比较
                grid_combined = torch.cat([grid_img1, grid_img2, grid_trans, grid_diff], dim=1)
                log_writer.add_image('val/comparison', grid_combined, epoch)
    # 计算平均值
    num_samples = len(data_loader.dataset)
    avg_loss = total_loss / num_samples
    avg_x_mae = total_x_mae / num_samples
    avg_y_mae = total_y_mae / num_samples
    avg_rz_mae = total_rz_mae / num_samples
    
    # 记录到tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/loss', avg_loss, epoch)
        log_writer.add_scalar('val/mae_x_mm', avg_x_mae, epoch)
        log_writer.add_scalar('val/mae_y_mm', avg_y_mae, epoch)
        log_writer.add_scalar('val/mae_rz_deg', avg_rz_mae, epoch)

    metric_logger.update(mae_x=avg_x_mae)
    metric_logger.update(mae_y=avg_y_mae)
    metric_logger.update(mae_rz=avg_rz_mae)

    # 同步并打印结果
    metric_logger.synchronize_between_processes()
    print('* Avg loss {:.3f}, MAE x {:.2f}mm, y {:.2f}mm, rz {:.2f}deg'
          .format(metric_logger.loss.global_avg,
                 metric_logger.mae_x.global_avg,
                 metric_logger.mae_y.global_avg,
                 metric_logger.mae_rz.global_avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建数据增强
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train =TransMAEDataset(args, is_train=True, transform=transform_train,  use_fix_template=args.use_fix_template)
    dataset_val = TransMAEDataset(args, is_train=False, transform=transform_val,  use_fix_template=args.use_fix_template)

    # 使用普通随机采样器
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 创建模型
    model = transmae.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        pretrained_path=args.mae_pretrained,
        qkv_bias=args.qkv_bias,
    )

    model.to(device)

    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    # 使用AdamW优化器
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.SmoothL1Loss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, criterion, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        val_stats = validate(
            model, data_loader_val,
            criterion, device, epoch, log_writer,
            args=args
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model,model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,save_full_state=False
            )

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config:
        args = load_yaml_config(args.config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
