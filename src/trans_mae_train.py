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
from config import INTRINSIC

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
    # 添加一个固定长度为2的列表参数
    parser.add_argument('--CXCY', type=float, nargs=2, default=[-0.0102543, -0.0334525], 
                    help='旋转中心坐标 [cx, cy]，范围(-1,1)')
    # 添加 lambda 相关参数
    parser.add_argument('--lambda_start', type=float, default=0.1,
                        help='lambda 权重的初始值')
    parser.add_argument('--lambda_end', type=float, default=0.5,
                        help='lambda 权重的最终值')
    parser.add_argument('--lambda_warmup_epochs', type=int, default=40,
                    help='lambda 权重从初始值增长到最终值所需的 epoch 数')
    parser.add_argument('--high_res_size', default=560, type=int)
    # 添加beta参数，控制pred_loss和trans_diff_loss的权重
    parser.add_argument('--beta', type=float, default=1.0,
                        help='权重参数，控制pred_loss和trans_diff_loss*beta的平衡')
    parser.add_argument('--intensity_scaling', type=list, default=[0.1, 0.6, 0.8, 1.0])
    parser.add_argument('--edge_enhancement', type=float, default=1.5)
    parser.add_argument('--illumination_alignment', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--use_chamfer_dist', type=bool, default=True)
    parser.add_argument('--chamfer_dist_type', type=str, default='L2', choices=['L1', 'L2'])
    parser.add_argument('--use_mask_weight', action='store_true')
    parser.add_argument('--pool_mode', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--sample_size', type=int, default=256)
    parser.add_argument('--use_ssim_loss', action='store_true')
    
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
    parser.add_argument('--sigma', type=float, nargs=2, default=[0.5, 0.5],)
    parser.add_argument('--method', type=str, default='touch_mask', choices=['gaussian', 'touch_mask', 'rgb_mask'], help='选择模型的训练方法')
    parser.add_argument('--touch_img_template_path', type=str, default='')
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
def get_current_lambda(progress, args):
    """
    根据训练进度计算当前 lambda 值
    
    Args:
        progress (float): 当前训练进度 (0.0 ~ 1.0)
        args: 参数配置
        
    Returns:
        float: 当前的 lambda 值
    """
    if progress >= args.lambda_warmup_epochs / args.epochs:
        return args.lambda_end
    
    # 线性增长
    normalized_progress = progress * args.epochs / args.lambda_warmup_epochs
    current_lambda = args.lambda_start + normalized_progress * (args.lambda_end - args.lambda_start)
    return current_lambda

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
        torch.Tensor: 形状为[B, 3]的变换矩阵参数 [theta, tx, ty]
    """
    B = vector.shape[0]
    device = vector.device
    
    # 提取参数
    tx_mm = vector[:, 0]  # x方向平移(mm)
    ty_mm = vector[:, 1]  # y方向平移(mm)
    theta_deg = vector[:, 2]  # 旋转角度(deg)
    
    # 提取内参
    fx , fy, flag = intrinsic
    
    # 提取图片尺寸
    H, W = img_size

    # 将角度转换为弧度
    theta_rad = theta_deg * (torch.pi / 180.0) * flag
    
    # 计算旋转中心(默认为图像中心)
    # cx = torch.zeros_like(tx_mm)  # 默认为0，即图像中心
    # cy = torch.zeros_like(ty_mm)  # 默认为0，即图像中心
    
    # 将物理平移量(mm)转换为归一化图像坐标
    # 归一化坐标范围为[-1, 1]，对应实际像素坐标[-W/2, W/2]和[-H/2, H/2]
    # 计算公式: tx_norm = tx_mm / (W/2 * fx)，其中W是图像宽度(像素)
    # 由于我们使用归一化坐标，W/2对应于1.0，所以公式简化为 tx_norm = tx_mm / fx
    tx_norm = tx_mm / (fx * W/2) # 图像x方向的归一化平移量
    ty_norm = ty_mm / (fy * H/2) # 图像y方向的归一化平移量
    
    # 构建完整的变换矩阵参数 [theta, tx, ty]
    transform_matrix = torch.stack([theta_rad, tx_norm, ty_norm], dim=1)
    
    return transform_matrix.to(device=device)

def create_pred_vector(T, intrinsic, img_size):
    """
    从变换矩阵参数中提取物理坐标系的平移和旋转参数
    
    Args:
        T (torch.Tensor): 形状为[B, 3]的变换矩阵参数 [theta, tx, ty]
        intrinsic (torch.Tensor): 形状为[B, 2]的张量，包含[fx, fy, flag]
            - fx, fy: 内参矩阵的焦距参数，单位为mm/pixel
            - flag: 旋转是否反向
            
    Returns:
        torch.Tensor: 形状为[B, 3]的张量，包含[tx, ty, theta]
            - tx, ty: 平移量，单位为mm
            - theta: 旋转角度，单位为deg
    """
    B = T.shape[0]
    device = T.device
    
    # 提取变换矩阵参数
    theta_rad = T[:,0]
    tx_norm = T[:, 1]
    ty_norm = T[:, 2]
    
    # 提取内参
    fx , fy, flag = intrinsic
    
    # 提取图片尺寸
    H , W = img_size
    
    # 转换为角度
    theta_deg = theta_rad * (180.0 / torch.pi) * flag
    
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

    for data_iter_step, (img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 计算当前的训练进度
        iter_progress = data_iter_step / len(data_loader) + epoch
        overall_progress = iter_progress / args.epochs
        # 每累积一定步数调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        high_res_img1 = high_res_img1.to(device, non_blocking=True)
        high_res_img2 = high_res_img2.to(device, non_blocking=True)
        touch_img_mask1 = touch_img_mask1.to(device, non_blocking=True)
        touch_img_mask2 = touch_img_mask2.to(device, non_blocking=True)
        sample_contour1 = sample_contour1.to(device, non_blocking=True)
        sample_contour2 = sample_contour2.to(device, non_blocking=True)
        label1 = label1.to(device, non_blocking=True)
        label2 = label2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 混合精度训练
            pred, trans_diff_loss, chamfer_loss, img2_trans = model(
                img1, img2, 
                high_res_x1=high_res_img1, 
                high_res_x2=high_res_img2,
                method = args.method,
                mask1 =touch_img_mask1,
                mask2 =touch_img_mask2,
                sample_contour1 = sample_contour1,
                sample_contour2 = sample_contour2,
                mask_ratio=args.mask_ratio, 
                sigma=args.sigma, 
                CXCY=args.CXCY
            )

            
            delta_label = label2 - label1  # 计算标签的差值 （B,3）

            pred_vector = create_pred_vector(pred, intrinsic=INTRINSIC,img_size=[560, 560])  # 将预测的变换矩阵参数转换为物理坐标系的平移和旋转参数
            delta_label = label_normalize(delta_label, weight=loss_norm)  # 对标签进行归一化
            pred_vector = label_normalize(pred_vector, weight=loss_norm)
            pred_loss = criterion(pred_vector, delta_label)  # 计算损失
            # 计算总损失
            current_lambda = get_current_lambda(overall_progress, args)
            loss = (1-current_lambda)*pred_loss + args.beta*current_lambda*trans_diff_loss + args.gamma*chamfer_loss

        loss_value = loss.item()
        pred_loss_value = pred_loss.item()
        trans_diff_loss_value = trans_diff_loss.item()
        chamfer_loss_value = chamfer_loss.item()

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
        metric_logger.update(pred_loss=pred_loss_value)  # 更新预测损失
        metric_logger.update(trans_diff_loss=trans_diff_loss_value)  # 更新变换损失

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)  # 更新学习率

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # 计算全局损失
        pred_loss_reduce = misc.all_reduce_mean(pred_loss_value)  # 计算全局预测损失
        trans_diff_loss_reduce = misc.all_reduce_mean(trans_diff_loss_value)  # 计算全局变换损失
        chamfer_loss_reduce = misc.all_reduce_mean(chamfer_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/lambda', current_lambda, epoch_1000x)
            log_writer.add_scalar('train/train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/pred_loss', pred_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train/trans_diff_loss', trans_diff_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train/chamfer_loss', chamfer_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()  # 跨进程同步
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate(model, data_loader, criterion, device, epoch, log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_loss = 0
    total_pred_loss = 0
    total_trans_diff_loss = 0
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
    
    # 计算当前epoch对应的总体进度
    overall_progress = epoch / args.epochs
    current_lambda = get_current_lambda(overall_progress, args)
    
    with torch.no_grad():
        for batch_idx, (img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2) in enumerate(metric_logger.log_every(data_loader, 20, header)):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            high_res_img1 = high_res_img1.to(device, non_blocking=True)
            high_res_img2 = high_res_img2.to(device, non_blocking=True)
            touch_img_mask1 = touch_img_mask1.to(device, non_blocking=True)
            touch_img_mask2 = touch_img_mask2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            batch_size = img1.size(0)
            
            with torch.cuda.amp.autocast():
                pred, trans_diff_loss, chamfer_loss, img2_trans = model(
                    img1, img2, 
                    high_res_x1=high_res_img1, 
                    high_res_x2=high_res_img2,
                    method = args.method,
                    mask1=touch_img_mask1,
                    mask2=touch_img_mask2,
                    mask_ratio=args.mask_ratio, 
                    sigma=args.sigma, 
                    CXCY=args.CXCY
                )
                
                delta_label = label2 - label1
                
                pred_vector = create_pred_vector(pred, intrinsic=INTRINSIC,img_size=[560, 560])
                mae_x, mae_y, mae_rz = calculate_dim_mae(pred_vector, delta_label)
                
                total_x_mae += mae_x.item() * batch_size
                total_y_mae += mae_y.item() * batch_size
                total_rz_mae += mae_rz.item() * batch_size

                delta_label = label_normalize(delta_label, weight=loss_norm)  # 对标签进行归一化
                pred_vector = label_normalize(pred_vector, weight=loss_norm)
                pred_loss = criterion(pred_vector, delta_label)  # 计算损失
                
                loss = (1-current_lambda)*pred_loss + args.beta*current_lambda*trans_diff_loss
                total_loss += loss.item() * batch_size
                total_pred_loss += pred_loss.item() * batch_size
                total_trans_diff_loss += trans_diff_loss.item() * batch_size

            metric_logger.update(loss=loss.item())  # 更新损失
            metric_logger.update(trans_diff_loss=trans_diff_loss.item())  # 更新变换损失
            
            # 每个epoch可视化第一个批次的图像和权重图
            if log_writer is not None and batch_idx == 0:
                n_vis = min(4, batch_size)
                
                # 可视化原始图像和变换后的图像
                img1_vis = denormalize_image(high_res_img1[:n_vis])
                img2_vis = denormalize_image(high_res_img2[:n_vis])
                img2_trans_vis = denormalize_image(img2_trans[:n_vis])
                
                # 计算差异图像
                diff_vis = torch.abs(img2_trans_vis - img1_vis)
                
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
                
                # 可视化权重图
                for i in range(n_vis):
                    # 获取当前样本的变换参数
                    sample_params = pred[i:i+1].detach()
                    
                    # 如果使用触摸掩码方法
                    if args.method == 'touch_mask' or args.method == 'rgb_mask':
                        # 获取当前样本的掩码
                        mask1 = touch_img_mask1[i:i+1]
                        mask2 = touch_img_mask2[i:i+1]
                        
                        # 对掩码应用变换
                        transformed_mask2 = model.forward_transfer(mask2, sample_params, CXCY=args.CXCY)
                        
                        # 合并掩码
                        merged_mask = torch.max(mask1, transformed_mask2)
                        
                        # 归一化权重图
                        H, W = merged_mask.shape[2:]
                        pixel_count = H * W
                        current_sum = merged_mask.sum(dim=(2, 3), keepdim=True)
                        scale_factor = pixel_count / (current_sum + 1e-8)
                        normalized_mask = merged_mask * scale_factor
                        
                        # 将权重图可视化为热力图
                        mask_vis = normalized_mask[0, 0].cpu()
                        mask_norm = (mask_vis - mask_vis.min()) / (mask_vis.max() - mask_vis.min() + 1e-8)
                        
                        # 创建彩色热力图
                        mask_heatmap = torch.zeros(3, mask_norm.shape[0], mask_norm.shape[1])
                        mask_heatmap[0] = mask_norm  # 红色通道表示权重
                        
                        # 记录到TensorBoard
                        log_writer.add_image(f'val/sample{i}/weight_map', mask_heatmap, epoch)
                        
                        # 将权重图叠加到原始图像上
                        img1_with_weight = img1_vis[i].clone()
                        
                        # 上采样权重图以匹配图像尺寸
                        if mask_norm.shape != img1_with_weight.shape[1:]:
                            mask_resized = torch.nn.functional.interpolate(
                                mask_norm.unsqueeze(0).unsqueeze(0),
                                size=img1_with_weight.shape[1:],
                                mode='bilinear'
                            ).squeeze(0).squeeze(0).to(img1_with_weight.device)
                        else:
                            mask_resized = mask_norm.to(img1_with_weight.device)
                        
                        # 叠加权重图（红色通道）
                        alpha = 0.7  # 透明度
                        img1_with_weight[0] = img1_with_weight[0] * (1-alpha) + mask_resized * alpha
                        
                        # 添加到TensorBoard
                        log_writer.add_image(f'val/sample{i}/img1_with_weight', img1_with_weight, epoch)
                    
                    # 如果使用高斯方法（保留原有逻辑）
                    elif args.method == 'gaussian':
                        # 基础权重图
                        base_weight = model.base_weight_map[0, 0].cpu()
                        
                        # 将基础权重图转为批次
                        batch_weight_map = model.base_weight_map.expand(1, 1, 
                                                                        model.base_weight_map.shape[2], 
                                                                        model.base_weight_map.shape[3])
                        
                        # 应用变换得到变换后的权重图
                        transformed_weight = model.forward_transfer(batch_weight_map.to(device), 
                                                                   sample_params, 
                                                                   CXCY=args.CXCY)
                        
                        # 归一化权重图以便更好地可视化
                        base_weight_norm = (base_weight - base_weight.min()) / (base_weight.max() - base_weight.min() + 1e-8)
                        trans_weight_norm = (transformed_weight[0, 0].cpu() - transformed_weight[0, 0].min().cpu()) / (transformed_weight[0, 0].max().cpu() - transformed_weight[0, 0].min().cpu() + 1e-8)
                        
                        # 创建彩色热力图
                        base_weight_heatmap = torch.zeros(3, base_weight.shape[0], base_weight.shape[1])
                        trans_weight_heatmap = torch.zeros(3, base_weight.shape[0], base_weight.shape[1])
                        
                        # 使用简单的热力图映射：红色通道表示权重
                        base_weight_heatmap[0] = base_weight_norm
                        trans_weight_heatmap[0] = trans_weight_norm
                        
                        # 记录到TensorBoard
                        log_writer.add_image(f'val/sample{i}/base_weight', base_weight_heatmap, epoch)
                        log_writer.add_image(f'val/sample{i}/transformed_weight', trans_weight_heatmap, epoch)
                        
                        # 将权重图叠加到原始图像上
                        img1_with_weight = img1_vis[i].clone()
                        img2_trans_with_weight = img2_trans_vis[i].clone()
                        
                        # 上采样权重图以匹配图像尺寸
                        if base_weight.shape != img1_with_weight.shape[1:]:
                            base_weight_resized = torch.nn.functional.interpolate(
                                base_weight_norm.unsqueeze(0).unsqueeze(0).to(img1_with_weight.device),
                                size=img1_with_weight.shape[1:],
                                mode='bilinear'
                            ).squeeze(0).squeeze(0)
                            
                            trans_weight_resized = torch.nn.functional.interpolate(
                                trans_weight_norm.unsqueeze(0).unsqueeze(0).to(img2_trans_with_weight.device),
                                size=img2_trans_with_weight.shape[1:],
                                mode='bilinear'
                            ).squeeze(0).squeeze(0)
                        else:
                            base_weight_resized = base_weight_norm.to(img1_with_weight.device)
                            trans_weight_resized = trans_weight_norm.to(img2_trans_with_weight.device)
                        
                        # 叠加权重图（红色通道）
                        alpha = 0.7  # 透明度
                        img1_with_weight[0] = img1_with_weight[0] * (1-alpha) + base_weight_resized * alpha
                        img2_trans_with_weight[0] = img2_trans_with_weight[0] * (1-alpha) + trans_weight_resized * alpha
                        
                        # 添加到TensorBoard
                        log_writer.add_image(f'val/sample{i}/img1_with_weight', img1_with_weight, epoch)
                        log_writer.add_image(f'val/sample{i}/img2_trans_with_weight', img2_trans_with_weight, epoch)
                
                # 额外添加一个合并的可视化图像，便于比较
                grid_combined = torch.cat([grid_img1, grid_img2, grid_trans, grid_diff], dim=1)
                log_writer.add_image('val/comparison', grid_combined, epoch)
                
    # 计算平均值
    num_samples = len(data_loader.dataset)
    avg_loss = total_loss / num_samples
    avg_pred_loss = total_pred_loss / num_samples
    avg_trans_diff_loss = total_trans_diff_loss / num_samples
    avg_x_mae = total_x_mae / num_samples
    avg_y_mae = total_y_mae / num_samples
    avg_rz_mae = total_rz_mae / num_samples
    
    # 记录到tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/loss', avg_loss, epoch)
        log_writer.add_scalar('val/pred_loss', avg_pred_loss, epoch)
        log_writer.add_scalar('val/trans_diff_loss', avg_trans_diff_loss, epoch)
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
        illumination_alignment=args.illumination_alignment,
        use_chamfer_dist=args.use_chamfer_dist,
        chamfer_dist_type=args.chamfer_dist_type,
        use_mask_weight=args.use_mask_weight,
        pool_mode=args.pool_mode,
        mask_size=args.high_res_size,
        use_ssim_loss=args.use_ssim_loss,
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
