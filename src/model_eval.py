import argparse
import yaml
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

# 导入你项目中的模块
import utils.misc as misc
import models.TransMAE as transmae
from utils.custome_datasets import EvalMAEDataset
from config import INTRINSIC

# --- 从训练脚本中复用的辅助函数 ---

def create_pred_vector(T, intrinsic, img_size):
    """
    从变换矩阵参数中提取物理坐标系的平移和旋转参数
    """
    B = T.shape[0]
    device = T.device
    theta_rad, tx_norm, ty_norm = T[:, 0], T[:, 1], T[:, 2]
    fx, fy, flag = intrinsic
    H, W = img_size
    theta_deg = theta_rad * (180.0 / torch.pi) * flag
    tx_mm = tx_norm * fx * W / 2
    ty_mm = ty_norm * fy * H / 2
    pred_vector = torch.stack([tx_mm, ty_mm, theta_deg], dim=1)
    return pred_vector.to(device=device)

def calculate_dim_mae(pred, target):
    """
    计算每个维度的平均绝对误差 (MAE)
    """
    mae_x = torch.mean(torch.abs(pred[:, 0] - target[:, 0]))
    mae_y = torch.mean(torch.abs(pred[:, 1] - target[:, 1]))
    mae_rz = torch.mean(torch.abs(pred[:, 2] - target[:, 2]))
    return mae_x.item(), mae_y.item(), mae_rz.item()

# --- 核心功能函数 ---

def get_args_parser():
    """
    定义命令行参数
    """
    parser = argparse.ArgumentParser('TransMAE Validation', add_help=False)
    # 必选参数
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint (.pth file)')
    
    # 可选参数，如果未在YAML中定义，可以在此覆盖
    parser.add_argument('--batch_size', type=int, help='Override batch size in config')
    parser.add_argument('--num_workers', type=int, help='Override num_workers in config')
    parser.add_argument('--device', default='cuda', help='Device to use for inference')
    
    return parser

def load_config_and_merge_args(cmd_args):
    """
    加载YAML配置并与命令行参数合并
    """
    with open(cmd_args.config, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.safe_load(f)

    # 将YAML配置转换为argparse.Namespace对象，方便访问
    config = argparse.Namespace(**yaml_cfg)

    # 命令行参数优先级更高，覆盖YAML中的配置
    for key, value in vars(cmd_args).items():
        if value is not None:
            setattr(config, key, value)
            
    return config

def load_model(args):
    """
    根据配置初始化模型并加载权重
    这是一个独立的接口，方便未来修改
    """
    print(f"Initializing model: {args.model}")
    # 使用和训练时相同的逻辑创建模型实例
    model = transmae.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        qkv_bias=args.qkv_bias,
        illumination_alignment=args.illumination_alignment,
        use_chamfer_dist=args.use_chamfer_dist,
        chamfer_dist_type=args.chamfer_dist_type,
        use_mask_weight=args.use_mask_weight,
        pool_mode=args.pool_mode,
        mask_size=args.high_res_size,
        use_ssim_loss=args.use_ssim_loss,
    )
    
    print(f"Loading weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # 根据你的保存方式选择加载
    # 如果保存的是 model.state_dict()
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    # 如果直接保存的 state_dict
    else:
        state_dict = checkpoint

    # 加载权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weight loading message: {msg}")
    
    return model

def run_validation(args):
    """
    执行验证流程
    """
    device = torch.device(args.device)
    
    # 1. 加载模型
    model = load_model(args)
    model.to(device)
    model.eval() # 设置为评估模式

    
    # 注意：is_train=False 表示使用验证集
    # !: 要考虑加载器的实例化和模型输入是否匹配
    dataset_val = EvalMAEDataset(args, use_fix_template=args.use_fix_template)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Validation dataset size: {len(dataset_val)}")

    # 3. 执行推理和评估
    total_x_mae, total_y_mae, total_rz_mae = 0, 0, 0
    num_samples = 0

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validating:'

    with torch.no_grad(): # 关闭梯度计算
        # !：输入模型的数据也要考虑一致性
        for (img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2) in metric_logger.log_every(data_loader_val, 20, header):
            img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
            high_res_img1, high_res_img2 = high_res_img1.to(device, non_blocking=True), high_res_img2.to(device, non_blocking=True)
            touch_img_mask1, touch_img_mask2 = touch_img_mask1.to(device, non_blocking=True), touch_img_mask2.to(device, non_blocking=True)
            label1, label2 = label1.to(device, non_blocking=True), label2.to(device, non_blocking=True)
            
            batch_size = img1.shape[0]

            # 模型前向传播
            pred, _, _, _ = model(
                img1, img2,
                high_res_x1=high_res_img1,
                high_res_x2=high_res_img2,
                method=args.method,
                mask1=touch_img_mask1,
                mask2=touch_img_mask2,
                mask_ratio=args.mask_ratio,
                sigma=args.sigma,
                CXCY=args.CXCY
            )
            
            # 计算真实标签差值
            delta_label = label2 - label1
            
            # 将模型输出转换为物理单位
            pred_vector = create_pred_vector(pred, intrinsic=INTRINSIC, img_size=[args.high_res_size, args.high_res_size])
            
            # 计算误差
            mae_x, mae_y, mae_rz = calculate_dim_mae(pred_vector, delta_label)
            
            # 累加误差
            total_x_mae += mae_x * batch_size
            total_y_mae += mae_y * batch_size
            total_rz_mae += mae_rz * batch_size
            num_samples += batch_size

    # 4. 计算并打印最终平均指标
    avg_x_mae = total_x_mae / num_samples
    avg_y_mae = total_y_mae / num_samples
    avg_rz_mae = total_rz_mae / num_samples

    print("\n--- Validation Finished ---")
    print(f"Total samples evaluated: {num_samples}")
    print(f"Average MAE X:  {avg_x_mae:.4f} mm")
    print(f"Average MAE Y:  {avg_y_mae:.4f} mm")
    print(f"Average MAE RZ: {avg_rz_mae:.4f} deg")
    print("---------------------------\n")


if __name__ == '__main__':
    parser = get_args_parser()
    cmd_args = parser.parse_args()
    
    # 加载配置
    config_args = load_config_and_merge_args(cmd_args)
    
    # 打印最终生效的配置
    print("Effective configuration:")
    print("{}".format(config_args).replace(', ', ',\n'))
    
    # 运行验证
    run_validation(config_args)