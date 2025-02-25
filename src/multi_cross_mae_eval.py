import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import models.TestMultiCrossMAE as multicrossmae
from pathlib import Path
from utils.custome_datasets import MultiCrossMAEDataset
from utils.VisionUtils import add_radial_noise, visualize_results_rgb_touch
from utils.DataAnalysis import plot_error_distribution,data_statistics

def get_default_args():
    """获取默认参数"""
    parser = get_args_parser()
    default_args = parser.parse_args([])
    return default_args

def load_yaml_config(yaml_path):
    """加载yaml配置并与命令行参数合并"""
    args = get_default_args()
    
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    if yaml_cfg:
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('MultiCrossMAE eval', add_help=False)
    # 基本参数
    parser.add_argument('--config', type=str, default='',
                        help='path to yaml config file')
    parser.add_argument('--weights', default='', type=str,
                        help='checkpoint path')
    
    # 模型参数
    parser.add_argument('--model', default='multicrossmae_vit_base', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--embed_dim', default=1024, type=int)
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--encoder_num_heads', default=16, type=int)
    parser.add_argument('--cross_num_heads', default=16, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--feature_dim', default=3, type=int)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--mask_ratio', default=1, type=float)
    
    # 评估参数
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./eval_results')
    parser.add_argument('--pair_downsample', default=1.0, type=float)
    parser.add_argument('--noise_level', default=0.1, type=float,
                    help='Maximum noise level for RGB images')
    parser.add_argument('--curve_type', default='gaussion', type=str)
    parser.add_argument('--optimize_type', default='mse', type=str)

    return parser


def calculate_dim_mae(pred, target):
    mae_x = pred[:,0] - target[:,0]
    mae_y = pred[:,1] - target[:,1]
    mae_rz = pred[:,2] - target[:,2]
    return mae_x, mae_y, mae_rz

def main(args):

    print(f"Loading checkpoint from {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # 创建模型
    model = multicrossmae.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        qkv_bias=args.qkv_bias,
    )
    
    # 加载权重
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    print(msg)
    
    model = model.to(args.device)
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    dataset =MultiCrossMAEDataset(args, is_train = False,rgb_transform=transform, touch_transform=transform, is_eval=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建结果保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录评估结果
    all_maes_x = []
    all_maes_y = []
    all_maes_rz = []
    # 添加进度条
    pbar = tqdm(dataloader, desc='Evaluating', ncols=100)
    with torch.no_grad():
        for batch_idx, (rgb_img1, rgb_img2, touch_img1, touch_img2,label1, label2, img1_name,img2_name) in enumerate(dataloader):
            rgb_img1, rgb_img2 = rgb_img1.to(args.device), rgb_img2.to(args.device)
            touch_img1, touch_img2 = touch_img1.to(args.device), touch_img2.to(args.device)
            label1, label2 = label1.to(args.device), label2.to(args.device)
            
            rgb_img1 = add_radial_noise(rgb_img1, args.noise_level)
            rgb_img2 = add_radial_noise(rgb_img2, args.noise_level)
            # 预测
            pred = model(rgb_img1, rgb_img2, touch_img1, touch_img2, mask_ratio=args.mask_ratio, mask_rgb=True)
            delta_label = label2 - label1
            
            # 计算MAE
            mae_x, mae_y, mae_rz = calculate_dim_mae(pred, delta_label)
            # 存储每个样本的MAE
            all_maes_x.extend(mae_x.cpu().tolist())
            all_maes_y.extend(mae_y.cpu().tolist())
            all_maes_rz.extend(mae_rz.cpu().tolist())
            
            # 更新进度条描述
            pbar.update(1)
            # 显示当前batch的平均值
            batch_mae_x = torch.abs(mae_x).mean().item()
            batch_mae_y = torch.abs(mae_y).mean().item()
            batch_mae_rz = torch.abs(mae_rz).mean().item()
            
            pbar.set_postfix({
                'MAE_X': f'{batch_mae_x:.4f}',
                'MAE_Y': f'{batch_mae_y:.4f}',
                'MAE_Rz': f'{batch_mae_rz:.4f}'
            })

            # 可视化结果
            for i in range(rgb_img1.size(0)):
                save_path = os.path.join(args.output_dir, f'pair_{batch_idx}_{i}.png')
                
                visualize_results_rgb_touch(
                    rgb_img1[i].cpu(), 
                    rgb_img2[i].cpu(),
                    touch_img1[i].cpu(),
                    touch_img2[i].cpu(),
                    pred[i].cpu().numpy(),
                    delta_label[i].cpu().numpy(),
                    save_path
                )
    
    # 计算并打印平均MAE
    all_maes = np.stack([all_maes_x, all_maes_y, all_maes_rz], axis=1)  # [num_samples, 3]
    avg_maes_abs = np.mean(np.abs(all_maes), axis=0)

    print(f'Average MAE:')
    print(f'MAE_X: {avg_maes_abs[0]:.4f} mm')
    print(f'MAE_Y: {avg_maes_abs[1]:.4f} mm')
    print(f'MAE_Rz: {avg_maes_abs[2]:.4f} deg')

    truncation_error_x = data_statistics(all_maes_x, error_cut=0.9973, dtype=args.curve_type,optimize_type=args.optimize_type)
    truncation_error_y = data_statistics(all_maes_y, error_cut=0.9973, dtype=args.curve_type,optimize_type=args.optimize_type)
    truncation_error_rz = data_statistics(all_maes_rz, error_cut=0.9973, dtype=args.curve_type,optimize_type=args.optimize_type)
    print(f'mu + 3 * std(X):{truncation_error_x:.4f} mm')
    print(f'mu + 3 * std(Y):{truncation_error_y:.4f} mm')
    print(f'mu + 3 * std(Rz):{truncation_error_rz:.4f} deg')

    # 绘制误差分布图
    dist_plot_path = os.path.join(args.output_dir, f'error_distribution_{args.curve_type}.png')
    plot_error_distribution(all_maes_x, all_maes_y, all_maes_rz, dist_plot_path,dtype=args.curve_type,optimize_type=args.optimize_type)
    pbar.close()
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    if args.config:
        args = load_yaml_config(args.config)
        
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)