import os
import argparse
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import datetime
from pathlib import Path

# 导入自定义模块
from models.LocalMAE import localmae_vit_base, localmae_vit_large
from utils.custome_datasets import LocalMAEDataset
from config import INTRINSIC
from src.local_mae_train import create_pred_vector, loss_norm, label_normalize, label_denormalize

def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('LocalMAE模型评估', add_help=True)
    
    # 模型和数据加载参数
    parser.add_argument('--config', type=str, default='./config/LocalMAE.yaml',
                        help='YAML配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='验证数据集路径，如不提供则使用配置文件中的路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='评估批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 评估参数
    parser.add_argument('--vis', action='store_true',
                        help='启用结果可视化')
    parser.add_argument('--save_path', type=str, default='./evaluation_results',
                        help='保存评估结果的路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='评估使用的设备 (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', 
                        help='调试模式，只评估一小部分数据')
    
    return parser

def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, checkpoint_path, device):
    """加载预训练模型"""
    print(f"正在从 {checkpoint_path} 加载模型")
    
    # 根据配置创建模型
    model_type = config.get('model', 'localmae_vit_base')
    if 'base' in model_type.lower():
        model = localmae_vit_base(
            cross_num_heads=config.get('cross_num_heads', 12),
            feature_dim=config.get('feature_dim', 3),
            qkv_bias=config.get('qkv_bias', True),
            mask_weight=config.get('mask_weight', True),
            use_chamfer_dist=config.get('use_chamfer_dist', False),
            use_value_as_weights=config.get('use_value_as_weight', False),
            chamfer_dist_type=config.get('chamfer_dist_type', 'L2')
        )
    elif 'large' in model_type.lower():
        model = localmae_vit_large(
            cross_num_heads=config.get('cross_num_heads', 16),
            feature_dim=config.get('feature_dim', 3),
            qkv_bias=config.get('qkv_bias', True),
            mask_weight=config.get('mask_weight', True),
            use_chamfer_dist=config.get('use_chamfer_dist', False),
            use_value_as_weights=config.get('use_value_as_weight', False),
            chamfer_dist_type=config.get('chamfer_dist_type', 'L2')
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否直接包含模型权重或是否在 'model' 键下
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
    else:
        model_state = checkpoint
        
    # 加载模型权重
    msg = model.load_state_dict(model_state, strict=False)
    print(f"模型加载信息: {msg}")
    
    model.to(device)
    model.eval()
    
    return model

def prepare_data(args, config):
    """准备数据加载器"""
    transform_val = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 更新配置以指向正确的数据路径
    config_copy = config.copy()
    if args.data_path:
        config_copy['data_path'] = args.data_path
    
    dataset_val = LocalMAEDataset(
        config_copy, 
        is_train=False, 
        transform=transform_val, 
        use_fix_template=config.get('use_fix_template', False)
    )
    
    # 如果是调试模式，只使用一小部分数据
    if args.debug:
        dataset_size = min(100, len(dataset_val))
        indices = torch.randperm(len(dataset_val))[:dataset_size].tolist()
        from torch.utils.data import Subset
        dataset_val = Subset(dataset_val, indices)
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset_val)}")
    print(f"批次大小: {args.batch_size}")
    print(f"总批次数: {len(dataloader_val)}")
    
    return dataloader_val

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    
    all_errors = []
    all_abs_errors = []
    all_pred_values = []
    all_true_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            # 解包数据
            if len(batch) == 8:
                img1, img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2 = batch
            else:
                # 处理Subset产生的数据格式
                img1, img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2 = batch[0]
                img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
                touch_img_mask1, touch_img_mask2 = touch_img_mask1.unsqueeze(0), touch_img_mask2.unsqueeze(0)
                sample_contour1, sample_contour2 = sample_contour1.unsqueeze(0), sample_contour2.unsqueeze(0)
                label1, label2 = label1.unsqueeze(0), label2.unsqueeze(0)
            
            # 移动数据到设备
            img1 = img1.to(device)
            img2 = img2.to(device)
            touch_img_mask1 = touch_img_mask1.to(device)
            touch_img_mask2 = touch_img_mask2.to(device)
            sample_contour1 = sample_contour1.to(device)
            sample_contour2 = sample_contour2.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            
            # 模型预测
            pred, _, _ = model(
                img1, img2, 
                touch_img_mask1=touch_img_mask1, 
                touch_img_mask2=touch_img_mask2,
                sample_contourl1=sample_contour1,
                sample_contourl2=sample_contour2,
                mask_ratio=0.0  # 评估时不使用掩码
            )
            
            # 计算预测值与真实值
            pred_vector = create_pred_vector(pred, intrinsic=INTRINSIC, img_size=[560, 560])
            delta_label = label2 - label1
            
            # 计算误差 (预测值 - 真实值)
            errors = pred_vector - delta_label
            abs_errors = torch.abs(errors)
            
            # 收集结果
            all_errors.append(errors.cpu().numpy())
            all_abs_errors.append(abs_errors.cpu().numpy())
            all_pred_values.append(pred_vector.cpu().numpy())
            all_true_values.append(delta_label.cpu().numpy())
    
    # 合并所有批次的结果
    all_errors = np.concatenate(all_errors, axis=0)
    all_abs_errors = np.concatenate(all_abs_errors, axis=0)
    all_pred_values = np.concatenate(all_pred_values, axis=0)
    all_true_values = np.concatenate(all_true_values, axis=0)
    
    # 计算每个维度的平均绝对误差(MAE)
    mae_per_dim = np.mean(all_abs_errors, axis=0)
    
    # 计算每个维度的平均误差(Mean Error)
    mean_error_per_dim = np.mean(all_errors, axis=0)
    
    # 计算每个维度的标准差(STD)
    std_per_dim = np.std(all_errors, axis=0)
    
    # 计算整体MAE
    overall_mae = np.mean(all_abs_errors)
    
    return {
        'mae_per_dim': mae_per_dim,
        'mean_error_per_dim': mean_error_per_dim,
        'std_per_dim': std_per_dim,
        'overall_mae': overall_mae,
        'all_errors': all_errors,
        'all_pred_values': all_pred_values,
        'all_true_values': all_true_values
    }

def visualize_results(results, save_path):
    """可视化评估结果"""
    os.makedirs(save_path, exist_ok=True)
    
    # 提取结果
    all_errors = results['all_errors']
    all_pred = results['all_pred_values']
    all_true = results['all_true_values']
    
    # 创建误差分布直方图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    dim_names = ['X (mm)', 'Y (mm)', 'Rotation (deg)']
    
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        # 绘制误差分布直方图
        ax.hist(all_errors[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=results['mean_error_per_dim'][i], color='r', linestyle='--', 
                  label=f'均值: {results["mean_error_per_dim"][i]:.4f}')
        
        # 添加均值±标准差区间
        ax.axvline(x=results['mean_error_per_dim'][i] + results['std_per_dim'][i], 
                  color='g', linestyle=':', 
                  label=f'均值±标准差: ({results["mean_error_per_dim"][i]:.4f}±{results["std_per_dim"][i]:.4f})')
        ax.axvline(x=results['mean_error_per_dim'][i] - results['std_per_dim'][i], 
                  color='g', linestyle=':')
        
        ax.set_title(f'{name} 误差分布')
        ax.set_xlabel('误差')
        ax.set_ylabel('频率')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'error_distribution.png'), dpi=300)
    plt.close(fig)
    
    # 创建误差箱线图
    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(all_errors, labels=dim_names)
    plt.title('误差分布箱线图')
    plt.ylabel('误差')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'error_boxplot.png'), dpi=300)
    plt.close(fig)
    
    # 创建预测值与真实值的散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.scatter(all_true[:, i], all_pred[:, i], alpha=0.5, s=10)
        
        # 添加对角线
        min_val = min(all_true[:, i].min(), all_pred[:, i].min())
        max_val = max(all_true[:, i].max(), all_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f'{name} 预测值 vs 真实值')
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pred_vs_true.png'), dpi=300)
    plt.close(fig)
    
    # 保存数值结果
    with open(os.path.join(save_path, 'evaluation_results.txt'), 'w') as f:
        f.write("评估结果\n")
        f.write("=================\n\n")
        
        f.write("平均绝对误差 (MAE):\n")
        for i, name in enumerate(dim_names):
            f.write(f"  {name}: {results['mae_per_dim'][i]:.6f}\n")
        f.write(f"  整体: {results['overall_mae']:.6f}\n\n")
        
        f.write("平均误差:\n")
        for i, name in enumerate(dim_names):
            f.write(f"  {name}: {results['mean_error_per_dim'][i]:.6f}\n")
        
        f.write("\n标准差 (STD):\n")
        for i, name in enumerate(dim_names):
            f.write(f"  {name}: {results['std_per_dim'][i]:.6f}\n")

def main():
    """主函数"""
    # 解析命令行参数
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
    save_path = os.path.join(args.save_path, f"{checkpoint_name}_{timestamp}")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_yaml_config(args.config)
    print(f"已加载配置文件: {args.config}")
    
    # 加载模型
    model = load_model(config, args.checkpoint, device)
    
    # 准备数据
    dataloader = prepare_data(args, config)
    
    # 评估模型
    print("\n开始评估模型...")
    results = evaluate_model(model, dataloader, device)
    
    # 打印评估结果
    print("\n评估结果:")
    print("=================")
    
    print("\n平均绝对误差 (MAE):")
    print(f"  X: {results['mae_per_dim'][0]:.6f} mm")
    print(f"  Y: {results['mae_per_dim'][1]:.6f} mm")
    print(f"  旋转: {results['mae_per_dim'][2]:.6f} 度")
    print(f"  整体: {results['overall_mae']:.6f}")
    
    print("\n平均误差:")
    print(f"  X: {results['mean_error_per_dim'][0]:.6f} mm")
    print(f"  Y: {results['mean_error_per_dim'][1]:.6f} mm")
    print(f"  旋转: {results['mean_error_per_dim'][2]:.6f} 度")
    
    print("\n标准差 (STD):")
    print(f"  X: {results['std_per_dim'][0]:.6f} mm")
    print(f"  Y: {results['std_per_dim'][1]:.6f} mm")
    print(f"  旋转: {results['std_per_dim'][2]:.6f} 度")
    
    # 可视化结果
    if args.vis:
        print("\n正在可视化结果...")
        visualize_results(results, save_path)
        print(f"可视化结果已保存至 {save_path}")
    
    # 保存评估结果
    with open(os.path.join(save_path, 'evaluation_results.txt'), 'w') as f:
        f.write("LocalMAE模型评估结果\n")
        f.write("===================\n\n")
        f.write(f"模型检查点: {args.checkpoint}\n")
        f.write(f"配置文件: {args.config}\n")
        f.write(f"评估时间: {timestamp}\n\n")
        
        f.write("平均绝对误差 (MAE):\n")
        f.write(f"  X: {results['mae_per_dim'][0]:.6f} mm\n")
        f.write(f"  Y: {results['mae_per_dim'][1]:.6f} mm\n")
        f.write(f"  旋转: {results['mae_per_dim'][2]:.6f} 度\n")
        f.write(f"  整体: {results['overall_mae']:.6f}\n\n")
        
        f.write("平均误差:\n")
        f.write(f"  X: {results['mean_error_per_dim'][0]:.6f} mm\n")
        f.write(f"  Y: {results['mean_error_per_dim'][1]:.6f} mm\n")
        f.write(f"  旋转: {results['mean_error_per_dim'][2]:.6f} 度\n\n")
        
        f.write("标准差 (STD):\n")
        f.write(f"  X: {results['std_per_dim'][0]:.6f} mm\n")
        f.write(f"  Y: {results['std_per_dim'][1]:.6f} mm\n")
        f.write(f"  旋转: {results['std_per_dim'][2]:.6f} 度\n")
    
    print(f"\n评估结果已保存至 {os.path.join(save_path, 'evaluation_results.txt')}")

if __name__ == "__main__":
    main()