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
from utils.multi_datasets import MultiCrossMAEDataset

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
    
    return parser



def visualize_results(rgb_img1_path, rgb_img2_path, touch_img1_path, touch_img2_path, pred, gt, save_path):
    """可视化RGB和触觉图像对以及预测结果
    Args:
        rgb_img1_path: RGB图像1路径
        rgb_img2_path: RGB图像2路径
        touch_img1_path: 触觉图像1路径
        touch_img2_path: 触觉图像2路径
        pred: 预测值 [dx, dy, drz]
        gt: 真实值 [dx, dy, drz]
        save_path: 保存路径
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示RGB图像对
    rgb_img1 = Image.open(rgb_img1_path)
    rgb_img2 = Image.open(rgb_img2_path)
    ax1.imshow(rgb_img1)
    ax2.imshow(rgb_img2)
    ax1.set_title('RGB Image 1')
    ax2.set_title('RGB Image 2')
    
    # 显示触觉图像对
    touch_img1 = Image.open(touch_img1_path)
    touch_img2 = Image.open(touch_img2_path)
    ax3.imshow(touch_img1)
    ax4.imshow(touch_img2)
    ax3.set_title('Touch Image 1')
    ax4.set_title('Touch Image 2')
    
    # 关闭坐标轴
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')
    
    # 添加预测值和真实值标题
    fig.suptitle(f'Prediction: dx={pred[0]:.2f}mm, dy={pred[1]:.2f}mm, drz={pred[2]:.2f}°\n' + 
                 f'Ground Truth: dx={gt[0]:.2f}mm, dy={gt[1]:.2f}mm, drz={gt[2]:.2f}°',
                 fontsize=12, y=0.95)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_dim_mae(pred, target):
    mae_x = torch.abs(pred[:,0] - target[:,0])
    mae_y = torch.abs(pred[:,1] - target[:,1])
    mae_rz = torch.abs(pred[:,2] - target[:,2])
    return mae_x, mae_y, mae_rz

def main(args):

    print(f"Loading checkpoint from {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # 创建模型
    model = multicrossmae.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        pretrained_path=args.mae_pretrained,
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
            batch_mae_x = mae_x.mean().item()
            batch_mae_y = mae_y.mean().item()
            batch_mae_rz = mae_rz.mean().item()
            
            pbar.set_postfix({
                'MAE_X': f'{batch_mae_x:.4f}',
                'MAE_Y': f'{batch_mae_y:.4f}',
                'MAE_Rz': f'{batch_mae_rz:.4f}'
            })

            # 可视化结果
            for i in range(rgb_img1.size(0)):
                root_path = os.path.join(args.data_path, 'val/')
                rgb_img1_path = os.path.join(root_path, 'rgb_images', img1_name[i])
                rgb_img2_path = os.path.join(root_path, 'rgb_images', img2_name[i])
                touch_img1_path = os.path.join(root_path, 'touch_images', 'gel_' + img1_name[i])
                touch_img2_path = os.path.join(root_path, 'touch_images', 'gel_' + img2_name[i])
                save_path = os.path.join(args.output_dir, f'pair_{batch_idx}_{i}.png')
                
                visualize_results(
                    rgb_img1_path, 
                    rgb_img2_path,
                    touch_img1_path,
                    touch_img2_path,
                    pred[i].cpu().numpy(),
                    delta_label[i].cpu().numpy(),
                    save_path
                )
    
    # 计算并打印平均MAE
    all_maes = np.stack([all_maes_x, all_maes_y, all_maes_rz], axis=1)  # [num_samples, 3]
    avg_maes = np.mean(all_maes, axis=0)
    std_maes = np.std(all_maes, axis=0)
    three_sigmas = avg_maes + 3 * std_maes
    print(f'Average MAE:')
    print(f'X: {avg_maes[0]:.4f} mm')
    print(f'Y: {avg_maes[1]:.4f} mm')
    print(f'Rz: {avg_maes[2]:.4f} deg')

    print(f'mu + 3 * std(X):{three_sigmas[0]:.4f} mm')
    print(f'mu + 3 * std(Y):{three_sigmas[1]:.4f} mm')
    print(f'mu + 3 * std(Rz):{three_sigmas[2]:.4f} deg')
    pbar.close()
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    if args.config:
        args = load_yaml_config(args.config)
        
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)