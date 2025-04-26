import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.TransUtils import GlobalIlluminationAlignment
from utils.TransUtils import torch_to_np


def visualize_alignment_effects(img1, img2, save_path=None):
    """
    可视化光照对齐前后的效果
    
    Args:
        img1: 第一张图像 (PIL Image 或 Tensor)
        img2: 第二张图像 (PIL Image 或 Tensor)
        save_path: 保存结果图像的路径
    """
    # 确保图像是张量形式
    if not isinstance(img1, torch.Tensor):
        transform = transforms.ToTensor()
        img1 = transform(img1).unsqueeze(0)  # [1, 3, H, W]
        img2 = transform(img2).unsqueeze(0)  # [1, 3, H, W]
    else:
        # 如果已经是张量但没有批次维度，添加批次维度
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
    
    # 确保设备一致
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 创建不同类型的光照对齐模块
    global_aligner_mean = GlobalIlluminationAlignment(match_variance=False, per_channel=True)
    global_aligner_full = GlobalIlluminationAlignment(match_variance=True, per_channel=True)
    
    # 应用不同的光照对齐方法
    # 1. 图像2全局均值对齐到图像1
    img2_global_mean = global_aligner_mean(img2, img1)
    
    # 2. 图像2全局均值和方差对齐到图像1
    img2_global_full = global_aligner_full(img2, img1)
    
    
    # 计算差异图
    diff_original = torch.abs(img1 - img2).mean(dim=1, keepdim=True)
    diff_global_mean = torch.abs(img1 - img2_global_mean).mean(dim=1, keepdim=True)
    diff_global_full = torch.abs(img1 - img2_global_full).mean(dim=1, keepdim=True)
    
    # 转换为NumPy数组用于可视化
    img1_np = torch_to_np(img1)
    img2_np = torch_to_np(img2)
    img2_global_mean_np = torch_to_np(img2_global_mean)
    img2_global_full_np = torch_to_np(img2_global_full)
    
    # 差异图也转换为NumPy数组
    diff_original_np = torch_to_np(diff_original.repeat(1, 3, 1, 1))
    diff_global_mean_np = torch_to_np(diff_global_mean.repeat(1, 3, 1, 1))
    diff_global_full_np = torch_to_np(diff_global_full.repeat(1, 3, 1, 1))
    
    # 创建可视化结果数组 - 正确地包含所有图像
    images = [
        img1_np, 
        img2_np, 
        diff_original_np,
        img2_global_mean_np, 
        diff_global_mean_np,
        img2_global_full_np, 
        diff_global_full_np
    ]
    
    titles = [
        'Original Image 1', 
        'Original Image 2', 
        'Original Difference',
        'Global Mean Alignment', 
        'Global Mean Alignment Diff',
        'Global Mean+Variance Alignment', 
        'Global Mean+Variance Alignment Diff'
    ]
    
    # 计算每个差异图的平均误差
    diff_metrics = [
        np.mean(diff_original_np),
        np.mean(diff_global_mean_np),
        np.mean(diff_global_full_np),
    ]
    
    diff_labels = [
        'Original Difference',
        'Global Mean Alignment',
        'Global Mean+Variance Alignment',
    ]
    
    # 可视化图像
    fig = plt.figure(figsize=(18, 12))
    
    # 图像网格 - 修改为2行3列，更简洁地展示内容
    grid = plt.GridSpec(2, 3, figure=fig)
    
    # 第一行：原始图像和差异
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(images[0])
    ax1.set_title(titles[0])
    ax1.axis('off')
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(images[1])
    ax2.set_title(titles[1])
    ax2.axis('off')
    
    ax3 = fig.add_subplot(grid[0, 2])
    ax3.imshow(images[2])
    ax3.set_title(titles[2])
    ax3.axis('off')
    
    # 第二行：全局均值对齐和全局均值+方差对齐
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(images[3])
    ax4.set_title(titles[3])
    ax4.axis('off')
    
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.imshow(images[5])
    ax5.set_title(titles[5])
    ax5.axis('off')
    
    # 添加第三格：差异图对比
    ax6 = fig.add_subplot(grid[1, 2])
    # 创建并排显示的差异图对比
    diff_comparison = np.hstack([images[4], images[6]])
    ax6.imshow(diff_comparison)
    ax6.set_title("Difference Comparison (Mean | Mean+Var)")
    ax6.axis('off')
    
    # 添加标题
    plt.suptitle('Illumination Alignment Comparison', fontsize=16)
    
    # 紧凑布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 绘制误差条形图
    fig_bar = plt.figure(figsize=(12, 6))
    plt.bar(diff_labels, diff_metrics, color=['gray', 'blue', 'green'])
    
    # 更新图表标题
    plt.title('Average Difference for Different Alignment Methods')
    plt.ylabel('Average Pixel Difference')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在每个柱子顶部显示数值
    for i, v in enumerate(diff_metrics):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    
    # 直方图可视化
    fig_hist = plt.figure(figsize=(16, 10))
    
    # 为每个通道创建直方图
    channels = ['Red', 'Green', 'Blue']
    images_for_hist = [img1_np, img2_np, img2_global_mean_np, img2_global_full_np]
    # 更新直方图标签
    labels_for_hist = ['Original Image 1', 'Original Image 2', 'Global Mean Alignment', 
                    'Global Mean+Variance Alignment']

    colors = ['r', 'g', 'b']
    
    for i in range(3):  # 对于每个通道
        ax = fig_hist.add_subplot(3, 1, i+1)
        
        for j, (img, label) in enumerate(zip(images_for_hist, labels_for_hist)):
            # 提取当前通道
            channel_data = img[:, :, i].flatten()
            
            # 绘制直方图
            ax.hist(channel_data, bins=50, alpha=0.3, label=label)
        
        # 添加标题和标签
        ax.set_title(f'{channels[i]} Channel Distribution')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示结果
    if save_path:
        try:
            # 保存主图像
            save_file = f"{save_path}_alignment.png"
            print(f"Saving: {save_file}")
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            
            # 保存误差条形图
            save_file = f"{save_path}_error_bars.png"
            print(f"Saving: {save_file}")
            fig_bar.savefig(save_file, dpi=300, bbox_inches='tight')
            
            # 保存直方图
            save_file = f"{save_path}_histograms.png"
            print(f"Saving: {save_file}")
            fig_hist.savefig(save_file, dpi=300, bbox_inches='tight')
            
            print(f"All results saved to: {save_path}_*.png")
        except Exception as e:
            print(f"Error during saving: {e}")
    else:
        plt.show()


def test_with_images(img1_path, img2_path, save_path=None):
    """使用指定的图像测试光照对齐效果"""
    # 加载图像
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 可视化光照对齐效果
    visualize_alignment_effects(img1, img2, save_path)

def main():
    parser = argparse.ArgumentParser(description='测试光照对齐效果')
    parser.add_argument('--img1', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision/train/images/image_3524P_0.png',help='第一张图像路径')
    parser.add_argument('--img2', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision/train/images/image_3524P_150.png', help='第二张图像路径')
    parser.add_argument('--samples', type=int, default=3, help='从数据集中抽取的样本数量')
    parser.add_argument('--save_dir', type=str, default='illumination_alignment_results', 
                       help='保存结果的目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    if args.save_dir:
        save_dir = os.path.join(os.getcwd(), args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
        
    # 设置保存路径
    save_path = os.path.join(save_dir, "custom_images") if save_dir else None
    
    # 执行图像处理和可视化
    test_with_images(args.img1, args.img2, save_path)

if __name__ == "__main__":
    main()