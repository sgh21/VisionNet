import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Footshone import MaskPatchPooling
from utils.TransUtils import TerraceMapGenerator
from config import EXPANSION_SIZE
def downsample_mask(mask, target_size=(224, 224)):
        """
        将掩码从原始尺寸下采样到目标尺寸
        
        Args:
            mask (torch.Tensor): 原始掩码, 形状为 [B, 1, H, W]
            target_size (tuple): 目标尺寸, 形式为 (height, width)
            
        Returns:
            torch.Tensor: 下采样后的掩码, 形状为 [B, 1, target_height, target_width]
        """
        return torch.nn.functional.interpolate(
            mask, 
            size=target_size, 
            mode='bilinear',  # 对于掩码，建议使用'nearest'或'bilinear'
            align_corners=False
        )
def visualize_mask_effect(img_path, touch_mask_path, patch_size=16, save_path=None):
    """
    可视化图像乘以掩码的效果
    
    Args:
        img_path: 原始图像路径
        touch_mask_path: 触摸掩码图像路径
        patch_size: patch大小
        save_path: 保存结果的路径（如果为None则直接显示）
    """
    # 加载图像和掩码
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # 缩小图像
    touch_mask = Image.open(touch_mask_path).convert('L')
    
    # 获取图像的基本信息
    img_filename = os.path.basename(img_path)
    serial = img_filename.split('_')[-2]  # 根据文件名获取类别
    
    # 转换为张量
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)  # [1, 3, H, W]
    
    # 生成地形图
    terrace_map_generator = TerraceMapGenerator(
        intensity_scaling=[0.0, 0.6, 0.8, 1.0],
        edge_enhancement=1.0,
        sample_size=100,
        expansion_size=EXPANSION_SIZE,
    )
    terrace_map, _ = terrace_map_generator(touch_mask, serial=serial)
    touch_mask_tensor = to_tensor(terrace_map).unsqueeze(0)  # [1, 1, H, W]
    touch_mask_tensor = downsample_mask(touch_mask_tensor, target_size=(224, 224))  # 下采样掩码
    
    # 仅保留掩码的第一个通道，作为灰度掩码
    mask_gray = touch_mask_tensor[:, 0:1, :, :]  # [1, 1, H, W]
    
    # 创建patch池化层
    img_size = img_tensor.shape[-1]
    mask_patch_pooling = MaskPatchPooling(img_size=img_size, patch_size=patch_size, pool_mode='mean')
    
    # 获取patch级掩码
    patch_mask = mask_patch_pooling(mask_gray)  # [1, N, 1]
    
    # 可视化patch级掩码
    num_patches = int(img_size / patch_size)
    patch_mask_reshaped = patch_mask.reshape(1, num_patches, num_patches, 1)
    patch_mask_img = patch_mask_reshaped.expand(-1, -1, -1, 3).permute(0, 3, 1, 2)
    patch_mask_img = torch.nn.functional.interpolate(
        patch_mask_img, size=(img_size, img_size), mode='nearest'
    )
    
    # 应用掩码到图像
    masked_img = img_tensor * mask_gray
    patch_masked_img = img_tensor * patch_mask_img
    
    # 可视化
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    mask_np = mask_gray.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).numpy()
    masked_img_np = masked_img.squeeze(0).permute(1, 2, 0).numpy()
    patch_mask_np = patch_mask_img.squeeze(0).permute(1, 2, 0).numpy()
    patch_masked_img_np = patch_masked_img.squeeze(0).permute(1, 2, 0).numpy()
    
    # 创建画布
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：原始图像、掩码图像和掩码效果
    axs[0, 0].imshow(img_np)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(mask_np, cmap='gray')
    axs[0, 1].set_title('Mask Image')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(masked_img_np)
    axs[0, 2].set_title('Masked Image')
    axs[0, 2].axis('off')
    
    # 第二行：Patch掩码和应用效果
    axs[1, 0].imshow(patch_mask_np, cmap='gray')
    axs[1, 0].set_title(f'Patch Mask (patch_size={patch_size})')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(patch_masked_img_np)
    axs[1, 1].set_title('Patch Masked Image')
    axs[1, 1].axis('off')
    
    # 计算掩码中心和强度
    mask_intensity = mask_np.mean()
    patch_mask_intensity = patch_mask_np.mean()
    y_indices, x_indices = np.where(mask_np[:, :, 0] > 0.5)
    if len(y_indices) > 0 and len(x_indices) > 0:
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)
        mask_info = f"Mask Center: ({center_x:.1f}, {center_y:.1f})\n"
        mask_info += f"Mask Intensity: {mask_intensity:.4f}\n"
        mask_info += f"Patch Mask Intensity: {patch_mask_intensity:.4f}"
    else:
        mask_info = "未检测到有效掩码区域"
    
    axs[1, 2].text(0.5, 0.5, mask_info, 
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axs[1, 2].transAxes,
                  fontsize=12)
    axs[1, 2].set_title('掩码统计信息')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='掩码应用效果可视化工具')
    parser.add_argument('--img', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_3524P_0.png', help='原始图像路径')
    parser.add_argument('--mask', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/touch_images_mask_process/gel_image_3524P_0.png', help='触摸掩码图像路径')
    parser.add_argument('--patch_size', type=int, default=16, help='patch大小')
    parser.add_argument('--save', type=str, default=None, help='保存结果图像的路径')
    
    args = parser.parse_args()
    
    visualize_mask_effect(args.img, args.mask, args.patch_size, args.save)

if __name__ == "__main__":
    main()