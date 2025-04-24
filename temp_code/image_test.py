import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

def compute_image_gradients(image_tensor):
    """
    计算图像的梯度
    
    Args:
        image_tensor: 输入图像张量 [B, C, H, W]
        
    Returns:
        grad_x, grad_y: x和y方向的梯度
    """
    # 定义Sobel滤波器
    device = image_tensor.device
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    B, C, H, W = image_tensor.shape
    grad_x = torch.zeros((B, C, H, W), device=device)
    grad_y = torch.zeros((B, C, H, W), device=device)
    
    # 对每个通道分别计算梯度
    for c in range(C):
        # 提取单通道
        channel = image_tensor[:, c:c+1, :, :]
        
        # 计算x和y方向的梯度
        pad = F.pad(channel, (1, 1, 1, 1), mode='replicate')
        grad_x[:, c:c+1, :, :] = F.conv2d(pad, sobel_x)
        grad_y[:, c:c+1, :, :] = F.conv2d(pad, sobel_y)
    
    return grad_x, grad_y

def compute_gradient_magnitude(image_tensor):
    """
    计算图像的梯度幅值
    
    Args:
        image_tensor: 输入图像张量 [B, C, H, W]
        
    Returns:
        梯度幅值图 [B, C, H, W]
    """
    grad_x, grad_y = compute_image_gradients(image_tensor)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def load_image(image_path, size=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载图像并转换为张量
    
    Args:
        image_path: 图像文件路径
        size: 可选，调整图像大小
        device: 计算设备
        
    Returns:
        image_tensor: 图像张量 [1, C, H, W]
    """
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小（如需要）
    if size is not None:
        image = image.resize(size)
    
    # 转换为张量
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).to(device)
    
    # 添加批处理维度
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def visualize_gradient_analysis(image_path, size=None, save_dir=None, prefix=''):
    """
    可视化图像及其梯度分析
    
    Args:
        image_path: 图像文件路径
        size: 可选，调整图像大小
        save_dir: 可选，保存结果的目录
        prefix: 文件名前缀
    """
    # 使用自动选择的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建保存目录（如果需要）
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载图像
    image_tensor = load_image(image_path, size, device)
    print(f"图像尺寸: {image_tensor.shape}")
    
    # 计算梯度
    grad_x, grad_y = compute_image_gradients(image_tensor)
    gradient_magnitude = compute_gradient_magnitude(image_tensor)
    
    # 将张量转换为numpy数组以便可视化
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    grad_x_np = grad_x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    grad_y_np = grad_y.squeeze(0).permute(1, 2, 0).cpu().numpy()
    grad_mag_np = gradient_magnitude.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 计算梯度统计信息
    grad_mean = grad_mag_np.mean()
    grad_std = grad_mag_np.std()
    grad_max = grad_mag_np.max()
    print(f"梯度统计: 均值={grad_mean:.4f}, 标准差={grad_std:.4f}, 最大值={grad_max:.4f}")
    
    # 针对可视化归一化梯度
    def normalize_for_display(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    grad_x_norm = normalize_for_display(grad_x_np)
    grad_y_norm = normalize_for_display(grad_y_np)
    grad_mag_norm = normalize_for_display(grad_mag_np)
    
    # 创建彩色梯度可视化
    grad_x_rgb = np.zeros_like(image_np)
    grad_y_rgb = np.zeros_like(image_np)
    grad_mag_rgb = np.zeros_like(image_np)
    
    # 使用热力图表示梯度
    cmap = plt.cm.viridis
    for c in range(3):  # 对每个颜色通道
        grad_x_rgb[:, :, c] = cmap(grad_x_norm[:, :, c])[:, :, 0]
        grad_y_rgb[:, :, c] = cmap(grad_y_norm[:, :, c])[:, :, 0]
        grad_mag_rgb[:, :, c] = cmap(grad_mag_norm[:, :, c])[:, :, 0]
    
    # 1. 显示原始图像和梯度
    plt.figure(figsize=(20, 10))
    
    plt.subplot(231)
    plt.imshow(image_np)
    plt.title('原始RGB图像')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(grad_x_norm.mean(axis=2), cmap='viridis')
    plt.title('X方向梯度')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(grad_y_norm.mean(axis=2), cmap='viridis')
    plt.title('Y方向梯度')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(grad_mag_norm.mean(axis=2), cmap='viridis')
    plt.title('梯度幅值')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(image_np)
    plt.imshow(grad_mag_norm.mean(axis=2), cmap='viridis', alpha=0.7)
    plt.title('RGB图像与梯度幅值叠加')
    plt.axis('off')
    
    # 提取图像文件名用于标题
    image_name = os.path.basename(image_path)
    plt.suptitle(f'图像梯度分析: {image_name}', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # 保存图像
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}gradient_analysis.png"), dpi=300)
    
    plt.show()
    plt.close()
    
    # 2. 绘制梯度幅值的3D地形图
    plt.figure(figsize=(16, 12))
    
    # 为了更好的可视化效果，可以对梯度幅值应用平滑
    from scipy.ndimage import gaussian_filter
    smoothed_magnitude = gaussian_filter(grad_mag_norm.mean(axis=2), sigma=1)
    
    # 创建网格
    h, w = smoothed_magnitude.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # 创建3D地形图
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x_mesh, y_mesh, smoothed_magnitude, 
                          cmap=cm.viridis, 
                          linewidth=0, 
                          antialiased=True)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置轴标签和标题
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('梯度幅值')
    ax.set_title(f'梯度幅值3D地形图: {image_name}')
    
    # 添加颜色条
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}gradient_terrain.png"), dpi=300)
    
    plt.show()
    plt.close()
    
    # 3. 绘制梯度方向的矢量场（下采样以提高可视化效果）
    plt.figure(figsize=(12, 10))
    
    # 展示原始图像
    plt.imshow(image_np)
    
    # 将梯度转换为平均值
    grad_x_mean = grad_x_np.mean(axis=2)
    grad_y_mean = grad_y_np.mean(axis=2)
    
    # 对梯度场进行下采样以提高可视化效果
    step = 20  # 每20个像素取一个点
    h, w = grad_x_mean.shape
    y_indices, x_indices = np.mgrid[0:h:step, 0:w:step]
    
    # 获取下采样位置的梯度
    x_sample = grad_x_mean[y_indices, x_indices]
    y_sample = grad_y_mean[y_indices, x_indices]
    
    # 绘制矢量场
    plt.quiver(x_indices, y_indices, x_sample, y_sample, 
              color='yellow', scale=50, width=0.001)
    
    plt.title(f'梯度方向矢量场: {image_name}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}gradient_vector_field.png"), dpi=300)
    
    plt.show()
    plt.close()
    
    return {
        'image': image_np,
        'grad_x': grad_x_np,
        'grad_y': grad_y_np,
        'grad_magnitude': grad_mag_np,
        'stats': {
            'mean': grad_mean,
            'std': grad_std,
            'max': grad_max
        }
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像梯度分析与可视化')
    parser.add_argument('--image', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_3530P_0.png', help='输入图像路径')
    parser.add_argument('--size', type=str, default=None, help='调整图像大小，格式为"宽,高"')
    parser.add_argument('--save-dir', type=str, default='./gradient_results', help='保存结果的目录')
    parser.add_argument('--prefix', type=str, default='', help='输出文件名前缀')
    
    args = parser.parse_args()
    
    # 解析大小参数
    size = None
    if args.size:
        width, height = map(int, args.size.split(','))
        size = (width, height)
    
    # 执行梯度分析与可视化
    results = visualize_gradient_analysis(
        args.image,
        size=size,
        save_dir=args.save_dir,
        prefix=args.prefix
    )
    
    print("分析完成!")

if __name__ == "__main__":
    main()