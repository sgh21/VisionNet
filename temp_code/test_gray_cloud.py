import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# 导入 TerrainPointCloud 模块
from utils.TransUtils import TerrainPointCloud, TerraceMapGenerator

def test_terrain_point_cloud():
    """测试 TerrainPointCloud 模块，加载灰度图并生成可视化"""
    
    # 灰度图文件夹路径 - 请替换为你的灰度图所在文件夹
    grayscale_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/data_all/masks/gray"
    mask_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/data_all/masks/binary"
    # 设置需要加载的灰度图数量
    num_images_to_load = 4 # 可以根据需要修改
    
    # 设置输出文件夹
    output_dir = "./terrain_point_cloud_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建转换器，将图像转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像转换为 tensor，并进行归一化 [0, 1]
    ])
    
    # 获取灰度图文件列表
    image_files = sorted([f for f in os.listdir(grayscale_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])[:num_images_to_load]
    
    if not image_files:
        print(f"在目录 {grayscale_dir} 中未找到图像文件")
        return
    from config import EXPANSION_SIZE
    print(f"找到 {len(image_files)} 个图像文件。正在加载...")
    # 创建TerraceMapGenerator (可选，用于生成梯田图)
    terrace_map_generator = TerraceMapGenerator(
        intensity_scaling=[0.0, 0.6, 0.8, 1.0],
        edge_enhancement=1.0,
        expansion_size=EXPANSION_SIZE,
        sample_size=256,
        debug=False
    )
    # 加载灰度图并转换为张量
    grayscale_tensors = []
    for img_file in image_files:
        img_path = os.path.join(grayscale_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        try:
            # 加载图像并转换为灰度
            img = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            # 转换为张量
            tensor = transform(img)  # [1, H, W]
            
            # 生成梯田图
            terrace_map, _ = terrace_map_generator(mask)
            terrace_map = transform(terrace_map)  # 转换为张量
            tensor = terrace_map*tensor  # 将梯田图应用于灰度图
            grayscale_tensors.append(tensor)
            print(f"已加载 {img_file}，形状: {tuple(tensor.shape)}")
        except Exception as e:
            print(f"无法加载图像 {img_file}: {e}")
    
    if not grayscale_tensors:
        print("没有成功加载任何图像")
        return
    
    # 将张量堆叠为批次
    batch_tensor = torch.stack(grayscale_tensors)  # [B, 1, H, W]
    batch_tensor = batch_tensor.cuda()
    print(f"批次张量形状: {tuple(batch_tensor.shape)}")
    
    # 创建 TerrainPointCloud 实例
    terrain_point_cloud = TerrainPointCloud(
        target_points=2048,
        min_value=0.05,
        stride=2,
        normalize_coords=True,
        importance_scaling=2.0
    )

    # 计时开始
    start_time = time.time()
    
    # 生成点云
    point_clouds = terrain_point_cloud(batch_tensor)
    
    # 计时结束
    elapsed_time = time.time() - start_time
    print(f"点云生成完成，耗时: {elapsed_time:.4f} 秒")
    print(f"点云形状: {tuple(point_clouds.shape)}")
    
    # 使用内置的可视化方法
    print("正在生成基本可视化...")
    terrain_point_cloud.visualize(batch_tensor, point_clouds)
    
    # 创建更详细的可视化
    print("正在生成详细可视化...")
    create_detailed_visualizations(batch_tensor, point_clouds, image_files, output_dir)
    
    return point_clouds

def create_detailed_visualizations(grayscale_tensor, point_clouds, image_names, output_dir):
    """创建更详细的点云可视化"""
    
    batch_size = grayscale_tensor.size(0)
    
    for b in range(batch_size):
        # 提取当前样本
        gray_np = grayscale_tensor[b, 0].cpu().numpy()
        point_cloud_np = point_clouds[b].detach().cpu().numpy()
        image_name = Path(image_names[b]).stem  # 移除文件扩展名
        
        # 1. 创建大图可视化
        fig = plt.figure(figsize=(18, 12))
        
        # 原始灰度图
        ax1 = fig.add_subplot(231)
        ax1.imshow(gray_np, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        
        # 热力图样式灰度图
        ax2 = fig.add_subplot(232)
        im = ax2.imshow(gray_np, cmap='viridis')
        ax2.set_title('Heatmap Style')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # 计算非零点比例
        non_zero_ratio = np.sum(gray_np > 0.05) / gray_np.size
        
        # 2D 点采样可视化
        ax3 = fig.add_subplot(233)
        ax3.imshow(gray_np, cmap='gray')
        
        # 将归一化坐标转换回原始坐标
        h = ((point_cloud_np[:, 1] + 1) / 2) * (gray_np.shape[0] - 1)
        w = ((point_cloud_np[:, 0] + 1) / 2) * (gray_np.shape[1] - 1)
        
        scatter = ax3.scatter(w, h, c=point_cloud_np[:, 2], s=10, cmap='plasma', alpha=0.7)
        ax3.set_title('2D sampling')
        ax3.axis('off')
        plt.colorbar(scatter, ax=ax3)
        
        # 3D 点云 (默认视角)
        ax4 = fig.add_subplot(234, projection='3d')
        scatter = ax4.scatter(
            point_cloud_np[:, 0], 
            point_cloud_np[:, 1], 
            point_cloud_np[:, 2], 
            c=point_cloud_np[:, 2], 
            s=15, 
            cmap='plasma', 
            alpha=0.8
        )
        ax4.set_title('3D Point Cloud (Default View)')
        ax4.set_xlabel('X轴')
        ax4.set_ylabel('Y轴')
        ax4.set_zlabel('Z轴')
        plt.colorbar(scatter, ax=ax4, shrink=0.6)
        
        # 3D 点云 (俯视图)
        ax5 = fig.add_subplot(235, projection='3d')
        scatter = ax5.scatter(
            point_cloud_np[:, 0], 
            point_cloud_np[:, 1], 
            point_cloud_np[:, 2], 
            c=point_cloud_np[:, 2], 
            s=15, 
            cmap='plasma', 
            alpha=0.8
        )
        ax5.view_init(elev=90, azim=0)  # 俯视图
        ax5.set_title('3D Point Cloud (Top View)')
        ax5.set_xlabel('X轴')
        ax5.set_ylabel('Y轴')
        ax5.set_zlabel('Z轴')
        
        # 3D 点云 (侧视图)
        ax6 = fig.add_subplot(236, projection='3d')
        scatter = ax6.scatter(
            point_cloud_np[:, 0], 
            point_cloud_np[:, 1], 
            point_cloud_np[:, 2], 
            c=point_cloud_np[:, 2], 
            s=15, 
            cmap='plasma', 
            alpha=0.8
        )
        ax6.view_init(elev=0, azim=90)  # 侧视图
        ax6.set_title('3D Point Cloud (Side View)')
        ax6.set_xlabel('X轴')
        ax6.set_ylabel('Y轴')
        ax6.set_zlabel('Z轴')
        
        # 添加图像信息
        fig.suptitle(f'图像: {image_name}', fontsize=16)
        
        # 添加点云统计信息
        info_text = (
            f"点云信息:\n"
            f"• 点数: {point_cloud_np.shape[0]}\n"
            f"• 最小Z值: {point_cloud_np[:, 2].min():.3f}\n"
            f"• 最大Z值: {point_cloud_np[:, 2].max():.3f}\n"
            f"• 平均Z值: {point_cloud_np[:, 2].mean():.3f}\n"
            f"• Z值标准差: {point_cloud_np[:, 2].std():.3f}\n"
            f"• 零值点比例: {np.sum(point_cloud_np[:, 2] == 0) / point_cloud_np.shape[0]:.2%}\n"
            f"• 灰度非零比例: {non_zero_ratio:.2%}"
        )
        fig.text(0.01, 0.02, info_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{image_name}_point_cloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存可视化到: {output_path}")
        
        plt.close()
        
        # 2. 创建交互式旋转视图 (保存多个角度)
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            point_cloud_np[:, 0], 
            point_cloud_np[:, 1], 
            point_cloud_np[:, 2], 
            c=point_cloud_np[:, 2], 
            s=20, 
            cmap='viridis', 
            alpha=0.8
        )
        
        ax.set_title(f'{image_name} 3D Point Cloud', fontsize=14)
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='高度值')
        
        # 保存不同视角
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            ax.view_init(elev=30, azim=angle)
            plt.tight_layout()
            angle_output_path = os.path.join(output_dir, f"{image_name}_angle_{angle}.png")
            plt.savefig(angle_output_path, dpi=200)
        
        plt.close()
        
        # 保存点云数据
        output_data_path = os.path.join(output_dir, f"{image_name}_point_cloud.npy")
        np.save(output_data_path, point_cloud_np)

if __name__ == "__main__":
    point_clouds = test_terrain_point_cloud()