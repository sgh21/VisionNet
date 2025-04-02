# -*- coding: utf-8 -*-
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F

def forward_transfer(x, T):
        """
        对输入进行旋转和平移变换，使零件与世界坐标系中的零件重叠
        变换顺序：先平移后旋转
        
        Args:
            x (Tensor): 输入数据，[B, C, H, W]
            T (Tensor): 变换矩阵，[B, 8] (a, b, c, d, cx, cy, tx, ty)
                其中[a, b; c, d]构成旋转矩阵R
                [cx, cy]为旋转中心坐标(归一化到[-1,1])
                [tx, ty]构成平移向量t（归一化到[-1,1]）
        Returns:
            Tensor: 变换后的图像，[B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 创建归一化网格坐标并展开到批次维度
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 计算变换的逆矩阵 (用于逆向映射)
        # 提取变换参数
        a = T[:, 0].view(B, 1, 1)  # [B, 1, 1]
        b_ = T[:, 1].view(B, 1, 1)
        c = T[:, 2].view(B, 1, 1)
        d = T[:, 3].view(B, 1, 1)
        cx = T[:, 4].view(B, 1, 1)  # 旋转中心(归一化坐标)
        cy = T[:, 5].view(B, 1, 1)
        tx = T[:, 6].view(B, 1, 1)  # 平移量(归一化坐标)
        ty = T[:, 7].view(B, 1, 1)
        
        # 计算行列式和逆矩阵 (更稳定的方式)
        det = a * d - b_ * c
        eps = 1e-6
        safe_det = torch.where(torch.abs(det) < eps, 
                           torch.ones_like(det) * eps * torch.sign(det), 
                           det)
        
        inv_a = d / safe_det
        inv_b = -b_ / safe_det
        inv_c = -c / safe_det
        inv_d = a / safe_det
        
        # 扩展网格坐标到批次维度
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        
        # 逆向映射坐标计算（从输出找输入）:
        # 1. 将坐标相对于旋转中心
        x_centered = grid_x - cx - tx
        y_centered = grid_y - cy - ty
        
        # 2. 应用旋转的逆变换
        x_unrotated = inv_a * x_centered + inv_b * y_centered
        y_unrotated = inv_c * x_centered + inv_d * y_centered
        
        # 3. 加回旋转中心
        x_in = x_unrotated + cx
        y_in = y_unrotated + cy
        
        
        # 组合成采样网格
        grid = torch.stack([x_in, y_in], dim=-1)  # [B, H, W, 2]
        
        # 使用grid_sample实现双线性插值
        return torch.nn.functional.grid_sample(
            x, 
            grid, 
            mode='bilinear',      
            padding_mode='zeros', 
            align_corners=True    
        )
def forward_transfer(x, params):
        """
        使用5参数[theta,cx,cy,tx,ty]应用仿射变换到输入图像
        
        Args:
            x (Tensor): 输入数据，[B, C, H, W]
            params (Tensor): 变换参数，[B, 5] (theta, cx, cy, tx, ty)
                其中theta是旋转角度(-π, π)
                [cx, cy]为旋转中心坐标(-1, 1)
                [tx, ty]构成平移向量(-1, 1)
                
        Returns:
            Tensor: 变换后的图像，[B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 提取参数
        theta = params[:, 0]  # 旋转角度，(-π, π)范围
        cx = params[:, 1]  # 旋转中心x，(-1, 1)范围
        cy = params[:, 2]  # 旋转中心y，(-1, 1)范围
        tx = params[:, 3]  # x方向平移，(-1, 1)范围
        ty = params[:, 4]  # y方向平移，(-1, 1)范围
        
        # 构建旋转矩阵
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 旋转矩阵元素
        a = cos_theta
        b = -sin_theta
        c = sin_theta
        d = cos_theta
        
        # 创建归一化网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 扩展网格坐标到批次维度
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        
        # 计算行列式(稳定性检查)
        det = a * d - b * c
        eps = 1e-6
        safe_det = torch.where(torch.abs(det) < eps, 
                           torch.ones_like(det) * eps * torch.sign(det), 
                           det)
        
        # 计算逆变换矩阵(用于逆向映射)
        inv_a = d / safe_det
        inv_b = -b / safe_det
        inv_c = -c / safe_det
        inv_d = a / safe_det
        
        # 将参数调整为[B,1,1]形状，方便广播
        inv_a = inv_a.view(B, 1, 1)
        inv_b = inv_b.view(B, 1, 1)
        inv_c = inv_c.view(B, 1, 1)
        inv_d = inv_d.view(B, 1, 1)
        cx = cx.view(B, 1, 1)
        cy = cy.view(B, 1, 1)
        tx = tx.view(B, 1, 1)
        ty = ty.view(B, 1, 1)
        
        # 逆向映射坐标计算（从输出找输入）:
        # 1. 先应用平移的逆变换
        x_after_trans = grid_x - tx  
        y_after_trans = grid_y - ty
        
        # 2. 将坐标相对于旋转中心
        x_centered = x_after_trans - cx
        y_centered = y_after_trans - cy
        
        # 3. 应用旋转的逆变换
        x_unrotated = inv_a * x_centered + inv_b * y_centered
        y_unrotated = inv_c * x_centered + inv_d * y_centered
        
        # 4. 加回旋转中心
        x_in = x_unrotated + cx
        y_in = y_unrotated + cy
        
        # 组合成采样网格
        grid = torch.stack([x_in, y_in], dim=-1)  # [B, H, W, 2]
        
        # 使用grid_sample实现双线性插值
        return torch.nn.functional.grid_sample(
            x, 
            grid, 
            mode='bilinear',      
            padding_mode='zeros', 
            align_corners=True    
        )
def create_transformation(rotation_angle=0, scale=1.0, tx=0, ty=0, cx=0, cy=0):
    """
    创建变换矩阵
    
    Args:
        rotation_angle: 旋转角度（弧度）
        scale: 缩放比例
        tx, ty: 平移距离
        
    Returns:
        Tensor: [1, 6] 变换矩阵参数
    """
    # 创建旋转矩阵
    # a = scale * np.cos(rotation_angle)
    # b = -scale * np.sin(rotation_angle)
    # c = scale * np.sin(rotation_angle)
    # d = scale * np.cos(rotation_angle)
    
    
    # 创建变换参数
    return torch.tensor([[rotation_angle, cx, cy, tx, ty]], dtype=torch.float32)

def visualize_transformation(img_path, output_path, rotation_angle=30, scale=1.0, tx=20, ty=20, cx=0, cy=0):
    """
    加载图像，应用变换，并可视化结果
    
    Args:
        img_path: 输入图像路径
        output_path: 输出图像保存路径
        rotation_angle: 旋转角度（度数）
        scale: 缩放比例
        tx, ty: 平移距离（像素）
    """
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    # 获取图像尺寸
    _, _, H, W = img_tensor.shape
    
    # 创建变换矩阵 (角度转弧度)
    T = create_transformation(
        rotation_angle=rotation_angle * np.pi / 180, 
        scale=scale,
        tx=tx,
        ty=ty,
        cx=cx,
        cy=cy
    )
    
    # 应用变换
    transformed_img = forward_transfer(img_tensor, T)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor[0].permute(1, 2, 0).numpy())
    plt.title('origin')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img[0].permute(1, 2, 0).numpy())
    plt.title(f'dis: theta={rotation_angle}°, scale={scale}, trans=({tx},{ty})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"结果已保存到 {output_path}")
# TODO:写一个测试软件，导入label然后变换成图片的旋转，查看效果
def main():
    parser = argparse.ArgumentParser(description='图像变换可视化')
    parser.add_argument('--img_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_path', type=str, default='transformed_image.png', help='输出图像路径')
    parser.add_argument('--rotation', type=float, default=90, help='旋转角度（度数）')
    parser.add_argument('--scale', type=float, default=1.0, help='缩放比例')
    parser.add_argument('--cx', type=float, default=-1, help='X方向平移距离')
    parser.add_argument('--cy', type=float, default=-1, help='Y方向平移距离')
    parser.add_argument('--tx', type=float, default=-1, help='X方向平移距离')
    parser.add_argument('--ty', type=float, default=-1, help='Y方向平移距离')
    
    args = parser.parse_args()
    
    visualize_transformation(
        args.img_path, 
        args.output_path,
        rotation_angle=args.rotation,
        scale=args.scale,
        tx=args.tx,
        ty=args.ty,
        cx=args.cx,
        cy=args.cy
    )

if __name__ == "__main__":
    main()