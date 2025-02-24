import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import glob
import argparse

def add_markers(image_path, grid_size, radius, rgb, output_dir):
    """在图片上添加标记点
    Args:
        image_path: 图片路径
        grid_size: (rows, cols)网格大小
        radius: 标记点半径
        rgb: 标记点颜色(R,G,B)
        output_dir: 输出目录
    """
    # 读取图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 计算网格间距
    h, w = img.size
    x_step = w // (grid_size[1] + 1)
    y_step = h // (grid_size[0] + 1)
    
    # 绘制网格点
    for i in range(1, grid_size[0] + 1):
        for j in range(1, grid_size[1] + 1):
            x = j * x_step
            y = i * y_step
            # 绘制圆形标记
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                        fill=rgb, outline=rgb)
    
    # 保存图片
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    img.save(save_path)

def process_directory(root_path, grid_size, radius, rgb):
    """处理目录下所有图片
    Args:
        root_path: 图片根目录
        grid_size: (rows, cols)网格大小
        radius: 标记点半径
        rgb: 标记点颜色(R,G,B)
    """
    # 创建输出目录
    output_dir = root_path + "_with_markers"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(root_path, "**", ext), 
                                   recursive=True))
    
    print(f"Found {len(image_files)} images")
    
    # 处理每张图片
    for img_path in tqdm(image_files, desc="Processing images"):
        add_markers(img_path, grid_size, radius, rgb, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Add markers to touch images')
    parser.add_argument('--root_path', type=str, required=True, 
                       help='Root directory containing images')
    parser.add_argument('--rows', type=int, default=8,
                       help='Number of grid rows')
    parser.add_argument('--cols', type=int, default=8,
                       help='Number of grid columns')
    parser.add_argument('--radius', type=int, default=4,
                       help='Radius of marker points')
    parser.add_argument('--rgb', type=int, nargs=3, default=[0, 0, 0],
                       help='RGB color of markers')
    
    args = parser.parse_args()
    
    process_directory(args.root_path, 
                     grid_size=(args.rows, args.cols),
                     radius=args.radius,
                     rgb=tuple(args.rgb))

if __name__ == '__main__':
    main()