import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import List, Tuple, Dict
import shutil

class YoloSegBackgroundChanger:
    """
    从YOLO-SEG标签提取实例分割区域并替换背景
    """
    def __init__(self, 
                 rgb_dir: str, 
                 labels_dir: str, 
                 background_image: str,
                 output_dir: str = "output",
                 visualization_dir: str = "visualization"):
        """
        初始化背景替换器
        
        参数:
            rgb_dir: RGB图像目录路径
            labels_dir: YOLO-SEG标签目录路径
            background_image: 背景图像路径
            output_dir: 输出目录
            visualization_dir: 可视化输出目录
        """
        self.rgb_dir = rgb_dir
        self.labels_dir = labels_dir
        self.background_image = background_image
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir
        
        # 读取背景图像
        self.bg_img = cv2.imread(background_image)
        if self.bg_img is None:
            raise FileNotFoundError(f"无法加载背景图像: {background_image}")
        self.bg_img = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2RGB)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # 类别颜色映射 (用于可视化)
        self.class_colors = {
            0: (255, 0, 0),    # 红色
            1: (0, 255, 0),    # 绿色
            2: (0, 0, 255),    # 蓝色
            3: (255, 255, 0),  # 黄色
            4: (255, 0, 255),  # 紫色
            5: (0, 255, 255),  # 青色
            6: (128, 128, 0),  # 橄榄色
            7: (128, 0, 128),  # 深紫色
            8: (0, 128, 128),  # 深青色
            9: (255, 165, 0),  # 橙色
        }
        
    def parse_yolo_seg_label(self, label_path: str) -> List[Dict]:
        """
        解析YOLO-SEG标签文件
        
        参数:
            label_path: 标签文件路径
            
        返回:
            分割对象列表，每个对象包含类别和多边形点
        """
        objects = []
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 5:  # 至少需要类别 + 2个点 (4个坐标)
                continue
                
            class_id = int(parts[0])
            points = []
            
            # 读取坐标 (按x,y对解析)
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    x = float(parts[i])
                    y = float(parts[i+1])
                    points.append((x, y))
            
            objects.append({
                'class_id': class_id,
                'points': points
            })
            
        return objects
    
    def create_mask_from_polygon(self, polygon_points: List[Tuple[float, float]], 
                                 image_shape: Tuple[int, int]) -> np.ndarray:
        """
        从归一化的多边形点创建二值掩码
        
        参数:
            polygon_points: 归一化的多边形点列表 (x,y值在0-1之间)
            image_shape: 原始图像形状 (高度, 宽度)
            
        返回:
            二值掩码
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 转换归一化坐标到像素坐标
        points = []
        for x, y in polygon_points:
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            points.append([pixel_x, pixel_y])
        
        # 转换为numpy数组并调整形状
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 填充多边形
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def extract_foreground(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        使用掩码提取前景
        
        参数:
            image: 原始图像
            mask: 二值掩码
            
        返回:
            前景图像 (透明背景)
        """
        # 创建RGBA图像 (带Alpha通道)
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # 复制RGB通道
        rgba[:, :, 0:3] = image
        
        # 设置Alpha通道 (掩码区域可见，其他区域透明)
        rgba[:, :, 3] = mask
        
        return rgba
    
    def overlay_on_background(self, foreground: np.ndarray, 
                             background: np.ndarray) -> np.ndarray:
        """
        将前景叠加到背景上
        
        参数:
            foreground: 前景图像 (RGBA)
            background: 背景图像 (RGB)
            
        返回:
            合成图像
        """
        # 确保背景图像尺寸与前景相同
        if background.shape[0] != foreground.shape[0] or background.shape[1] != foreground.shape[1]:
            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # 创建背景的RGBA版本
        bg_rgba = np.zeros((background.shape[0], background.shape[1], 4), dtype=np.uint8)
        bg_rgba[:, :, 0:3] = background
        bg_rgba[:, :, 3] = 255  # 背景完全不透明
        
        # 前景掩码
        alpha = foreground[:, :, 3] / 255.0
        
        # 扩展alpha以便进行广播
        alpha = np.repeat(alpha[:, :, np.newaxis], 4, axis=2)
        
        # 混合前景和背景
        blended = foreground * alpha + bg_rgba * (1 - alpha)
        
        # 转换回无符号整数
        return blended.astype(np.uint8)
    
    def visualize_process(self, original_image: np.ndarray, 
                         masks: List[np.ndarray], 
                         result_image: np.ndarray,
                         class_ids: List[int],
                         filename: str) -> None:
        """
        可视化处理过程
        
        参数:
            original_image: 原始图像
            masks: 分割掩码列表
            result_image: 结果图像
            class_ids: 类别ID列表
            filename: 输出文件名
        """
        # 创建合并的掩码用于可视化
        combined_mask = np.zeros_like(original_image)
        
        # 根据类别给掩码添加颜色
        for mask, class_id in zip(masks, class_ids):
            color = self.class_colors.get(class_id, (255, 255, 255))  # 默认为白色
            for c in range(3):
                combined_mask[:, :, c] = np.maximum(
                    combined_mask[:, :, c], 
                    mask * color[c] // 255
                )
        
        # 创建带有轮廓的可视化
        contour_vis = original_image.copy()
        for mask, class_id in zip(masks, class_ids):
            color = self.class_colors.get(class_id, (255, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_vis, contours, -1, color, 2)
        
        # 创建掩码边界叠加
        mask_overlay = cv2.addWeighted(original_image, 0.7, combined_mask, 0.3, 0)
        
        # 创建图像网格
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(original_image)
        plt.title("origin")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(mask_overlay)
        plt.title("mask_overlay")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(contour_vis)
        plt.title("contour_vis")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(result_image[:,:,:3])  # 仅显示RGB通道
        plt.title("result_image")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, f"{filename}_process.png"))
        plt.close()
    
    def process_image(self, image_file: str) -> None:
        """
        处理单个图像
        
        参数:
            image_file: 图像文件路径
        """
        # 获取文件名和路径
        file_name = os.path.basename(image_file)
        base_name = os.path.splitext(file_name)[0]
        label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        # 检查标签文件是否存在
        if not os.path.exists(label_file):
            print(f"警告: 标签文件不存在: {label_file}")
            return
        
        # 读取图像
        image = cv2.imread(image_file)
        if image is None:
            print(f"错误: 无法读取图像: {image_file}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 解析YOLO-SEG标签
        objects = self.parse_yolo_seg_label(label_file)
        
        if not objects:
            print(f"警告: 没有找到有效的分割对象: {label_file}")
            return
        
        # 创建掩码
        masks = []
        class_ids = []
        for obj in objects:
            mask = self.create_mask_from_polygon(obj['points'], image.shape[:2])
            masks.append(mask)
            class_ids.append(obj['class_id'])
        
        # 合并所有掩码
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)
        
        # 提取前景
        foreground = self.extract_foreground(image, combined_mask)
        
        # 叠加到背景上
        result = self.overlay_on_background(foreground, self.bg_img)
        
        # 保存结果
        result_path = os.path.join(self.output_dir, f"{base_name}.png")
        result_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(result_path, result_rgb)
        
        # 可视化处理过程
        self.visualize_process(image, masks, result, class_ids, base_name)
        
        print(f"已处理: {file_name} -> {result_path}")
    
    def process_all(self) -> None:
        """处理目录中所有图像"""
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(self.rgb_dir, "*.png")) + \
                      glob.glob(os.path.join(self.rgb_dir, "*.jpg"))
        
        print(f"找到 {len(image_files)} 个图像文件.")
        
        # 处理每个图像
        for image_file in image_files:
            self.process_image(image_file)
        
        print(f"完成! 结果已保存到: {self.output_dir}")
        print(f"可视化结果已保存到: {self.visualization_dir}")
# TODO: bg 跑上了，整理下实验数据和实验思路，想一想下一步优化和问题，以及系统实验设计
# 主程序
if __name__ == "__main__":
    # 设置路径
    data_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/val"  # 更改为您的数据集目录
    rgb_dir = os.path.join(data_dir, "rgb_images")
    labels_dir = os.path.join(data_dir, "labels")
    background_image = os.path.join(data_dir, "image_crop_background.png")
    output_dir = os.path.join(data_dir, "output")
    visualization_dir = os.path.join(data_dir, "visualization")
    
    # 创建并运行背景替换器
    bg_changer = YoloSegBackgroundChanger(
        rgb_dir=rgb_dir,
        labels_dir=labels_dir,
        background_image=background_image,
        output_dir=output_dir,
        visualization_dir=visualization_dir
    )
    
    # 处理所有图像
    bg_changer.process_all()