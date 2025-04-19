import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog

class DeepEdgeDetector:
    """
    使用OpenCV的深度学习模型进行边缘检测
    支持HED (Holistically-Nested Edge Detection) 模型
    """
    
    def __init__(self, models_dir='./weights'):
        """
        初始化边缘检测器
        
        Args:
            models_dir: 模型文件存放目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # 模型文件路径
        self.hed_model = self.models_dir / 'hed_pretrained_bsds.caffemodel'
        self.hed_prototxt = self.models_dir / 'deploy.prototxt'
        
        # 检查模型文件
        self._check_models()
        
    def _check_models(self):
        """检查模型文件是否存在，不存在则提供下载链接"""
        missing_models = []
        
        # 检查HED模型
        if not self.hed_model.exists() or not self.hed_prototxt.exists():
            missing_models.append(('HED', [
                f"wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt -O {self.hed_prototxt}",
                f"wget http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel -O {self.hed_model}"
            ]))
        
        # 如果有缺失的模型，打印下载指令
        if missing_models:
            print("\n===== 缺失模型文件 =====")
            for model_name, commands in missing_models:
                print(f"\n缺失{model_name}模型文件，请使用以下命令下载:")
                for cmd in commands:
                    print(f"  {cmd}")
            print("\n或者手动下载文件并放置在目录：", self.models_dir)
        
        return len(missing_models) == 0
    
    def detect_edges(self, image, method='hed', post_process=True, threshold=25):
        """
        检测图像边缘
        
        Args:
            image: 输入图像，可以是RGB、BGR或灰度图
            method: 边缘检测方法，目前仅支持'hed'
            post_process: 是否对结果进行后处理
            threshold: 二值化阈值
            
        Returns:
            edges: 边缘检测结果
            time_taken: 处理时间
        """
        # 检查图像是否为灰度图
        is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
        
        # 如果是灰度图，转换为3通道
        if is_grayscale:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
        # 目前仅支持HED方法
        if method.lower() == 'hed':
            return self._hed_edge_detection(image, post_process, threshold)
        else:
            raise ValueError(f"不支持的方法: {method}. 目前仅支持 'hed'")
    
    def _hed_edge_detection(self, image, post_process=True, threshold=25):
        """
        使用HED模型检测边缘
        
        Args:
            image: 输入图像，BGR格式
            post_process: 是否进行后处理
            threshold: 二值化阈值
            
        Returns:
            edges: 边缘检测结果
            time_taken: 处理时间
        """
        # 检查模型是否存在
        if not self.hed_model.exists() or not self.hed_prototxt.exists():
            print("HED模型文件不存在，请先下载模型")
            return None, 0
        
        start_time = time.time()
        
        try:
            # 加载模型
            net = cv2.dnn.readNetFromCaffe(str(self.hed_prototxt), str(self.hed_model))
            
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 准备输入
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(
                rgb_image, 
                scalefactor=1.0, 
                size=(width, height),
                mean=(104.00698793, 116.66876762, 122.67891434),
                swapRB=False,
                crop=False
            )
            
            # 设置输入
            net.setInput(blob)
            
            # 前向传播
            hed_output = net.forward()
            
            # 处理输出
            edges = hed_output[0, 0]
            
            # 转换为8位无符号整数
            edges = (edges * 255).astype(np.uint8)
            
            # 可选的后处理
            if post_process:
                edges = self._post_process_edges(edges, threshold)
            
            time_taken = time.time() - start_time
            
            return edges, time_taken
            
        except Exception as e:
            print(f"HED边缘检测出错: {e}")
            return None, 0
    
    def _post_process_edges(self, edges, threshold=25):
        """
        对边缘检测结果进行后处理
        
        Args:
            edges: 边缘检测结果
            threshold: 二值化阈值
            
        Returns:
            processed_edges: 处理后的边缘图
        """
        # 二值化
        _, binary_edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        
        # 创建核
        kernel = np.ones((3, 3), np.uint8)
        
        # 先腐蚀以细化边缘
        eroded_edges = cv2.erode(binary_edges, kernel, iterations=1)
        
        # 再膨胀以连接断开的边缘
        dilated_edges = cv2.dilate(eroded_edges, kernel, iterations=1)
        
        return dilated_edges
    
    def get_edge_overlay(self, image, edges, alpha=0.7, color=(0, 0, 255)):
        """
        创建边缘叠加在原图上的效果
        
        Args:
            image: 原始图像
            edges: 边缘图
            alpha: 透明度
            color: 边缘颜色 (BGR格式)
            
        Returns:
            overlay: 叠加后的图像
        """
        # 确保图像为3通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 创建边缘掩码
        mask = edges > 0
        
        # 创建叠加图
        overlay = image.copy()
        overlay[mask] = color
        
        # 混合原图和叠加图
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return blended

def select_image():
    """使用文件对话框选择图片"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择图像",
        filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return file_path

def visualize_results(image, results, output_path=None):
    """
    可视化边缘检测结果
    
    Args:
        image: 原始图像
        results: 字典，包含不同方法的边缘检测结果和处理时间
        output_path: 保存图像的路径
    """
    # 确保有结果
    if not results:
        print("没有可视化的结果")
        return
    
    # 计算布局
    n_methods = len(results)
    
    # 使用2行布局: 第一行为原图和边缘图，第二行为叠加效果
    n_cols = n_methods + 1  # +1 为原图
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, n_cols, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 边缘检测结果
    for i, (method, (edges, overlay, time_taken)) in enumerate(results.items(), 1):
        # 边缘图 - 第一行
        plt.subplot(2, n_cols, i + 1)  # 位置2, 3, ...
        plt.imshow(edges, cmap='gray')
        plt.title(f'{method} 边缘\n({time_taken:.3f}秒)')
        plt.axis('off')
        
        # 叠加图 - 第二行
        plt.subplot(2, n_cols, n_cols + i)  # 位置 n_cols+1, n_cols+2, ...
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'{method} 叠加效果')
        plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()

def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(description='使用OpenCV的深度学习模型进行边缘检测')
    parser.add_argument('--image', help='输入图像路径，如不提供则使用文件选择对话框')
    parser.add_argument('--models', default='./weights', help='模型目录路径')
    parser.add_argument('--output', default='./output', help='输出目录路径')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 选择或加载图像
    if args.image:
        image_path = args.image
    else:
        print("请选择一张图像...")
        image_path = select_image()
        if not image_path:
            print("未选择图像，程序退出")
            return
    
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 检查图像通道
    is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
    if is_grayscale:
        print("检测到灰度图像")
    else:
        print("检测到彩色图像")
    
    # 初始化边缘检测器
    detector = DeepEdgeDetector(models_dir=args.models)
    
    # 结果字典
    results = {}
    
    # 执行HED边缘检测
    print("正在使用HED模型进行边缘检测...")
    hed_edges, hed_time = detector.detect_edges(image, method='hed')
    if hed_edges is not None:
        hed_overlay = detector.get_edge_overlay(image, hed_edges)
        results['HED'] = (hed_edges, hed_overlay, hed_time)
        print(f"HED边缘检测完成，耗时: {hed_time:.3f}秒")
    
    # 获取传统Canny边缘检测作为比较
    print("正在使用Canny边缘检测作为比较...")
    start_time = time.time()
    
    # 如果是彩色图像，转为灰度
    if not is_grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() if len(image.shape) == 2 else image[:,:,0]
    
    # 高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    canny_edges = cv2.Canny(blurred, 50, 150)
    canny_time = time.time() - start_time
    
    # 创建Canny叠加图
    canny_overlay = detector.get_edge_overlay(image, canny_edges)
    results['Canny'] = (canny_edges, canny_overlay, canny_time)
    print(f"Canny边缘检测完成，耗时: {canny_time:.3f}秒")
    
    # 生成输出文件名
    image_name = Path(image_path).stem
    output_path = output_dir / f"{image_name}_edge_detection_results.png"
    
    # 可视化结果
    print("正在生成可视化结果...")
    visualize_results(image, results, output_path)
    
    # 保存单独的边缘检测结果
    for method, (edges, overlay, _) in results.items():
        edge_path = output_dir / f"{image_name}_{method}_edges.png"
        overlay_path = output_dir / f"{image_name}_{method}_overlay.png"
        
        cv2.imwrite(str(edge_path), edges)
        cv2.imwrite(str(overlay_path), overlay)
    
    print(f"所有结果已保存到: {output_dir}")
    print("程序完成!")

if __name__ == "__main__":
    main()