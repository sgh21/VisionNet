import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random


# *: 灰度图像生成
def load_image(image_path):
    """
    加载图像并处理为RGB格式
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        numpy.ndarray: 加载的RGB图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 转换为RGB格式(OpenCV默认为BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_images(img1, img2):
    """调整图片大小使两张图片尺寸一致"""
    # 获取两张图片的高度和宽度的最小值
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    h = min(h1, h2)
    w = min(w1, w2)
    
    # 调整图片大小
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    
    return img1_resized, img2_resized

def compute_difference_map(template_img, target_img, method ='cv2'):
    """
    计算模板图像和目标图像之间的差异图
    
    Args:
        template_img (numpy.ndarray): 模板图像(RGB)
        target_img (numpy.ndarray): 目标图像(RGB)
        
    Returns:
        numpy.ndarray: 差异灰度图
    """
    # 确保图像尺寸相同
    if template_img.shape != target_img.shape:
        raise ValueError("模板图像和目标图像的尺寸必须相同")
    
    if method == 'cv2':
        # 使用OpenCV的absdiff函数计算差异
        diff = cv2.absdiff(template_img, target_img)
        # 转换为灰度图
        diff_gray = np.max(diff, axis=2)
    else:
        # 计算各通道的绝对差异
        diff_r = np.abs(template_img[:,:,0].astype(np.int32) - target_img[:,:,0].astype(np.int32))
        diff_g = np.abs(template_img[:,:,1].astype(np.int32) - target_img[:,:,1].astype(np.int32))
        diff_b = np.abs(template_img[:,:,2].astype(np.int32) - target_img[:,:,2].astype(np.int32))
        
        # 取各通道差异的最大值作为灰度图
        diff_gray = np.maximum(diff_r, np.maximum(diff_g, diff_b)).astype(np.uint8)
    
    return diff_gray

def crop_center(img, crop_width, crop_height,padding_box = None):
    h, w = img.shape[:2]
    if padding_box is not None:
        # 计算padding_box的坐标
        x1,y1,x2,y2 = padding_box
        img[:y1,:,:] = 0
        img[y2:,:,:] = 0
        img[:,:x1,:] = 0
        img[:,x2:,:] = 0

    x = w//2 - crop_width//2
    y = h//2 - crop_height//2

    return img[y:y+crop_height, x:x+crop_width]

def apply_nlm_filter(image, h=4, template_window_size=7, search_window_size=21):
    """
    应用非局部均值(Non-Local Means)滤波
    
    Args:
        image (numpy.ndarray): 输入灰度图像
        h (int): 滤波强度参数
        template_window_size (int): 模板窗口大小
        search_window_size (int): 搜索窗口大小
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    # 确保为单通道图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 应用NLM滤波
    filtered_img = cv2.fastNlMeansDenoising(
        image, 
        None, 
        h=h, 
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )
    
    return filtered_img

def binary_threshold(image, threshold=20, type='otsu'):
    """
    应用二值化阈值处理
    
    Args:
        image (numpy.ndarray): 输入图像
        threshold (int): 阈值
        
    Returns:
        numpy.ndarray: 二值化图像
    """
    # 确保为单通道图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 应用二值化
    if type == 'otsu':
        # 使用Otsu方法自动计算阈值
        ret, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif type == 'simple':
        ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif type == 'adaptive':
        # 自适应阈值
        binary_img = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            31, 
            5
        )
        ret = None
    return ret, binary_img

# *： 特征提取（边缘，掩膜，纹理）
def enhance_limited_range_contrast(gray_image, min_val=0, max_val=100, visualize=True, save_path=None):
    """
    对灰度图像中指定范围的像素值进行对比度增强，通过截取特定范围并重新映射到0-255
    
    Args:
        gray_image (numpy.ndarray): 输入灰度图像
        min_val (int): 要考虑的最小像素值，默认为0
        max_val (int): 要考虑的最大像素值，默认为100
        visualize (bool): 是否可视化结果，默认为True
        save_path (str): 可视化结果保存路径，若为None则不保存
        
    Returns:
        numpy.ndarray: 对比度增强后的灰度图像
    """
    # 确保输入为灰度图
    if len(gray_image.shape) > 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    
    # 复制原始图像以避免修改
    img_copy = gray_image.copy()
    
    # 统计原始像素值分布
    hist_original = cv2.calcHist([img_copy], [0], None, [256], [0, 256])
    
    # 计算实际像素值范围
    actual_min = np.min(img_copy)
    actual_max = np.max(img_copy)
    
    print(f"原始图像像素值范围: {actual_min} - {actual_max}")
    
    # 限制值到指定范围
    limited_img = np.clip(img_copy, min_val, max_val)
    
    # 线性拉伸：将限制范围内的值映射到0-255
    # 公式: (pixel - min_val) * (255 / (max_val - min_val))
    alpha = 255.0 / (max_val - min_val) if max_val > min_val else 0
    stretched_img = np.clip((limited_img - min_val) * alpha, 0, 255).astype(np.uint8)
    
    # 对线性拉伸后的图像进行直方图均衡化，进一步增强对比度
    equalized_img = cv2.equalizeHist(stretched_img)
    
    # 统计处理后的像素值分布
    hist_stretched = cv2.calcHist([stretched_img], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
    
    if visualize:
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Display original image
        plt.subplot(2, 3, 1)
        plt.imshow(img_copy, cmap='gray')
        plt.title(f'Original Grayscale [{actual_min}-{actual_max}]')
        plt.axis('off')
        
        # Display linearly stretched image
        plt.subplot(2, 3, 2)
        plt.imshow(stretched_img, cmap='gray')
        plt.title(f'Linear Stretch [{min_val}-{max_val}] → [0-255]')
        plt.axis('off')
        
        # Display equalized image
        plt.subplot(2, 3, 3)
        plt.imshow(equalized_img, cmap='gray')
        plt.title('After Histogram Equalization')
        plt.axis('off')
        
        # Display original histogram
        plt.subplot(2, 3, 4)
        plt.plot(hist_original, color='black')
        plt.xlim([0, 256])
        plt.title('Original Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Display histogram after linear stretching
        plt.subplot(2, 3, 5)
        plt.plot(hist_stretched, color='black')
        plt.xlim([0, 256])
        plt.title('Histogram after Linear Stretch')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Display histogram after equalization
        plt.subplot(2, 3, 6)
        plt.plot(hist_equalized, color='black')
        plt.xlim([0, 256])
        plt.title('Histogram after Equalization')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比度增强结果已保存至: {save_path}")
        
        # 显示结果
        plt.show()
    
    # 返回最终的均衡化图像
    return equalized_img, stretched_img

def adaptive_range_enhancement(gray_image, percentile_low=1, percentile_high=99, visualize=True, save_path=None):
    """
    基于百分位数自适应地增强灰度图像对比度
    
    Args:
        gray_image (numpy.ndarray): 输入灰度图像
        percentile_low (int): 低百分位数，用于确定下限截断点，默认为1
        percentile_high (int): 高百分位数，用于确定上限截断点，默认为99
        visualize (bool): 是否可视化结果，默认为True
        save_path (str): 可视化结果保存路径，若为None则不保存
        
    Returns:
        numpy.ndarray: 对比度增强后的灰度图像
    """
    # 确保输入为灰度图
    if len(gray_image.shape) > 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    
    # 计算百分位数阈值
    min_val = np.percentile(gray_image, percentile_low)
    max_val = np.percentile(gray_image, percentile_high)
    
    print(f"自适应范围: {min_val:.1f} ({percentile_low}百分位) - {max_val:.1f} ({percentile_high}百分位)")
    
    # 调用增强函数
    return enhance_limited_range_contrast(gray_image, min_val, max_val, visualize, save_path)


def morphological_operations(image, operation='open_close', kernel_size=3, iterations_open=1, iterations_close=1, visualize=True, save_path=None):
    """
    对图像执行形态学操作（开运算和/或闭运算）
    
    Args:
        image (numpy.ndarray): 输入灰度或二值图像
        operation (str): 执行的操作类型:
            - 'open': 仅执行开运算（先腐蚀后膨胀，去除小白点）
            - 'close': 仅执行闭运算（先膨胀后腐蚀，填充小黑洞）
            - 'open_close': 先开运算后闭运算（默认，先去噪再填充）
            - 'close_open': 先闭运算后开运算
        kernel_size (int): 形态学操作的结构元素大小
        iterations_open (int): 开运算的迭代次数
        iterations_close (int): 闭运算的迭代次数
        visualize (bool): 是否可视化结果
        save_path (str): 可视化结果保存路径
        
    Returns:
        numpy.ndarray: 形态学处理后的图像
    """
    # 确保输入为单通道图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 如果输入不是二值图像，进行二值化处理
    if image.dtype != np.uint8 or np.max(image) > 1:
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary_image = image.copy()
    
    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 保存中间结果用于可视化
    results = [binary_image.copy()]
    titles = ['Original Image']
    
    # 执行指定的形态学操作
    if operation == 'open' or operation == 'open_close' or operation == 'close_open':
        if operation == 'close_open':
            # 如果是先闭后开，则先保存闭运算的结果
            closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
            results.append(closed_image.copy())
            titles.append(f'Closing (k={kernel_size}, i={iterations_close})')
            opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
            results.append(opened_image)
            titles.append(f'Opening (k={kernel_size}, i={iterations_open})')
            processed_image = opened_image
        else:
            # 先执行开运算
            opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
            results.append(opened_image.copy())
            titles.append(f'Opening (k={kernel_size}, i={iterations_open})')
            processed_image = opened_image
            
            if operation == 'open_close':
                # 如果需要，再执行闭运算
                closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
                results.append(closed_image)
                titles.append(f'Closing (k={kernel_size}, i={iterations_close})')
                processed_image = closed_image
    
    elif operation == 'close':
        # 仅执行闭运算
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
        results.append(closed_image)
        titles.append(f'Closing (k={kernel_size}, i={iterations_close})')
        processed_image = closed_image
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    if visualize:
        # 确定布局
        n_images = len(results)
        cols = min(n_images, 4)  # 最多4列
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(cols*4, rows*4))
        
        # 显示原始图像和处理结果
        for i, (img, title) in enumerate(zip(results, titles)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Morphological processing results saved to: {save_path}")
        
        plt.show()
    
    return processed_image
def compute_gradient_map(gray_image, method='sobel', ksize=3, normalize=True, visualize=False, save_path=None):
    """
    计算灰度图像的梯度图
    
    参数:
        gray_image (numpy.ndarray): 输入灰度图像
        method (str): 梯度计算方法，可选:
            - 'sobel': Sobel算子 (默认)
            - 'scharr': Scharr算子 (边缘更敏感)
            - 'prewitt': Prewitt算子
            - 'roberts': Roberts交叉算子
            - 'laplacian': 拉普拉斯算子 (二阶导数)
        ksize (int): 核大小 (Sobel, Prewitt 和 Laplacian)，默认为3
        normalize (bool): 是否将梯度图归一化到0-255范围
        visualize (bool): 是否可视化梯度图
        save_path (str): 可视化结果保存路径，若为None则不保存
        
    返回:
        numpy.ndarray: 梯度图 (边缘强度图)
    """
    # 确保输入为灰度图
    if len(gray_image.shape) > 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    
    # 创建输出图像副本，避免修改原图
    img = gray_image.copy().astype(np.float32)
    
    # 选择梯度计算方法
    if method.lower() == 'sobel':
        # Sobel算子: 分别计算x和y方向梯度
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
        
        # 计算梯度幅值
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 计算梯度方向 (弧度)
        grad_dir = cv2.phase(grad_x, grad_y)
    
    elif method.lower() == 'scharr':
        # Scharr算子: 边缘更敏感的Sobel变体
        grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1)
        
        # 计算梯度幅值
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 计算梯度方向
        grad_dir = cv2.phase(grad_x, grad_y)
    
    elif method.lower() == 'prewitt':
        # 自定义Prewitt算子
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        grad_x = cv2.filter2D(img, -1, kernel_x)
        grad_y = cv2.filter2D(img, -1, kernel_y)
        
        # 计算梯度幅值
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 计算梯度方向
        grad_dir = cv2.phase(grad_x, grad_y)
    
    elif method.lower() == 'roberts':
        # 自定义Roberts交叉算子
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        grad_x = cv2.filter2D(img, -1, kernel_x)
        grad_y = cv2.filter2D(img, -1, kernel_y)
        
        # 计算梯度幅值
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # 计算梯度方向
        grad_dir = cv2.phase(grad_x, grad_y)
    
    elif method.lower() == 'laplacian':
        # 拉普拉斯算子: 计算二阶导数
        grad_mag = cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)
        
        # 取绝对值 (拉普拉斯结果可能为负)
        grad_mag = np.abs(grad_mag)
        
        # 拉普拉斯算子没有方向信息
        grad_dir = None
    
    else:
        raise ValueError(f"不支持的梯度计算方法: {method}, 支持的方法有: sobel, scharr, prewitt, roberts, laplacian")
    
    # 归一化梯度幅值到0-255范围
    if normalize:
        # 避免除以零
        if np.max(grad_mag) > 0:
            grad_mag = 255 * (grad_mag / np.max(grad_mag))
        grad_mag = grad_mag.astype(np.uint8)
    
    # 可视化结果
    if visualize:
        plt.figure(figsize=(15, 8))
        
        # 显示原始灰度图
        plt.subplot(1, 3, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.title('原始灰度图')
        plt.axis('off')
        
        # 显示梯度幅值图
        plt.subplot(1, 3, 2)
        plt.imshow(grad_mag, cmap='viridis')
        plt.title(f'{method.capitalize()} 梯度幅值')
        plt.axis('off')
        
        # 显示梯度方向图 (如果有)
        if grad_dir is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(grad_dir, cmap='hsv')
            plt.title('梯度方向')
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"梯度图结果已保存至: {save_path}")
        
        plt.show()
    
    return grad_mag

def detect_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """
    使用Canny边缘检测器检测边缘
    
    Args:
        image (numpy.ndarray): 输入灰度图像
        low_threshold (int): 低阈值
        high_threshold (int): 高阈值
        aperture_size (int): Sobel算子的孔径大小
        
    Returns:
        numpy.ndarray: 边缘图像
    """
    # 确保为单通道图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 应用Canny边缘检测
    edges = cv2.Canny(
        image, 
        low_threshold, 
        high_threshold, 
        apertureSize=aperture_size
    )
    
    return edges

def overlay_edges_on_image(image, edges, color=(0, 255, 0), thickness=2):
    """
    将检测到的边缘叠加到原始图像上
    
    Args:
        image (numpy.ndarray): 原始彩色图像
        edges (numpy.ndarray): 边缘图像（二值图像，边缘为255，背景为0）
        color (tuple): 边缘的颜色，格式为(B,G,R)
        thickness (int): 边缘线条的粗细
        
    Returns:
        numpy.ndarray: 带有叠加边缘的图像
    """
    # 确保输入图像是彩色的
    if len(image.shape) == 2 or image.shape[2] == 1:
        # 如果是灰度图，转换为彩色图
        image_with_edges = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # 如果已经是彩色图，创建副本
        image_with_edges = image.copy()
    
    # 找到边缘的位置（边缘像素值为255）
    edge_pixels = np.where(edges == 255)
    
    # 将边缘位置设置为指定颜色
    # 对于每个找到的边缘像素
    if thickness == 1:
        image_with_edges[edge_pixels[0], edge_pixels[1]] = color
    else:
        # 找到轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        cv2.drawContours(image_with_edges, contours, -1, color, thickness)
    
    return image_with_edges


# 添加连通性分析函数
def connected_components_analysis(binary_image, connectivity=8, min_area=100, max_area=None):
    """
    对二值图像进行连通性分析，返回标记的实例图
    
    Args:
        binary_image (numpy.ndarray): 二值图像
        connectivity (int): 连通性，可以是4或8
        min_area (int): 最小区域面积，小于此面积的区域将被过滤掉
        max_area (int): 最大区域面积阈值，大于此面积的区域将被过滤掉，None表示不设上限
        
    Returns:
        tuple: (labeled_image, stats, centroids)
            - labeled_image: 标记好的图像，每个连通区域有唯一的ID
            - stats: 每个连通区域的统计信息 [x, y, width, height, area]
            - centroids: 每个连通区域的中心点坐标
            - filtered_labels: 过滤后的连通区域标签列表
    """
    # 确保输入为二值图像
    if len(binary_image.shape) > 2:
        raise ValueError("输入必须是二值图像")
    
    # 连通性分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, 
        connectivity=connectivity
    )
    
        # 过滤区域（标签0通常是背景）
    filtered_labels = []
    for i in range(1, num_labels):  # 从1开始，跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):
            filtered_labels.append(i)
    
    return labels, stats, centroids, filtered_labels

def analyze_regions_with_flood_fill(binary_image, tolerance=10, min_area=50, max_area=None, connectivity=4):
    """
    使用洪水填充算法对二值图像进行区域分割
    
    Args:
        binary_image (numpy.ndarray): 输入的二值图像
        tolerance (int): 像素值相似度容差，控制填充范围
        min_area (int): 最小区域面积阈值，小于此面积的区域将被过滤掉
        max_area (int): 最大区域面积阈值，大于此面积的区域将被过滤掉，None表示不设上限
        connectivity (int): 连通性类型，4或8
        
    Returns:
        tuple: (region_masks, region_stats, seed_points)
            - region_masks: 字典，键为区域ID，值为该区域的二值掩码
            - region_stats: 字典，键为区域ID，值为该区域的统计信息 [x, y, width, height, area]
            - seed_points: 字典，键为区域ID，值为该区域的种子点坐标
    """
    # 确保输入为二值图像
    if len(binary_image.shape) > 2:
        raise ValueError("输入必须是二值图像")
    
    # 确保图像是uint8类型
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    
    # 确保是标准二值图像（0和255）
    if set(np.unique(binary_image)).issubset({0, 1}):
        binary_image = binary_image * 255
    
    # 创建辅助变量
    height, width = binary_image.shape
    processed_mask = np.zeros((height, width), dtype=np.uint8)  # 记录已处理的像素
    region_id = 1
    region_masks = {}
    region_stats = {}
    seed_points = {}
    
    # 洪水填充的标志位
    flood_flags = (
        connectivity |  # 连通性
        (255 << 8) |  # 填充值
        cv2.FLOODFILL_FIXED_RANGE |  # 使用固定范围比较
        cv2.FLOODFILL_MASK_ONLY  # 只填充掩码
    )
    
    # 创建前景点列表（白色区域的点）
    foreground_points = np.column_stack(np.where(binary_image == 255))
    np.random.shuffle(foreground_points)  # 随机打乱，避免填充顺序偏差
    
    # 逐点尝试填充
    for y, x in foreground_points:
        # 如果当前点已经被处理过，跳过
        if processed_mask[y, x] > 0:
            continue
        
        # 创建洪水填充掩码（需要比原图大两个像素）
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        
        # 执行洪水填充
        seed_point = (x, y)
        cv2.floodFill(
            binary_image.copy(),  # 使用副本避免修改原图
            flood_mask,
            seed_point,
            255,
            (tolerance,) * 3,
            (tolerance,) * 3,
            flood_flags
        )
        
        # 提取有效区域掩码（去掉额外的边框）
        region_mask = flood_mask[1:-1, 1:-1]
        
        # 计算区域面积
        area = np.sum(region_mask > 0)
        
        # 检查面积是否符合条件
        if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):
            # 计算区域的统计信息
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                stats_info = [x, y, w, h, area]
                
                # 保存区域信息
                region_masks[region_id] = region_mask
                region_stats[region_id] = stats_info
                seed_points[region_id] = seed_point
                
                # 更新区域ID
                region_id += 1
        
        # 更新已处理掩码
        processed_mask = processed_mask | region_mask
    
    print(f"通过洪水填充找到 {len(region_masks)} 个符合条件的区域")
    
    return region_masks, region_stats, seed_points


# *： 结果可视化
# 修改可视化函数，解决多图显示问题
def visualize_images(images, titles=None, figsize=None, cmap=None, rows=None, cols=None, save_path=None, display=True, block=True):
    """
    通用图像可视化函数，可展示任意数量的图像并自动排布

    Args:
        images (list): 图像列表，每个元素是一个numpy数组
        titles (list): 标题列表，与images列表一一对应。若为None则不显示标题
        figsize (tuple): 图形大小 (width, height)，若为None则自动计算
        cmap (str/list): 颜色映射，可以是单一字符串应用到所有图像，或者列表为每个图像单独指定
        rows (int): 自定义行数，若为None则自动计算
        cols (int): 自定义列数，若为None则自动计算
        save_path (str): 保存路径，若为None则不保存
        display (bool): 是否尝试在屏幕上显示图像，默认为True
        block (bool): 显示图像时是否阻塞程序执行，默认为True

    Returns:
        matplotlib.figure.Figure: 生成的图像对象
    """
    # 处理空输入
    if not images:
        raise ValueError("必须提供至少一张图片")
    
    # 获取图片数量
    n_images = len(images)
    
    # 处理标题
    if titles is None:
        titles = [f"图像 {i+1}" for i in range(n_images)]
    elif len(titles) != n_images:
        raise ValueError("标题数量必须与图像数量相同")
    
    # 自动计算行列数
    if rows is None and cols is None:
        # 自动确定行列数：尽量接近正方形布局
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # 自动计算图形大小
    if figsize is None:
        # 根据图像数量和行列数计算合适的图形大小
        base_size = 3  # 基础单元大小
        figsize = (cols * base_size, rows * base_size)
    
    # 创建图形和子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    # 处理颜色映射
    if cmap is None:
        # 默认使用灰度图显示单通道图像，彩色图显示多通道图像
        cmaps = [None] * n_images
    elif isinstance(cmap, str):
        # 对所有图像使用相同的颜色映射
        cmaps = [cmap] * n_images
    else:
        # 对每个图像使用指定的颜色映射
        cmaps = cmap
        if len(cmaps) != n_images:
            raise ValueError("颜色映射列表长度必须等于图像数量")
    
    # 绘制图像
    for i in range(n_images):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        # 获取当前图像
        img = images[i]
        
        # 自动判断使用的颜色映射
        if cmaps[i] is None and (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
            # 单通道图像默认使用灰度
            curr_cmap = 'gray'
        else:
            curr_cmap = cmaps[i]
        
        # 显示图像
        ax.imshow(img, cmap=curr_cmap)
        ax.set_title(titles[i])
        
        # 关闭刻度
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 隐藏多余的子图
    for i in range(n_images, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    # 尝试显示图像
    if display:
        try:
            if block:
                plt.show()  # 阻塞显示
            else:
                plt.show(block=False)  # 非阻塞显示
                plt.pause(0.1)  # 确保图像渲染
        except Exception as e:
            print(f"无法显示图像: {e}, 请检查matplotlib配置或环境")
            print("提示: 如果在无GUI环境下运行，可以将display参数设置为False")
    
    return fig

def visualize_components(image, labels, filtered_labels, stats=None, centroids=None, 
                         add_labels=True, add_boxes=True, alpha=0.7, seed=None):
    """
    使用随机颜色可视化连通区域
    
    Args:
        image (numpy.ndarray): 原始图像，用作背景
        labels (numpy.ndarray): 标记图像，每个像素值为该连通区域的标签
        filtered_labels (list): 要显示的连通区域标签列表
        stats (numpy.ndarray, optional): 连通区域统计信息
        centroids (numpy.ndarray, optional): 连通区域中心点坐标
        add_labels (bool): 是否在连通区域添加标签编号
        add_boxes (bool): 是否添加外接矩形框
        alpha (float): 叠加颜色的透明度 (0-1)
        seed (int): 随机数种子，用于生成一致的随机颜色
        
    Returns:
        numpy.ndarray: 带有随机颜色标记的连通区域可视化图像
    """
    # 设置随机数种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保输入图像是3通道RGB格式
    if len(image.shape) == 2:
        # 灰度图转RGB
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # 已经是RGB格式，创建副本
        vis_img = image.copy()
    else:
        raise ValueError("输入图像必须是灰度图或RGB图像")
    
    # 创建结果图像
    height, width = labels.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个过滤后的标签生成随机颜色
    colors = []
    for _ in range(len(filtered_labels)):
        # 生成饱和的随机颜色，避免太暗或太亮
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        colors.append((r, g, b))
    
    # 为每个连通区域上色
    for i, label in enumerate(filtered_labels):
        # 创建当前标签的掩码
        mask = labels == label
        
        # 为掩码区域上色
        result[mask] = colors[i]
        
        # 处理统计信息
        if stats is not None and centroids is not None:
            # 获取当前区域的统计信息
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            
            # 在结果图像上添加外接矩形框
            if add_boxes:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 1)
            
            # 添加标签文本
            if add_labels:
                # 获取中心点坐标
                cx, cy = centroids[label]
                cx, cy = int(cx), int(cy)
                
                # 确定标签位置（避免文本超出图像边界）
                label_x = max(0, min(width - 30, cx - 10))
                label_y = max(15, min(height - 5, cy))
                
                # 在结果图像上添加标签编号
                cv2.putText(
                    result, 
                    f"{label}", 
                    (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
    
    # 将连通区域叠加到原始图像上
    overlay = cv2.addWeighted(vis_img, 1 - alpha, result, alpha, 0)
    
    return overlay

def visualize_flood_fill_regions(image, region_masks, region_stats=None, seed_points=None, 
                                add_labels=True, add_boxes=True, alpha=0.7, seed=None):
    """
    使用随机颜色可视化洪水填充得到的区域
    
    Args:
        image (numpy.ndarray): 原始图像，用作背景
        region_masks (dict): 区域掩码字典，键为区域ID，值为掩码
        region_stats (dict): 区域统计信息字典，键为区域ID，值为统计信息 [x, y, width, height, area]
        seed_points (dict): 种子点字典，键为区域ID，值为种子点坐标
        add_labels (bool): 是否在区域添加标签编号
        add_boxes (bool): 是否添加外接矩形框
        alpha (float): 叠加颜色的透明度 (0-1)
        seed (int): 随机数种子，用于生成一致的随机颜色
        
    Returns:
        numpy.ndarray: 带有随机颜色标记的区域可视化图像
    """
    # 设置随机数种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保输入图像是3通道RGB格式
    if len(image.shape) == 2:
        # 灰度图转RGB
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # 已经是RGB格式，创建副本
        vis_img = image.copy()
    else:
        raise ValueError("输入图像必须是灰度图或RGB图像")
    
    # 获取图像尺寸
    height, width = vis_img.shape[:2]
    
    # 创建结果图像
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个区域生成随机颜色
    region_ids = list(region_masks.keys())
    colors = []
    for _ in range(len(region_ids)):
        # 生成饱和的随机颜色，避免太暗或太亮
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        colors.append((r, g, b))
    
    # 为每个区域上色
    for i, region_id in enumerate(region_ids):
        mask = region_masks[region_id] > 0
        result[mask] = colors[i]
        
        # 处理统计信息
        if region_stats is not None:
            x, y, w, h, area = region_stats[region_id]
            
            # 在结果图像上添加外接矩形框
            if add_boxes:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 1)
            
            # 添加标签文本
            if add_labels:
                # 确定标签位置（避免文本超出图像边界）
                label_x = max(0, min(width - 30, x + w // 2 - 10))
                label_y = max(15, min(height - 5, y + h // 2))
                
                # 在结果图像上添加标签编号
                cv2.putText(
                    result, 
                    f"{region_id}", 
                    (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
        
        # 标记种子点
        if seed_points is not None and add_labels:
            seed_x, seed_y = seed_points[region_id]
            cv2.circle(result, (seed_x, seed_y), 2, (255, 255, 255), -1)
    
    # 将区域叠加到原始图像上
    overlay = cv2.addWeighted(vis_img, 1 - alpha, result, alpha, 0)
    
    return overlay

# 修改主函数以包含实例分割
def main():
    # img_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_3560P2.png'
    # img_template_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_raw1.png'
    
    img_path = '/home/sgh/data/WorkSpace/BTBInsertionV2/documents/train_data_0411/original/gel_images_crop_01_3560P/gel_image_3560P_156.png'
    img_template_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_template.png'
    # 创建输出目录
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    img = load_image(img_path)
    img_template = load_image(img_template_path)
    
    # 确保图像大小一致
    if img.shape != img_template.shape:
        img_template, img = resize_images(img_template, img)
    
    # 计算差异图
    diff_map = compute_difference_map(img_template, img, method='cv2')
    
    # 应用NLM滤波
    filtered_diff = apply_nlm_filter(diff_map, h=4)
    # 计算差异图的梯度图
    grad_map = compute_gradient_map(filtered_diff, method='sobel', ksize=3, visualize=False)
    # 使用直方图均衡化增强对比度
    _, equalization_diff = adaptive_range_enhancement(filtered_diff, percentile_low=0, percentile_high=100, visualize=False)
    
    # TODO：不再自适应二值化，而是将Mask应用到原图上采样，从而获得灰度图
    # 二值化图像
    ret, binary_diff = binary_threshold(equalization_diff, type='adaptive')
    
    # 可视化初始结果
    images_list1 = [img_template, img, diff_map, filtered_diff,  binary_diff, grad_map]
    titles_list1 = ['template', 'dst', 'diff_map', 'NLM-filtered',  'binary_diff', 'grad_map' ]

    visualize_images(
        images_list1, 
        titles=titles_list1, 
        figsize=(20, 10), 
        save_path=os.path.join(output_dir, 'stage1_preprocessing.png'),
        display=True,
    )
    #  形态学操作（开运算和闭运算）
    processed_diff = morphological_operations(
        binary_diff, 
        operation='open_close', 
        kernel_size=3, 
        iterations_open=1, 
        iterations_close=1,
        visualize=False,
        save_path=os.path.join(output_dir, 'stage2_morphological_operations.png')
    )
    binary_inverted = cv2.bitwise_not(processed_diff)
    # 执行连通性分析
    labels, stats, centroids, filtered_labels = connected_components_analysis(
        binary_inverted,
        connectivity=4,
        min_area=500,   # 最小区域面积，可以调整
    )
    # TODO:格局外轮廓的矩形度来过滤背景
    from utils.TransUtils import filter_by_rectangularity
    rect_filtered_labels, rectangularity_values, mask = filter_by_rectangularity(
        binary_inverted,
        labels,
        stats,
        centroids=centroids,
        filtered_labels=filtered_labels,
        min_rectangularity=0.65,  # 最小矩形度，可以调整
        use_convex_hull=False,
        visualize=False,
        original_image=img,
        output_mask= True
    )

    # white_background = np.ones_like(img) * 255
    # # 可视化连通区域
    # components_vis = visualize_components(
    #     white_background,  # 使用原始图像作为背景
    #     labels,
    #     rect_filtered_labels,
    #     stats,
    #     centroids,
    #     add_labels=True,
    #     add_boxes=False,
    #     alpha=0.6  # 透明度
    # )

    # 显示结果
    images_list2 = [
        img, 
        binary_diff, 
        processed_diff, 
        # components_vis,
        mask
    ]
    
    titles_list2 = [
        'Original Image', 
        'Binary Image', 
        'Morphologically Processed', 
        # 'Connected Components',
        'Filtered Mask'
    ]

    visualize_images(
        images_list2, 
        titles=titles_list2, 
        figsize=(20, 10), 
        save_path=os.path.join(output_dir, 'connected_components_analysis.png'),
        display=True
    )

if __name__ == "__main__":
    # 示例用法
    main()
    
# def main():
#     import time
#     import matplotlib.pyplot as plt
    
#     # 用于存储性能数据的字典
#     performance_data = {
#         'step_names': [],
#         'durations': []
#     }
    
#     # 记录处理步骤的函数
#     def record_step(step_name, func, *args, **kwargs):
#         start_time = time.time()
        
#         # 执行函数并计时
#         result = func(*args, **kwargs)
        
#         # 记录耗时
#         duration = time.time() - start_time
#         performance_data['step_names'].append(step_name)
#         performance_data['durations'].append(duration)
#         print(f"Step '{step_name}' took: {duration:.4f} seconds")
        
#         return result
    
#     img_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_3540P3.png'
#     img_template_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_raw1.png'
    
#     # 创建输出目录
#     output_dir = './output'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 加载图像
#     img = record_step('Load Target Image', load_image, img_path)
#     img_template = record_step('Load Template Image', load_image, img_template_path)
    
#     # 确保图像大小一致
#     if img.shape != img_template.shape:
#         img_template, img = record_step('Resize Images', resize_images, img_template, img)
    
#     # 计算差异图
#     diff_map = record_step('Compute Difference Map', compute_difference_map, img_template, img, method='cv2')
    
#     # 应用NLM滤波
#     filtered_diff = record_step('NLM Filtering', apply_nlm_filter, diff_map, h=4)
    
#     # 使用直方图均衡化增强对比度
#     _, equalization_diff = record_step('Contrast Enhancement', 
#                                      adaptive_range_enhancement, 
#                                      filtered_diff, 
#                                      percentile_low=0, 
#                                      percentile_high=100, 
#                                      visualize=False)
    
#     # 二值化图像
#     ret, binary_diff = record_step('Binary Thresholding', binary_threshold, equalization_diff, type='adaptive')
    
#     # 形态学操作（开运算和闭运算）
#     processed_diff = record_step('Morphological Operations', 
#                                morphological_operations,
#                                binary_diff, 
#                                operation='close_open', 
#                                kernel_size=3, 
#                                iterations_open=1, 
#                                iterations_close=1,
#                                visualize=False,
#                                save_path=os.path.join(output_dir, 'stage2_morphological_operations.png'))
    
#     binary_inverted = record_step('Image Inversion', cv2.bitwise_not, processed_diff)

#     # 执行连通性分析
#     labels, stats, centroids, filtered_labels = record_step('Connected Components Analysis', 
#                                                          connected_components_analysis,
#                                                          binary_inverted,
#                                                          connectivity=4,
#                                                          min_area=500)
    
#     # 矩形度过滤
#     from utils.TouchMapUtils import filter_by_rectangularity
#     rect_filtered_labels, rectangularity_values = record_step('Rectangularity Filtering', 
#                                                            filter_by_rectangularity,
#                                                            binary_inverted,
#                                                            labels,
#                                                            stats,
#                                                            centroids=centroids,
#                                                            filtered_labels=filtered_labels,
#                                                            min_rectangularity=0.55,
#                                                            use_convex_hull=False,
#                                                            visualize=False,
#                                                            original_image=img)

#     white_background = np.ones_like(img) * 255
    
#     # 可视化连通区域
#     components_vis = record_step('Generate Component Visualization', 
#                                visualize_components,
#                                white_background,
#                                labels,
#                                rect_filtered_labels,
#                                stats,
#                                centroids,
#                                add_labels=True,
#                                add_boxes=False,
#                                alpha=0.6)
    
#     # 总处理时间
#     total_time = sum(performance_data['durations'])
#     print(f"Total processing time: {total_time:.4f} seconds")
    
#     # 生成甘特图
#     create_gantt_chart(performance_data, os.path.join(output_dir, 'performance_gantt_chart.png'))

# def create_gantt_chart(performance_data, save_path=None):
#     """
#     创建性能甘特图
    
#     Args:
#         performance_data (dict): 包含'step_names', 'durations'的字典
#         save_path (str, optional): 保存路径
#     """
#     # 提取数据
#     step_names = performance_data['step_names']
#     durations = performance_data['durations']
    
#     # 计算总时间和每个步骤的百分比
#     total_time = sum(durations)
#     percentages = [100 * d / total_time for d in durations]
    
#     # 计算每个步骤的起始位置（基于序号，而不是实际时间）
#     positions = []
#     current_pos = 0
#     for duration in durations:
#         positions.append(current_pos)
#         current_pos += duration
    
#     # 创建甘特图
#     fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    
#     # 自定义颜色映射
#     cmap = plt.cm.get_cmap('viridis', len(step_names))
    
#     # 反转y轴顺序，使处理流程从上到下显示
#     ax.set_yticks(range(len(step_names)))
#     ax.set_yticklabels(step_names[::-1])  # 反转标签顺序
#     ax.set_ylim(-0.5, len(step_names) - 0.5)
    
#     # 添加水平网格线
#     ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
#     # 绘制条形（从下到上）
#     for i, (name, duration, percent, position) in enumerate(zip(
#             step_names[::-1],  # 反转顺序
#             durations[::-1], 
#             percentages[::-1], 
#             positions[::-1])):
#         # 计算条形相对宽度以保持比例
#         bar = ax.barh(i, duration, height=0.5, 
#                 color=cmap(1-i/len(step_names)), alpha=0.8, 
#                 label=f'{name} ({percent:.1f}%)')
        
#         # 在条形上添加持续时间文本
#         text_x = duration / 2
#         text_y = i
#         ax.text(text_x, text_y, f'{duration:.3f}s', 
#                 ha='center', va='center', color='white', fontweight='bold')
    
#     # 设置图表属性
#     ax.set_xlabel('Processing Time (seconds)')
#     ax.set_title('Performance Analysis of Image Processing Pipeline')
    
#     # 添加图例
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), 
#               ncol=2, frameon=True, fancybox=True, shadow=True)
    
#     plt.tight_layout()
    
#     # 保存图表
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Performance chart saved to: {save_path}")
    
#     plt.show()
# if __name__ == "__main__":
#     # 示例用法
#     main()