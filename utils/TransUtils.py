import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from typing import Optional, Tuple, Dict, List, Union


class TouchWeightMapTransform:
    """
    触觉图像处理转换类，将触觉图像转换为权重掩码图

    处理流程:
    1. 与模板图像计算差异
    2. 降噪滤波
    3. 对比度增强
    4. 二值化处理
    5. 形态学操作
    6. 连通区域分析和几何特性过滤
    
    参数:
        template_path (str): 模板图像路径
        template_img (torch.Tensor, PIL.Image, np.ndarray): 模板图像对象
        min_area (int): 最小连通区域面积，默认为500
        min_rectangularity (float): 最小矩形度阈值，默认为0.55
        morph_operation (str): 形态学操作类型，默认为'close_open'
        morph_kernel_size (int): 形态学操作核大小，默认为3
        to_tensor (bool): 是否将结果转换为tensor，默认为True
        normalized (bool): 是否将结果归一化到[0,1]范围，默认为True
    """
    def __init__(
        self,
        template_path: Optional[str] = None,
        template_img: Optional[Union[torch.Tensor, Image.Image, np.ndarray]] = None,
        min_area: int = 500,
        min_rectangularity: float = 0.55,
        morph_operation: str = 'close_open',
        morph_kernel_size: int = 3,
        canvas_size: tuple[int, int] = (560, 560),
        M : np.ndarray = None,
        to_tensor: bool = True,
        normalized: bool = True,
    ):
        # 验证模板输入
        if template_path is not None:
            self.template_img = self._load_image(template_path)
        elif template_img is not None:
            self.template_img = self._convert_to_numpy(template_img)
        else:
            raise ValueError("必须提供template_path或template_img参数")
        
        # 存储参数
        self.min_area = min_area
        self.min_rectangularity = min_rectangularity
        self.morph_operation = morph_operation
        self.morph_kernel_size = morph_kernel_size
        self.canvas_size = canvas_size
        self.M = M
        self.to_tensor = to_tensor
        self.normalized = normalized
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像并处理为RGB格式"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式(OpenCV默认为BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def _convert_to_numpy(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> np.ndarray:
        """将不同格式的图像转换为numpy数组"""
        if isinstance(image, torch.Tensor):
            # 处理PyTorch张量
            if image.ndim == 4:  # [B,C,H,W]
                image = image.squeeze(0)  # 移除批次维度
            
            if image.ndim == 3:  # [C,H,W]
                image = image.permute(1, 2, 0)  # 转为[H,W,C]
            
            # 如果是归一化的张量，转换为0-255范围
            if image.max() <= 1.0:
                image = image * 255.0
                
            return image.cpu().numpy().astype(np.uint8)
        
        elif isinstance(image, Image.Image):
            # 处理PIL图像
            return np.array(image)
        
        elif isinstance(image, np.ndarray):
            # 已经是numpy数组
            return image
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def _compute_difference_map(self, template_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """计算模板图像和目标图像之间的差异图"""
        # 确保图像尺寸相同
        if template_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (template_img.shape[1], template_img.shape[0]))
        
        # 使用OpenCV的absdiff函数计算差异
        diff = cv2.absdiff(template_img, target_img)
        # 转换为灰度图
        diff_gray = np.max(diff, axis=2)
        
        return diff_gray
    
    def _apply_nlm_filter(self, image: np.ndarray, h: int = 4, 
                          template_window_size: int = 7, 
                          search_window_size: int = 21) -> np.ndarray:
        """应用非局部均值滤波进行降噪"""
        # 确保为单通道图像
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 应用NLM滤波
        filtered_img = cv2.fastNlMeansDenoising(
            image.astype(np.uint8), 
            None, 
            h=h, 
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        
        return filtered_img
    
    def _adaptive_range_enhancement(self, gray_image: np.ndarray, 
                                   percentile_low: int = 0, 
                                   percentile_high: int = 100) -> np.ndarray:
        """基于百分位数自适应地增强灰度图像对比度"""
        # 确保输入为灰度图
        if len(gray_image.shape) > 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        
        # 计算百分位数阈值
        min_val = np.percentile(gray_image, percentile_low)
        max_val = np.percentile(gray_image, percentile_high)
        
        # 线性拉伸
        alpha = 255.0 / (max_val - min_val) if max_val > min_val else 0
        stretched_img = np.clip((gray_image - min_val) * alpha, 0, 255).astype(np.uint8)
        
        # 直方图均衡化，进一步增强对比度
        equalized_img = cv2.equalizeHist(stretched_img)
        
        return equalized_img, stretched_img
    
    def _binary_threshold(self, image: np.ndarray, threshold: int = 20, 
                         threshold_type: str = 'adaptive') -> np.ndarray:
        """应用二值化阈值处理"""
        # 确保为单通道图像
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 应用二值化
        if threshold_type == 'otsu':
            # 使用Otsu方法自动计算阈值
            _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'simple':
            _, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        elif threshold_type == 'adaptive':
            # 自适应阈值
            binary_img = cv2.adaptiveThreshold(
                image, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                31, 
                5
            )
        else:
            raise ValueError(f"不支持的阈值类型: {threshold_type}")
            
        return binary_img
    
    def _morphological_operations(self, image: np.ndarray, 
                                 operation: str = 'close_open', 
                                 kernel_size: int = 3, 
                                 iterations_open: int = 1, 
                                 iterations_close: int = 1) -> np.ndarray:
        """对图像执行形态学操作（开运算和/或闭运算）"""
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
        
        # 执行指定的形态学操作
        if operation == 'open' or operation == 'open_close' or operation == 'close_open':
            if operation == 'close_open':
                # 先闭后开
                closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
                processed_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
            else:
                # 先执行开运算
                opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
                processed_image = opened_image
                
                if operation == 'open_close':
                    # 再执行闭运算
                    processed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
        
        elif operation == 'close':
            # 仅执行闭运算
            processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
        
        else:
            raise ValueError(f"不支持的形态学操作: {operation}")
        
        return processed_image
    
    def _connected_components_analysis(self, binary_image: np.ndarray, 
                                      connectivity: int = 8, 
                                      min_area: int = 100) -> Tuple:
        """对二值图像进行连通性分析，返回标记的实例图"""
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
            if area >= min_area:
                filtered_labels.append(i)
        
        return labels, stats, centroids, filtered_labels
    def _filter_by_rectangularity(self, binary_image: np.ndarray, 
                             labels: np.ndarray, 
                             stats: np.ndarray, 
                             centroids: np.ndarray, 
                             filtered_labels: List[int],
                             min_rectangularity: float = 0.55,
                             use_convex_hull: bool = False) -> Tuple:
        """
        根据区域的矩形度对连通区域进行过滤
        
        Args:
            binary_image (np.ndarray): 二值图像
            labels (np.ndarray): 标记图像，每个像素值为该连通区域的标签
            stats (np.ndarray): 每个连通区域的统计信息 [x, y, width, height, area]
            centroids (np.ndarray): 每个连通区域的中心点坐标
            filtered_labels (List[int]): 已初步过滤的连通区域标签列表
            min_rectangularity (float): 最小矩形度阈值，范围[0,1]，值越接近1表示区域越接近矩形
            use_convex_hull (bool): 是否使用凸包计算矩形度，对于有孔洞的区域更为准确
            
        Returns:
            Tuple: (rect_filtered_labels, rectangularity_values, mask)
                - rect_filtered_labels: 按矩形度过滤后的连通区域标签列表
                - rectangularity_values: 字典，键为标签，值为对应的矩形度
                - mask: 过滤后区域的掩码图像
        """
        # 过滤后的标签列表
        rect_filtered_labels = []
        # 记录每个区域的矩形度
        rectangularity_values = {}
        
        # 创建掩码图像
        mask = np.zeros_like(binary_image)
        
        # 对每个标签区域计算矩形度
        for label in filtered_labels:
            # 创建当前标签的掩码
            region_mask = (labels == label).astype(np.uint8) * 255
            
            # 找到区域的轮廓
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # 使用最大的轮廓
            contour = max(contours, key=cv2.contourArea)
            
            # 如果选择使用凸包，则计算凸包
            if use_convex_hull:
                contour = cv2.convexHull(contour)
            
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算轮廓面积
            contour_area = cv2.contourArea(contour)
            
            # 计算最小外接矩形面积
            rect_area = cv2.contourArea(box)
            
            # 计算矩形度：轮廓面积 / 最小外接矩形面积
            # 防止除以零
            if rect_area > 0:
                rectangularity = contour_area / rect_area
            else:
                rectangularity = 0
            
            # 记录矩形度
            rectangularity_values[label] = rectangularity
            
            # 根据矩形度过滤
            if rectangularity >= min_rectangularity:
                rect_filtered_labels.append(label)
                # 将区域添加到掩码
                mask = cv2.bitwise_or(mask, region_mask)
        
        # # 输出过滤结果
        # removed_count = len(filtered_labels) - len(rect_filtered_labels)
        # print(f"矩形度过滤: 从 {len(filtered_labels)} 个区域中移除了 {removed_count} 个不规则区域")
        
        return rect_filtered_labels, rectangularity_values, mask
    
    def _transform_and_place(
        self,
        binary_img: np.ndarray,
        canvas_size: tuple[int, int],
        M: np.ndarray,
        interpolation: int = cv2.INTER_NEAREST,
        border_value: int = 0
    ) -> np.ndarray:
        """
        将一张二值图像按给定仿射矩阵 M 变换后，贴到指定大小的画布上。

        Args:
            binary_img: np.ndarray of shape (H, W), dtype=uint8, values in {0,255}
            canvas_size: (canvas_h, canvas_w)，输出画布尺寸
            M:        2x3 仿射变换矩阵，例如 [[scale, 0, x_offset], [0, scale, y_offset]]
            interpolation: 插值方法，二值图推荐 INTER_NEAREST
            border_value:   画布边界和空白处的填充值，默认为 0

        Returns:
            placed: np.ndarray of shape (canvas_h, canvas_w), dtype=uint8
        """
        canvas_h, canvas_w = canvas_size

        # warpAffine 会把输入图像经 M 变换后放到一个新图像中，
        # 这里直接指定输出尺寸为画布大小
        placed = cv2.warpAffine(
            binary_img,
            M,
            (canvas_w, canvas_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        return placed
    
    def __call__(self, img: Union[torch.Tensor, Image.Image, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        将触觉图像处理为权重掩码
        
        参数:
            img: 输入图像，可以是PyTorch张量、PIL图像或NumPy数组
            
        返回:
            处理后的掩码图像，格式取决于to_tensor参数
        """
        # 1. 转换输入图像为numpy数组
        img_np = self._convert_to_numpy(img)
        
        # 2. 计算与模板的差异图
        diff_map = self._compute_difference_map(self.template_img, img_np)
        
        # 3. 应用NLM滤波
        filtered_diff = self._apply_nlm_filter(diff_map, h=4)
        
        # 4. 对比度增强
        _, enhanced_diff = self._adaptive_range_enhancement(filtered_diff, percentile_low=0, percentile_high=100)
        
        # 5. 二值化图像
        binary_diff = self._binary_threshold(enhanced_diff, threshold_type='adaptive')
        
        # 6. 形态学操作
        processed_diff = self._morphological_operations(
            binary_diff, 
            operation=self.morph_operation,
            kernel_size=self.morph_kernel_size
        )
        
        # 7. 图像反转（对于连通分析）
        binary_inverted = cv2.bitwise_not(processed_diff)
        
        # 8. 连通性分析
        labels, stats, centroids, filtered_labels = self._connected_components_analysis(
            binary_inverted,
            connectivity=4,
            min_area=self.min_area
        )
        
        # 9. 根据矩形度过滤连通区域
        rect_filtered_labels, rectangularity_values, mask = self._filter_by_rectangularity(
            binary_inverted,
            labels,
            stats,
            centroids,
            filtered_labels,
            min_rectangularity=self.min_rectangularity,
            use_convex_hull=False
        )
        if self.M is None:
            # 如果没有提供仿射矩阵，则使用单位矩阵
            self.M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        mask_transformed = self._transform_and_place(
            mask,
            canvas_size=self.canvas_size,
            M=self.M,  # 单位矩阵
            interpolation=cv2.INTER_NEAREST,
            border_value=0
        )
        # 10. 根据需要转换格式
        if self.to_tensor:
            # 归一化，使值范围在0-1之间
            if self.normalized:
                mask_transformed = mask_transformed / 255.0
                
            # 转换为tensor
            mask_tensor = torch.from_numpy(mask_transformed).float()
            # 添加通道维度
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0) # [H,W] -> [1,H,W]
                
            return mask_tensor
        else:
            return mask_transformed

class GlobalIlluminationAlignment(nn.Module):
    """
    执行全局光照对齐
    
    通过计算整个图像的统计特性（平均值和可选的标准差）来进行光照对齐，
    应用线性变换 y = ax + b 使源图像在统计上与目标图像匹配
    
    参数:
        eps (float): 防止除零的小数值
        match_variance (bool): 是否匹配方差。如果为True，同时调整对比度
        per_channel (bool): 是否为每个通道单独执行对齐
    """
    def __init__(self, eps=1e-6, match_variance=False, per_channel=True):
        super().__init__()
        self.eps = eps
        self.match_variance = match_variance
        self.per_channel = per_channel

    def forward(self, img, template):
        """
        将输入图像的光照调整为与模板图像一致
        
        参数:
            img (torch.Tensor): 输入图像张量 (B, C, H, W)，值范围[0, 1]或[0, 255]
            template (torch.Tensor): 模板图像张量 (B, C, H, W)，值范围[0, 1]或[0, 255]
            
        返回:
            torch.Tensor: 光照对齐后的图像张量 (B, C, H, W)，与输入值范围相同
        """
        # 确保输入是PyTorch张量
        if not isinstance(img, torch.Tensor) or not isinstance(template, torch.Tensor):
            raise TypeError("输入必须是PyTorch张量")
        
        # 检查输入尺寸
        if img.dim() != 4 or template.dim() != 4:
            raise ValueError("输入张量必须是4维 (B, C, H, W)")
        
        # 如果形状不同，调整img大小以匹配template
        if img.shape[-2:] != template.shape[-2:]:
            img = F.interpolate(img, size=template.shape[-2:], mode='bilinear', align_corners=False)
        
        # 确定图像是否已归一化
        is_normalized = img.max() <= 1.1
        
        # 如果已归一化，转换为0-255范围
        if is_normalized:
            img_proc = img * 255.0
            template_proc = template * 255.0
        else:
            img_proc = img.clone()
            template_proc = template.clone()
        
        # 将输入转为浮点型
        img_proc = img_proc.float()
        template_proc = template_proc.float()
        
        # 计算全局统计量
        if self.per_channel:
            # 每个通道单独计算统计量
            # 形状: [B, C]
            dim = [2, 3]  # 在H和W维度上计算
        else:
            # 所有通道一起计算
            # 形状: [B, 1]
            dim = [1, 2, 3]  # 在C、H和W维度上计算
        
        # 计算均值
        img_mean = img_proc.mean(dim=dim, keepdim=True)  # [B, C, 1, 1] 或 [B, 1, 1, 1]
        template_mean = template_proc.mean(dim=dim, keepdim=True)
        
        if self.match_variance:
            # 计算标准差
            img_std = torch.std(img_proc, dim=dim, keepdim=True) + self.eps
            template_std = torch.std(template_proc, dim=dim, keepdim=True) + self.eps
            
            # 计算缩放因子a和偏移量b
            a = template_std / img_std
            b = template_mean - a * img_mean
            
            # 防止极端缩放值
            a = torch.clamp(a, 0.1, 10.0)
        else:
            # 仅匹配均值，保持方差不变
            a = torch.ones_like(img_mean)
            b = template_mean - img_mean
        
        # 应用变换: aligned = a * x + b
        aligned_img = a * img_proc + b
        
        # 确保值在有效范围内
        aligned_img = torch.clamp(aligned_img, 0, 255)
        
        # 如果输入是归一化的，转换回归一化格式
        if is_normalized:
            aligned_img = aligned_img / 255.0
        
        return aligned_img
    
    def align_image(self, img, target_mean, target_std=None):
        """
        将输入图像的光照调整为与目标均值和标准差一致
        
        参数:
            img (torch.Tensor): 输入图像张量 (B, C, H, W)，值范围[0, 1]或[0, 255]
            target_mean (torch.Tensor): 目标均值 (B, C, 1, 1) 或 (B, 1, 1, 1)
            target_std (torch.Tensor, optional): 目标标准差 (B, C, 1, 1) 或 (B, 1, 1, 1)
            
        返回:
            torch.Tensor: 光照对齐后的图像张量 (B, C, H, W)，与输入值范围相同
        """
        # 确保输入是PyTorch张量
        if not isinstance(img, torch.Tensor) or not isinstance(target_mean, torch.Tensor):
            raise TypeError("输入必须是PyTorch张量")
        
        # 确定图像是否已归一化
        is_normalized = img.max() <= 1.1
        
        # 如果已归一化，转换为0-255范围
        if is_normalized:
            img_proc = img * 255.0
            target_mean_proc = target_mean * 255.0
            if target_std is not None:
                target_std_proc = target_std * 255.0
        else:
            img_proc = img.clone()
            target_mean_proc = target_mean.clone()
            if target_std is not None:
                target_std_proc = target_std.clone()
        
        # 将输入转为浮点型
        img_proc = img_proc.float()
        
        # 计算统计量
        if self.per_channel:
            dim = [2, 3]  # 在H和W维度上计算
        else:
            dim = [1, 2, 3]  # 在C、H和W维度上计算
            img_proc = img_proc.mean(dim=1, keepdim=True)
        
        # 计算均值
        img_mean = img_proc.mean(dim=dim, keepdim=True)
        
        if self.match_variance and target_std is not None:
            # 计算标准差
            img_std = torch.std(img_proc, dim=dim, keepdim=True) + self.eps
            
            # 计算缩放因子a和偏移量b
            a = target_std_proc / img_std
            b = target_mean_proc - a * img_mean
            
            # 防止极端缩放值
            a = torch.clamp(a, 0.1, 10.0)
        else:
            # 仅匹配均值，保持方差不变
            a = torch.ones_like(img_mean)
            b = target_mean_proc - img_mean
        
        # 应用变换: aligned = a * x + b
        aligned_img = a * img_proc + b
        
        # 确保值在有效范围内
        aligned_img = torch.clamp(aligned_img, 0, 255)
        
        # 如果输入是归一化的，转换回归一化格式
        if is_normalized:
            aligned_img = aligned_img / 255.0
        
        return aligned_img
    
class PatchBasedIlluminationAlignment(nn.Module):
    """
    Performs illumination alignment patch-by-patch.

    Divides images into non-overlapping patches (window_size).
    Calculates statistics from a larger surrounding region (kernel_size).
    Applies linear transformation (ax+b) to the window_size patch
    based on kernel_size statistics to match the template.
    """
    def __init__(self, window_size=16, kernel_size=32, eps=1e-6, keep_variance=False):
        """
        Args:
            window_size (int): Size of the non-overlapping patches for applying the transform.
                               Must divide image height and width. Must be <= kernel_size.
            kernel_size (int): Size of the surrounding region for calculating statistics.
                               Must be >= window_size. Should ideally be odd for clear centering,
                               but code handles even sizes too.
            eps (float): Small value to prevent division by zero.
            keep_variance (bool): 如果为True，保持原图像的方差不变，只对均值进行调整。
        """
        super().__init__()

        if window_size > kernel_size:
            raise ValueError("kernel_size must be greater than or equal to window_size")

        self.window_size = window_size
        self.kernel_size = kernel_size
        self.eps = eps
        self.keep_variance = keep_variance

        # Padding needed to extract kernel_size patches centered on window_size patches
        self.pad_k = (self.kernel_size - self.window_size)// 2
        # Stride for unfold is the window_size for non-overlapping patches
        self.stride = self.window_size

    def forward(self, img, template):
        """
        Args:
            img (torch.Tensor): Input image tensor (B, C, H, W). Values [0, 1] or [0, 255].
            template (torch.Tensor): Template image tensor (B, C, H, W). Values [0, 1] or [0, 255].

        Returns:
            torch.Tensor: Aligned image tensor (B, C, H, W) with the same value range as input.
        """
        # --- Input Validation and Preparation ---
        if not isinstance(img, torch.Tensor) or not isinstance(template, torch.Tensor):
            raise TypeError("Inputs must be PyTorch tensors.")

        B, C, H, W = img.shape
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(f"window_size ({self.window_size}) must divide image height ({H}) and width ({W})")

        if img.shape != template.shape:
            img = F.interpolate(img, size=template.shape[-2:], mode='bilinear', align_corners=False)

        is_normalized = img.max() <= 1.1
        if is_normalized:
            img_proc = img * 255.0
            template_proc = template * 255.0
        else:
            img_proc = img.clone() # Use clone to avoid modifying original
            template_proc = template.clone()

        img_proc = img_proc.float()
        template_proc = template_proc.float()
        device = img.device

        # --- Calculate Statistics from Kernel Regions ---
        # Pad images for kernel extraction
        img_padded_k = F.pad(img_proc, (self.pad_k, self.pad_k, self.pad_k, self.pad_k), mode='reflect')
        template_padded_k = F.pad(template_proc, (self.pad_k, self.pad_k, self.pad_k, self.pad_k), mode='reflect')

        # Unfold to get kernel_size patches, striding by window_size
        unfold_k = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=0)
        img_patches_k = unfold_k(img_padded_k)          # (B, C*k*k, L)
        template_patches_k = unfold_k(template_padded_k) # (B, C*k*k, L)

        # Reshape for stats calculation
        k_sq = self.kernel_size * self.kernel_size
        img_patches_k = img_patches_k.view(B, C, k_sq, -1)          # (B, C, k*k, L)
        template_patches_k = template_patches_k.view(B, C, k_sq, -1) # (B, C, k*k, L)
        L = img_patches_k.shape[-1] # Number of patches

        # Calculate mean and std over kernel patches
        img_mean_k = img_patches_k.mean(dim=2)              # (B, C, L)
        template_mean_k = template_patches_k.mean(dim=2)    # (B, C, L)

        # --- Calculate Transformation Parameters ---
        if self.keep_variance:
            # 如果设置为保持方差，则a=1，只对均值进行调整
            a = torch.ones_like(img_mean_k)
            b = template_mean_k - img_mean_k
        else:
            # 原始计算方式，同时调整均值和方差
            img_std_k = img_patches_k.std(dim=2, unbiased=False) + self.eps # (B, C, L) # Use population std
            template_std_k = template_patches_k.std(dim=2, unbiased=False) + self.eps # (B, C, L)
            a = template_std_k / img_std_k
            b = template_mean_k - a * img_mean_k
            
        # 排除异常值
        a = torch.clamp(a, 0.01, 100.0)
        b = torch.clamp(b, -255.0, 255.0)

        # --- Extract Window Patches and Apply Transform ---
        # Unfold the original image to get non-overlapping window_size patches
        unfold_w = nn.Unfold(kernel_size=self.window_size, stride=self.stride, padding=0)
        img_patches_w = unfold_w(img_proc) # (B, C*w*w, L)

        # Reshape for applying transform
        w_sq = self.window_size * self.window_size
        img_patches_w = img_patches_w.view(B, C, w_sq, L) # (B, C, w*w, L)

        # Expand a and b to match patch dimensions for broadcasting
        # a: (B, C, L) -> (B, C, 1, L)
        # b: (B, C, L) -> (B, C, 1, L)
        a_expanded = a.unsqueeze(2)
        b_expanded = b.unsqueeze(2)

        # Apply transformation: aligned = a * original + b
        aligned_patches_w = a_expanded * img_patches_w + b_expanded # (B, C, w*w, L)
    
        # --- Reconstruct Image ---
        # Reshape back for folding
        aligned_patches_w = aligned_patches_w.view(B, C * w_sq, L) # (B, C*w*w, L)

        # Use Fold to reconstruct the image from non-overlapping patches
        fold_w = nn.Fold(output_size=(H, W), kernel_size=self.window_size, stride=self.stride, padding=0)
        aligned_img = fold_w(aligned_patches_w)

        # Clamp final output and convert back to original range if needed
        aligned_img = torch.clamp(aligned_img, 0, 255)

        if is_normalized:
            aligned_img = aligned_img / 255.0
            # aligned_img = torch.clamp(aligned_img, 0, 1.0) # Ensure range safety

        return aligned_img

# Helper functions (can be in the same file or imported)
def np_to_torch(img_np, device='cuda'):
    """Converts NumPy image (H, W, C) or (B, H, W, C) to PyTorch tensor (B, C, H, W)."""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, using CPU.")
    if img_np.ndim == 3:
        img_np = np.expand_dims(img_np, 0) # Add batch dim
    img_tensor = torch.from_numpy(np.transpose(img_np, (0, 3, 1, 2))).float()
    return img_tensor.to(device)

def torch_to_np(tensor):
    """Converts PyTorch tensor (B, C, H, W) to NumPy image (B, H, W, C) or (H, W, C)."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (0, 2, 3, 1))
    if img_np.shape[0] == 1:
        img_np = img_np[0] # Remove batch dim if size 1
    # Handle potential float output if input was normalized
    if img_np.max() <= 1.1:
         img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    else:
         img_np = img_np.clip(0, 255).astype(np.uint8)
    return img_np

class TerraceMapGenerator:
    """
    将掩码图像转换为"梯田图"的变换类 (兼容 torchvision.transforms)
    
    处理流程:
    1. 提取掩码边缘并检测内外轮廓
    2. 拟合旋转矩形
    3. 生成多层次梯田图 (由内到外亮度递减)
    4. 在边缘处增强亮度
    
    参数:
        intensity_scaling: 强度缩放倍数列表 (从内到外的亮度比例)
        edge_enhancement: 边缘增强比例
        expansion_ratio: 矩形扩大系数
        output_tensor: 是否返回tensor而不是PIL图像
    """
    def __init__(self, 
                 intensity_scaling: List[float] = [0.1, 0.6, 0.8, 1.0],
                 edge_enhancement: float = 2.0,
                 expansion_size: Dict[str, List[float]] = None,
                 sample_size: int = 256,
                 debug: bool = False):
        """
        初始化梯田图生成器
        
        参数:
            intensity_scaling: 强度缩放倍数列表 (从内到外的亮度比例)
            edge_enhancement: 边缘增强比例
            expansion_ratio: 矩形扩大尺寸
            aspect_ratio: Target aspect ratio for rectangle (width/height), None for preserving original ratio
            output_tensor: 是否返回tensor而不是PIL图像
        """
        self.intensity_scaling = intensity_scaling
        self.edge_enhancement = edge_enhancement
        self.expansion_size = expansion_size 
        self.sample_size = sample_size
        self.debug = debug
        
        # 初始化内部状态
        self.original_mask = None
        self.edge_mask = None
        self.terrace_map = None
        self.rect_params = None
        
    def __call__(self, img, serial = '3524P'):
        """
        处理掩码图像(兼容torchvision.transforms.Compose)
        
        参数:
            img: 输入掩码图像，PIL格式
            
        返回:
            梯田图，格式取决于初始化参数
        """
        # 生成梯田图
        terrace_map, outer_contour = self.generate_terrace_map(img, serial = serial)
        # print(outer_contour)
        return self.terrace_map, outer_contour
        
    def _convert_to_numpy(self, mask_img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        将掩码图像转换为numpy数组
        
        参数:
            mask_img: 输入掩码，可以是文件路径、PIL图像或numpy数组
            
        返回:
            二值掩码，numpy数组格式
        """
        if isinstance(mask_img, str):
            # 从文件路径加载
            mask = np.array(Image.open(mask_img).convert('L'))
           
        elif isinstance(mask_img, Image.Image):
            # 转换PIL图像为灰度
            mask = np.array(mask_img.convert('L'))
            
        elif isinstance(mask_img, np.ndarray):
            # 将numpy数组转换为灰度(如果需要)
            if len(mask_img.shape) == 3:
                # 转换RGB/RGBA为灰度
                mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            else:
                mask = mask_img
        else:
            raise ValueError("不支持的掩码类型。必须是文件路径、PIL图像或numpy数组")
        
        # 转换为二值掩码
        if np.max(mask) > 1:
            mask = (mask > 127).astype(np.uint8) * 255
            
        self.original_mask = mask
        return mask
    
    def _detect_contours(self, mask: np.ndarray) -> Tuple[List, List]:
        """
        检测掩码的内外轮廓
        
        参数:
            mask: 输入二值掩码
            
        返回:
            (内轮廓列表, 外轮廓列表)
        """
        # 确保掩码是二值的
        binary_mask = (mask > 127).astype(np.uint8)
        
        # 查找所有轮廓
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return [], []
            
        # 根据层次结构分离内外轮廓
        outer_contours = []
        inner_contours = []
        
        # 如果层次结构有效，使用它来区分内外轮廓
        if hierarchy is not None and hierarchy.shape[1] > 0:
            for i, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                # h[3] > -1 表示这个轮廓有父轮廓，因此是内轮廓
                if h[3] > -1:
                    inner_contours.append(contour)
                else:
                    outer_contours.append(contour)
        else:
            # 如果没有层次结构，则所有轮廓都视为外轮廓
            outer_contours = contours
        
        # 按面积排序轮廓（从大到小）
        outer_contours = sorted(outer_contours, key=cv2.contourArea, reverse=True)
        inner_contours = sorted(inner_contours, key=cv2.contourArea, reverse=True)
        
        if self.debug:
            # 将轮廓绘制在原图上，按内外不同颜色可视化
            # 创建彩色图像用于显示轮廓
            contour_vis = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
            
            # 绘制外轮廓(绿色)
            cv2.drawContours(contour_vis, outer_contours, -1, (0, 255, 0), 2)
            
            # 绘制内轮廓(红色)
            cv2.drawContours(contour_vis, inner_contours, -1, (0, 0, 255), 2)
            
            # 保存可视化结果
            self.contour_vis = contour_vis
        
        return inner_contours, outer_contours
    
    def _fit_rotated_rectangle(self, contour: np.ndarray) -> Tuple:
        """
        将旋转矩形拟合到轮廓
        
        参数:
            contour: 输入轮廓
            
        返回:
            矩形参数 (中心点, (宽度, 高度), 角度)
        """
        # 拟合旋转矩形
        rect = cv2.minAreaRect(contour)
        self.rect_params = rect
        return rect
    
    def _expand_rectangle(self, rect: Tuple, expansion_wh: List[float]) -> Tuple:
        """
        Expand rectangle with optional aspect ratio control
        
        Parameters:
            rect: Rectangle parameters
            expansion: Expansion factor
            
        Returns:
            Expanded rectangle parameters
        """
        # Get rectangle parameters
        (cx, cy), (width, height), angle = rect
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        # Calculate new width and height
        new_width, new_height = expansion_wh
        
        # Create new rectangle
        expanded_rect = ((cx, cy), (new_width-10, new_height-20), angle)
        
        return expanded_rect
    
    def _create_filled_mask(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        创建轮廓填充的掩码
        
        参数:
            contour: 轮廓
            shape: 输出掩码形状
            
        返回:
            填充掩码
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)  # -1表示填充
        return mask
    
    def _create_rectangle_mask(self, rect: Tuple, shape: Tuple[int, int]) -> np.ndarray:
        """
        创建矩形填充的掩码
        
        参数:
            rect: 矩形参数
            shape: 输出掩码形状
            
        返回:
            填充掩码
        """
        mask = np.zeros(shape, dtype=np.uint8)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        cv2.fillPoly(mask, [box_points], 255)
        return mask
    
    def _extract_edges(self, mask: np.ndarray, thickness: int = 1) -> np.ndarray:
        """
        提取掩码的边缘
        
        参数:
            mask: 输入掩码
            thickness: 边缘粗细
            
        返回:
            边缘掩码
        """
        # 使用形态学梯度提取边缘
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        
        # 如果需要更粗的边缘，可以使用膨胀
        if thickness > 1:
            edge = cv2.dilate(edge, kernel, iterations=thickness-1)
            
        return edge
    def resample_contour(self, contour, num_samples):
        """
        对 OpenCV 轮廓做等距重采样。

        参数
        ----
        contour : np.ndarray of shape (N,1,2) 或 (N,2)
            单条闭合轮廓点集。
        num_samples : int
            需要采样的总点数。

        返回
        ----
        samples : np.ndarray of shape (num_samples, 2)
            在轮廓上等距分布的采样点坐标。
        """
        # 1. 展平到 (N,2) 并转为 float
        pts = contour.reshape(-1, 2).astype(np.float64)
        N = pts.shape[0]

        # 2. 计算每条边的向量和长度，首尾连通
        #    diffs[i] = pts[(i+1)%N] - pts[i]
        diffs = np.vstack([pts[1:] - pts[:-1], pts[0] - pts[-1]])
        seg_len = np.hypot(diffs[:,0], diffs[:,1])  # 每条边的长度，shape=(N,)

        # 3. 累积长度
        cumlen = np.concatenate([[0], np.cumsum(seg_len)])
        perim = cumlen[-1]

        # 4. 等距采样的距离位置（从 0 到 perim，不包含 perim 自身）
        sample_d = np.linspace(0, perim, num_samples, endpoint=False)

        # 5. 对每个采样距离 d：
        #    - 用 searchsorted 找到它位于哪条边 cumlen[i] <= d < cumlen[i+1]
        #    - 计算该边上插值比例 t = (d - cumlen[i]) / seg_len[i]
        #    - 插值 pts[i] + t * diffs[i]
        idx = np.searchsorted(cumlen, sample_d, side='right') - 1
        t = (sample_d - cumlen[idx]) / seg_len[idx]

        # 6. 生成采样点
        p0 = pts[idx]
        d = diffs[idx]
        samples = p0 + (d.T * t).T  # 广播插值

        return samples
    
    def generate_terrace_map(self, mask_img: Union[str, Image.Image, np.ndarray], serial: str = '3524P') -> np.ndarray:
        """
        从掩码生成梯田图
        
        参数:
            mask_img: 输入掩码
            serial: 序列号，用于调试或标识
            
        返回:
            梯田图 (灰度图)
        """
        # 转换为numpy数组
        mask = self._convert_to_numpy(mask_img)
        
        # 获取图像形状
        height, width = mask.shape
        shape = (height, width)
        
        # 检测轮廓
        inner_contours, outer_contours = self._detect_contours(mask)

        # 初始化梯田图 (全黑背景)
        terrace_map = np.zeros(shape, dtype=np.float32)
        
        # 创建梯田级别掩码
        level_masks = []
        
        # 第4级 (最外层): 整张图
        full_mask = np.ones(shape, dtype=np.uint8) * 255
        level_masks.append(full_mask)
        
        # 第3级: 拟合并扩展矩形
        if outer_contours:
            # 拟合矩形到最大外轮廓
            rect = self._fit_rotated_rectangle(outer_contours[0])
            # 扩展矩形
            expansion_wh = self.expansion_size.get(serial, [300.0, 150.0])
            expanded_rect = self._expand_rectangle(rect, expansion_wh)
            # 创建矩形掩码
            rect_mask = self._create_rectangle_mask(expanded_rect, shape)
            level_masks.append(rect_mask)
        else:
            # 如果没有外轮廓，使用整个图像
            level_masks.append(full_mask)
        
        # 第2级: 外轮廓填充
        if outer_contours:
            outer_mask = self._create_filled_mask(outer_contours[0], shape)
            level_masks.append(outer_mask)
        else:
            # 如果没有外轮廓，使用已有的最小掩码
            level_masks.append(level_masks[-1] if level_masks else full_mask)
            
        # 第1级 (最内层): 内轮廓填充
        if inner_contours:
            inner_mask = self._create_filled_mask(inner_contours[0], shape)
            level_masks.append(inner_mask)
        else:
            # 如果没有内轮廓，但有外轮廓
            if outer_contours:
                # 尝试使用腐蚀来模拟内轮廓
                kernel = np.ones((15, 15), np.uint8)
                eroded_mask = cv2.erode(level_masks[-1], kernel, iterations=1)
                if np.sum(eroded_mask) > 0:  # 确保腐蚀后的掩码不为空
                    level_masks.append(eroded_mask)
                else:
                    level_masks.append(level_masks[-1])
            else:
                # 如果既没有内轮廓也没有外轮廓，使用已有的最小掩码
                level_masks.append(level_masks[-1] if level_masks else full_mask)
        
        # 确保我们有足够的强度缩放值
        while len(self.intensity_scaling) < len(level_masks):
            self.intensity_scaling.append(0.1)  # 默认添加较低的强度
        intensity_values = self.intensity_scaling[:len(level_masks)]
        
        # 初始化梯田图（全黑背景）
        terrace_map = np.zeros(shape, dtype=np.float32)
        # 从外到内直接绘制梯田图（后绘制的会覆盖先绘制的）
        for i, (mask, intensity) in enumerate(zip(level_masks, intensity_values)):
            # 将当前区域的强度应用到梯田图
            terrace_map[mask > 0] = intensity * 255
        
        # 检测梯田图的边缘
        all_edges = self._extract_edges(terrace_map, thickness=2)

        # 在边缘处增强亮度
        edge_indices = all_edges > 0
        terrace_map[edge_indices] = np.minimum(terrace_map[edge_indices] * self.edge_enhancement, 255)
        
        # 转换为8位无符号整数
        self.terrace_map = terrace_map.astype(np.uint8)

        # *: 可视化debug，可以去除
        if self.debug:
            level_masks.append(self.contour_vis)  # 添加原始掩码用于调试
            level_masks.append(all_edges)  # 添加当前掩码用于调试
            # 测试可视化
            from utils.VisualizeParam import visualize_images
            image_list = level_masks
            title_list = ["full_mask", "rect_mask", "outer_mask", "inner_mask", "contour_mask", "all_edges"]
            visualize_images(image_list, title_list, save_path=None, display=True)
        
        sample_contour = self.resample_contour(outer_contours[0], self.sample_size)
        return self.terrace_map, torch.from_numpy(sample_contour)

class VisionTerraceMapGenerator:
    """
    将掩码图像转换为"梯田图"的变换类 (兼容 torchvision.transforms)
    
    处理流程:
    1. 提取掩码边缘并检测内外轮廓
    2. 拟合旋转矩形
    3. 生成多层次梯田图 (由内到外亮度递减)
    4. 在边缘处增强亮度
    
    参数:
        intensity_scaling: 强度缩放倍数列表 (从内到外的亮度比例)
        edge_enhancement: 边缘增强比例
        expansion_ratio: 矩形扩大系数
        output_tensor: 是否返回tensor而不是PIL图像
    """
    def __init__(self, 
                 intensity_scaling: List[float] = [0.1, 0.6, 0.8, 1.0],
                 edge_enhancement: float = 2.0,
                 expansion_size: Dict[str, List[float]] = None,
                 sample_size: int = 256,
                 yolo_model_path: str = None,
                 return_mask: bool = False,
                 debug: bool = False):
        
        self.yolo_model = YOLO(model=yolo_model_path) if yolo_model_path else None
        self.return_mask = return_mask

        self.transform = TerraceMapGenerator(
            intensity_scaling=intensity_scaling,
            edge_enhancement=edge_enhancement,
            expansion_size=expansion_size,
            sample_size=sample_size,
            debug=debug
        )
    
    def __call__(self, img, serial = '3524P'):
        """
        处理掩码图像(兼容torchvision.transforms.Compose)
        
        参数:
            img: 输入掩码图像，PIL格式
            
        返回:
            梯田图，格式取决于初始化参数
        """
        mask_img = self._yolo_inference(img)
        if self.return_mask:
            return mask_img
        
        # 生成梯田图
        terrace_map, outer_contour = self.transform(img, serial = serial)
        return terrace_map, outer_contour
    
    def _yolo_inference(self, img):
        """
        使用YOLO模型进行推理
        
        参数:
            img: 输入图像，PIL格式
            
        返回:
            检测结果
        """
        # 使用YOLO模型进行推理，返回一张mask
        if self.yolo_model is not None:
            results = self.yolo_model.predict(img, conf=0.25, iou=0.45, device='cuda:0')
            # 处理结果，返回mask
            mask_tensor = results[0].masks.data

            # 如果有多个掩码，合并它们
            if mask_tensor.shape[0] > 1:
                mask_combined = torch.max(mask_tensor, dim=0)[0]
            else:
                mask_combined = mask_tensor.squeeze(0)
                
            # 将掩码转换为NumPy数组
            mask_np = mask_combined.cpu().numpy()
            
            # 确保掩码值在0-255范围内
            mask_np = (mask_np * 255).astype(np.uint8)
            
            # 转换为PIL图像
            from PIL import Image
            mask_pil = Image.fromarray(mask_np)
            
            return mask_pil
        else:
            # 如果没有检测到掩码，返回空白掩码
            from PIL import Image
            blank_mask = np.zeros(img.size[::-1], dtype=np.uint8)
            mask_pil = Image.fromarray(blank_mask)
            return mask_pil

class SSIM(nn.Module):
    """
    使用PyTorch实现的结构相似性指数(SSIM)，与NumPy/scipy版本功能一致
    
    支持将彩色图像自动转为灰度图，并计算SSIM
    """
    def __init__(self, win_size=11, sigma=1.5, k1=0.01, k2=0.03, 
                 gaussian_weights=True, use_sample_covariance=True):
        """
        初始化SSIM计算模块
        
        Args:
            win_size (int): 窗口大小，必须是奇数
            sigma (float): 高斯权重的标准差
            k1 (float): SSIM计算中的常数1
            k2 (float): SSIM计算中的常数2
            gaussian_weights (bool): 是否使用高斯加权窗口
            use_sample_covariance (bool): 如果为True，使用N-1归一化协方差
        """
        super(SSIM, self).__init__()
        
        # 确保窗口大小是奇数
        if win_size % 2 == 0:
            win_size = win_size + 1
            
        self.win_size = win_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.gaussian_weights = gaussian_weights
        self.use_sample_covariance = use_sample_covariance
        
        # 创建窗口
        self.register_buffer('window', self._create_window())
        
        # RGB转灰度的权重（标准化）
        self.register_buffer('rgb_weights', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
    def _create_window(self):
        """
        创建用于计算SSIM的窗口函数，与scipy版本匹配
        """
        if self.gaussian_weights:
            # 使用与scipy相同的参数
            truncate = 3.5
            radius = int(truncate * self.sigma + 0.5)
            
            # 创建坐标网格
            x, y = torch.meshgrid(
                torch.arange(-radius, radius + 1, dtype=torch.float32),
                torch.arange(-radius, radius + 1, dtype=torch.float32),
                indexing='ij'
            )
            
            # 计算高斯核
            kernel = torch.exp(-((x.pow(2) + y.pow(2)) / (2 * self.sigma**2)))
            kernel = kernel / kernel.sum()
            
            # 调整为所需的窗口大小
            if kernel.shape[0] < self.win_size:
                # 如果核比窗口小，需要填充
                padding = (self.win_size - kernel.shape[0]) // 2
                kernel = F.pad(kernel, (padding, padding, padding, padding), mode='constant', value=0)
            
            # 如果核比窗口大，需要截取中心部分
            if kernel.shape[0] > self.win_size:
                offset = (kernel.shape[0] - self.win_size) // 2
                kernel = kernel[offset:offset+self.win_size, offset:offset+self.win_size]
        else:
            # 均匀窗口
            kernel = torch.ones((self.win_size, self.win_size), dtype=torch.float32)
            kernel = kernel / kernel.sum()
            
        # 将窗口整形为卷积核格式 (1, 1, H, W)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _to_grayscale(self, x):
        """
        将RGB图像转换为灰度图像
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            灰度图像 (B, 1, H, W)
        """
        if x.size(1) == 3:  # RGB图像
            return (x * self.rgb_weights).sum(dim=1, keepdim=True)
        elif x.size(1) == 1:  # 已经是灰度图
            return x
        else:  # 对于其他通道数，取平均值
            return x.mean(dim=1, keepdim=True)
    
    def forward(self, img1, img2, data_range=None, full=False, wo_light=False, loss=False):
        """
        计算SSIM
        
        Args:
            img1: 第一张图像 (B, C, H, W)
            img2: 第二张图像 (B, C, H, W)
            data_range: 数据范围，如果为None则从数据类型推断
            full: 如果为True，返回完整的SSIM图和均值，否则只返回均值
            wo_light: 如果为True，不考虑亮度项（仅结构和对比度）
            
        Returns:
            如果full=False，返回SSIM得分
            如果full=True，返回(SSIM得分, SSIM图)
        """
        # 检查输入
        if img1.shape != img2.shape:
            raise ValueError("输入图像必须具有相同的尺寸")
        
        # 转换为灰度图
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)
        
        # 确定数据范围
        if data_range is None:
            if gray1.max() <= 1.1:  # 归一化数据[0,1]
                data_range = 1.0
            else:  # 假设[0,255]范围
                data_range = 255.0
        
        # 常数
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        
        # 填充图像以进行卷积（与scipy行为匹配）
        pad = (self.win_size - 1) // 2
        gray1_padded = F.pad(gray1, (pad, pad, pad, pad), mode='reflect')
        gray2_padded = F.pad(gray2, (pad, pad, pad, pad), mode='reflect')
        
        # 样本协方差标准化
        if self.use_sample_covariance:
            cov_norm = self.win_size * self.win_size / (self.win_size * self.win_size - 1)
        else:
            cov_norm = 1.0
            
        # 使用卷积计算均值
        window = self.window.to(gray1.device)
        ux = F.conv2d(gray1_padded, window, padding=0)
        uy = F.conv2d(gray2_padded, window, padding=0)
        
        # 计算平方和协方差
        uxx = F.conv2d(gray1_padded * gray1_padded, window, padding=0)
        uyy = F.conv2d(gray2_padded * gray2_padded, window, padding=0)
        uxy = F.conv2d(gray1_padded * gray2_padded, window, padding=0)
        
        # 计算方差和协方差
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        
        # SSIM公式计算
        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux.pow(2) + uy.pow(2) + C1
        B2 = vx + vy + C2
        
        if wo_light:
            # 不考虑亮度项，只计算结构和对比度
            S = A2 / B2
        else:
            # 完整SSIM
            D = B1 * B2
            S = (A1 * A2) / D
        
        # 如需裁剪边缘，与scipy行为匹配
        # if pad > 0:
        #     S_valid = S[:, :, pad:-pad, pad:-pad]
        # else:
        #     S_valid = S
        
        # 计算平均SSIM
        mssim = S.mean()
        # print(S.shape)
        if loss:
            if full:
                return 1 - mssim, 1 - S
            else:
                return 1 - mssim
        else:
            if full:
                return mssim, S
            else:
                return mssim


def ssim_pytorch(img1, img2, win_size=11, sigma=1.5, k1=0.01, k2=0.03,
                data_range=None, gaussian_weights=True, use_sample_covariance=True,
                full=False, wo_light=False):
    """
    使用PyTorch计算SSIM的便捷函数，与NumPy/scipy版本功能一致
    
    Args:
        img1, img2: 输入图像张量，形状为(B,C,H,W)
        win_size: 窗口大小，必须是奇数
        sigma: 高斯权重的标准差
        k1, k2: SSIM计算中的常数
        data_range: 数据范围，默认根据数据类型推断
        gaussian_weights: 是否使用高斯加权窗口
        use_sample_covariance: 如果为True，使用N-1归一化协方差
        full: 如果为True，返回完整的SSIM图像和均值
        wo_light: 如果为True，不考虑亮度项(仅考虑结构和对比度)
        
    Returns:
        如果full=False，返回SSIM得分(标量)
        如果full=True，返回(SSIM得分, SSIM图像)
    """
    # 处理单个图像输入（添加批次维度）
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # 创建SSIM模块
    ssim_module = SSIM(
        win_size=win_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        gaussian_weights=gaussian_weights,
        use_sample_covariance=use_sample_covariance
    ).to(img1.device)
    
    # 计算SSIM
    with torch.no_grad():
        result = ssim_module(img1, img2, data_range=data_range, full=full, wo_light=wo_light)
    
    return result


def ssim_np2torch(img1, img2, win_size=11, sigma=1.5, k1=0.01, k2=0.03,
                data_range=None, gaussian_weights=True, use_sample_covariance=True,
                full=False, wo_light=False):
    """
    使用PyTorch计算SSIM的便捷函数，与NumPy/scipy版本功能一致
    
    Args:
        img1, img2: 输入图像，可以是NumPy数组(H,W)或(H,W,C)或PyTorch张量(C,H,W)或(B,C,H,W)
        win_size: 窗口大小，必须是奇数
        sigma: 高斯权重的标准差
        k1, k2: SSIM计算中的常数
        data_range: 数据范围，默认根据数据类型推断
        gaussian_weights: 是否使用高斯加权窗口
        use_sample_covariance: 如果为True，使用N-1归一化协方差
        full: 如果为True，返回完整的SSIM图像和均值
        wo_light: 如果为True，不考虑亮度项(仅考虑结构和对比度)
        
    Returns:
        如果full=False，返回SSIM得分(标量)
        如果full=True，返回(SSIM得分, SSIM图像)
    """
    # 处理不同格式的输入
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_array = isinstance(img1, np.ndarray)
    # 处理NumPy数组输入
    if isinstance(img1, np.ndarray):

        # 将NumPy数组转换为PyTorch张量
        is_gray1 = len(img1.shape) == 2
        
        if is_gray1:
            # 将灰度图添加通道维度
            img1_tensor = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
        else:
            # 将HWC格式转换为BCHW格式
            img1_tensor = torch.from_numpy(np.transpose(img1, (2, 0, 1))).float().unsqueeze(0)
            
        img1 = img1_tensor.to(device)
    
    if isinstance(img2, np.ndarray):
        # 将NumPy数组转换为PyTorch张量
        is_gray2 = len(img2.shape) == 2
        
        if is_gray2:
            # 将灰度图添加通道维度
            img2_tensor = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
        else:
            # 将HWC格式转换为BCHW格式
            img2_tensor = torch.from_numpy(np.transpose(img2, (2, 0, 1))).float().unsqueeze(0)
            
        img2 = img2_tensor.to(device)
    
    # 处理PyTorch张量格式的输入
    if img1.dim() == 2:  # (H, W)
        img1 = img1.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度 -> (1, 1, H, W)
    elif img1.dim() == 3:  # (C, H, W)
        img1 = img1.unsqueeze(0)  # 添加批次维度 -> (1, C, H, W)
        
    if img2.dim() == 2:  # (H, W)
        img2 = img2.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度 -> (1, 1, H, W)
    elif img2.dim() == 3:  # (C, H, W)
        img2 = img2.unsqueeze(0)  # 添加批次维度 -> (1, C, H, W)
    
    # 创建SSIM模块
    ssim_module = SSIM(
        win_size=win_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        gaussian_weights=gaussian_weights,
        use_sample_covariance=use_sample_covariance
    ).to(device)
    
    # 将图像移动到相同设备
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 计算SSIM
    with torch.no_grad():
        result = ssim_module(img1, img2, data_range=data_range, full=full, wo_light=wo_light)
    
    # 如果返回了SSIM图，处理输出格式
    if full:
        mssim, ssim_map = result
        
        # 保持与输入相同的格式返回SSIM图
        if is_array:
            # 如果输入是NumPy数组，将SSIM图也转回NumPy格式
            ssim_map = ssim_map.squeeze().cpu().numpy()
        
        return mssim, ssim_map
    else:
        return result
    
def filter_by_rectangularity(binary_image, labels, stats, centroids, filtered_labels, 
                            min_rectangularity=0.7, use_convex_hull=False, visualize=False, 
                            original_image=None, save_path=None, output_mask = False):
    """
    根据区域的矩形度对连通区域进行过滤
    
    Args:
        binary_image (numpy.ndarray): 二值图像
        labels (numpy.ndarray): 标记图像，每个像素值为该连通区域的标签
        stats (numpy.ndarray): 每个连通区域的统计信息 [x, y, width, height, area]
        centroids (numpy.ndarray): 每个连通区域的中心点坐标
        filtered_labels (list): 已初步过滤的连通区域标签列表
        min_rectangularity (float): 最小矩形度阈值，范围[0,1]，值越接近1表示区域越接近矩形
        use_convex_hull (bool): 是否使用凸包计算矩形度，对于有孔洞的区域更为准确
        visualize (bool): 是否可视化矩形度计算结果
        original_image (numpy.ndarray): 原始图像，用于可视化
        save_path (str): 可视化结果保存路径
        output_mask (bool): 是否输出矩形度过滤后的掩码图像
        
    Returns:
        tuple: (rect_filtered_labels, rectangularity_values)
            - rect_filtered_labels: 按矩形度过滤后的连通区域标签列表
            - rectangularity_values: 字典，键为标签，值为对应的矩形度
            - mask(optional): 可选的掩码图像    
    """
    # 确保二值图像是单通道的
    if len(binary_image.shape) > 2:
        raise ValueError("二值图像必须是单通道的")
    
    # 计算每个区域的矩形度
    rectangularity_values = {}
    rect_filtered_labels = []
    
    # 为可视化准备数据
    if visualize:
        vis_data = {
            'original': [],   # 原始掩码
            'contour': [],    # 轮廓
            'rect': [],       # 最小外接矩形
            'values': [],     # 矩形度值
            'labels': [],     # 标签
            'colors': []      # 颜色（绿色通过，红色过滤掉）
        }
    
    for label in filtered_labels:
        # 创建当前标签的掩码
        mask = (labels == label).astype(np.uint8) * 255
        
        # 找到区域的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:  # 防御性编程，确保找到轮廓
            continue
            
        contour = max(contours, key=cv2.contourArea)  # 取最大轮廓
        
        # 如果选择使用凸包，则计算凸包
        if use_convex_hull:
            contour = cv2.convexHull(contour)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算轮廓面积
        contour_area = cv2.contourArea(contour)
        
        # 计算外接矩形面积
        rect_area = cv2.contourArea(box)
        
        # 计算矩形度 (轮廓面积 / 最小外接矩形面积)
        rectangularity = contour_area / rect_area if rect_area > 0 else 0
        
        # 存储矩形度值
        rectangularity_values[label] = rectangularity
        
        # 过滤矩形度不够的区域
        if rectangularity >= min_rectangularity:
            rect_filtered_labels.append(label)
        
        # 准备可视化数据
        if visualize:
            vis_data['original'].append(mask)
            vis_data['contour'].append(contour)
            vis_data['rect'].append(box)
            vis_data['values'].append(rectangularity)
            vis_data['labels'].append(label)
            vis_data['colors'].append('green' if rectangularity >= min_rectangularity else 'red')
    
    # 输出过滤结果
    removed_count = len(filtered_labels) - len(rect_filtered_labels)
    print(f"矩形度过滤: 从 {len(filtered_labels)} 个区域中移除了 {removed_count} 个不规则区域")
    
    # 可视化矩形度计算结果
    if visualize and vis_data['original']:
        import matplotlib.pyplot as plt
        # 准备可视化
        num_regions = len(vis_data['original'])
        rows = min(4, num_regions)
        cols = (num_regions + rows - 1) // rows
        
        plt.figure(figsize=(cols * 5, rows * 4))
        
        for i in range(num_regions):
            plt.subplot(rows, cols, i + 1)
            
            # 创建可视化图像
            if original_image is not None:
                if len(original_image.shape) == 2:
                    vis_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                else:
                    vis_img = original_image.copy()
                
                # 提取当前掩码区域
                current_mask = vis_data['original'][i]
                
                # 叠加掩码到原图 (半透明)
                masked_img = vis_img.copy()
                masked_img[current_mask > 0] = [0, 0, 200]  # 蓝色填充掩码区域
                overlay = cv2.addWeighted(vis_img, 0.7, masked_img, 0.3, 0)
            else:
                # 如果没有原图，就只显示掩码
                overlay = np.stack([vis_data['original'][i]] * 3, axis=2)
            
            # 绘制轮廓和矩形
            result = overlay.copy()
            color = (0, 255, 0) if vis_data['colors'][i] == 'green' else (255, 0, 0)
            cv2.drawContours(result, [vis_data['contour'][i]], 0, color, 2)
            cv2.drawContours(result, [vis_data['rect'][i]], 0, (255, 255, 0), 2)
            
            # 显示图像
            plt.imshow(result)
            plt.title(f"标签 {vis_data['labels'][i]}, 矩形度 = {vis_data['values'][i]:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"矩形度可视化已保存至: {save_path}")
        
        plt.show()
    if output_mask:
        # 创建掩码图像
        mask_image = np.ones_like(binary_image, dtype=np.uint8)*255
        for label in rect_filtered_labels:
            mask_image[labels == label] = 0
        return rect_filtered_labels, rectangularity_values, mask_image
    
    return rect_filtered_labels, rectangularity_values


def crop_and_rotate(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    def crop_center(img, crop_width, crop_height):
        h, w = img.shape[:2]

        x = w//2 - crop_width//2
        y = h//2 - crop_height//2

        return img[y:y+crop_height, x:x+crop_width]
    
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    img_cropped = crop_center(img_rotated, 448, 448)

    return img_cropped


def overlay_mask_on_image(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    mask: Union[torch.Tensor, Image.Image, np.ndarray],
    alpha: float = 0.5,
    mask_color: Tuple[int, int, int] = (0, 0, 255),  # 默认蓝色
    to_tensor: bool = False,
    normalized: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    将二进制掩码叠加到RGB图像上
    
    参数:
        image: 输入RGB图像，可以是PyTorch张量、PIL图像或NumPy数组
        mask: 输入二进制掩码，可以是PyTorch张量、PIL图像或NumPy数组
        alpha: 掩码的透明度，范围[0,1]，0表示完全透明，1表示完全不透明
        mask_color: 掩码的颜色，RGB格式
        to_tensor: 是否将结果转换为PyTorch张量
        normalized: 是否将结果归一化到[0,1]范围
        
    返回:
        叠加后的图像，格式取决于to_tensor参数
    """
    # 转换图像为numpy数组
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:  # [B,C,H,W]
            image = image.squeeze(0)  # 移除批次维度
        
        if image.ndim == 3:  # [C,H,W]
            image = image.permute(1, 2, 0).contiguous()  # 转为[H,W,C]
        
        # 如果是归一化的张量，转换为0-255范围
        if image.max() <= 1.0 and image.dtype != torch.uint8:
            image = (image * 255.0).to(torch.uint8)
            
        image_np = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        image_np = np.array(image)
    else:  # np.ndarray
        image_np = image.copy()
    
    # 确保图像是RGB格式
    if image_np.ndim == 2:  # 灰度图
        image_np = np.stack([image_np] * 3, axis=2)
    elif image_np.shape[2] == 4:  # RGBA图
        image_np = image_np[:, :, :3]
    
    # 转换掩码为numpy数组
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4:  # [B,C,H,W]
            mask = mask.squeeze(0)  # 移除批次维度
        
        if mask.ndim == 3 and mask.shape[0] == 1:  # [1,H,W]
            mask = mask.squeeze(0)  # 移除通道维度
        
        if mask.ndim == 3:  # [C,H,W]
            mask = mask.permute(1, 2, 0).contiguous()  # 转为[H,W,C]
            mask = mask.mean(axis=2)  # 多通道取平均
        
        # 转换为二值掩码
        if mask.max() <= 1.0 and mask.dtype != torch.uint8:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = (mask > 0).cpu().numpy().astype(np.uint8)
    elif isinstance(mask, Image.Image):
        mask_np = np.array(mask)
        if mask_np.ndim == 3:  # 多通道
            mask_np = mask_np.mean(axis=2)
        mask_np = (mask_np > 0).astype(np.uint8)
    else:  # np.ndarray
        mask_np = mask.copy()
        if mask_np.ndim == 3:  # 多通道
            mask_np = mask_np.mean(axis=2)
        mask_np = (mask_np > 0).astype(np.uint8)
    
    # 确保掩码与图像尺寸匹配
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # 创建彩色掩码
    color_mask = np.zeros_like(image_np)
    color_mask[mask_np > 0] = mask_color
    
    # 叠加掩码到原图像
    overlay = cv2.addWeighted(image_np, 1, color_mask, alpha, 0)
    
    # 根据需要返回不同格式
    if to_tensor:
        # 转换为PyTorch张量
        if normalized:
            overlay = overlay.astype(np.float32) / 255.0
            
        tensor = torch.from_numpy(overlay).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        
        if normalized:
            tensor = tensor.float()
        else:
            tensor = tensor.to(torch.uint8)
            
        return tensor
    else:
        # 保持为NumPy数组
        if normalized:
            overlay = overlay.astype(np.float32) / 255.0
            
        return overlay

def convert_all_img(img_dir, output_dir, transform):
    """
    遍历目录中的所有图像，应用转换，并保存结果
    
    参数:
        img_dir: 输入图像目录
        output_dir: 输出图像目录
        transform: 转换函数
    """
    # 尝试处理一张图像
    from PIL import Image

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 首先获取所有图像文件
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 使用tqdm创建进度条
    from tqdm import tqdm
    for filename in tqdm(image_files, desc="处理图像", unit="张"):
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        transformed_img = transform(img)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, transformed_img)
    
    print(f"✓ 已将 {len(image_files)} 张图像处理并保存到 {output_dir}")

def convert_all_img2mask(img_dir, output_dir, transform):
    """
    遍历目录中的所有图像，应用转换，并保存结果
    
    参数:
        img_dir: 输入图像目录
        output_dir: 输出图像目录
        transform: 转换函数
    """
    # 尝试处理一张图像
    from PIL import Image

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 首先获取所有图像文件
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 使用tqdm创建进度条
    from tqdm import tqdm
    for filename in tqdm(image_files, desc="处理图像", unit="张"):
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        transformed_img = transform(img)
        output_path = os.path.join(output_dir, filename)
        # 保存PIL图像
        transformed_img.save(output_path)
    
    print(f"✓ 已将 {len(image_files)} 张图像处理并保存到 {output_dir}")

# *: 根据触觉生成touch_masks
# if __name__ == '__main__':
#     # 测试代码
#     img_path = '/home/sgh/data/WorkSpace/VisionNet/dataset/result/gel_image_template.png'
    
#     from config import PARAMS
#     M = PARAMS['m']
#     transform = TouchWeightMapTransform(
#         template_path=img_path,
#         min_area=500,
#         morph_operation='close_open',
#         min_rectangularity=0.9,
#         M = M,
#         canvas_size=(560, 560),
#         to_tensor=False,
#         normalized=False,
#     )
#     img_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/new_vision_touch/unpack/val/val_all/touch_images"
#     output_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/new_vision_touch/unpack/val/val_all/touch_masks"
#     convert_all_img(img_dir=img_dir, output_dir=output_dir, transform=transform)

# *: 根据视觉生成vision_masks
if __name__ == '__main__':
    
    from config import PARAMS
    yolo_model_path = PARAMS['pin_black_model_path']

    transform = VisionTerraceMapGenerator(
        yolo_model_path=yolo_model_path,
        return_mask=True,
    )
    img_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/data_all/rgb_images"
    output_dir = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/data_all/rgb_masks"

    convert_all_img2mask(img_dir=img_dir, output_dir=output_dir, transform=transform)