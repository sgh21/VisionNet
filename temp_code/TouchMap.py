import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List, Dict, Any
import math
import torch

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
        self.generate_terrace_map(img, serial = serial)
        
        return Image.fromarray(self.terrace_map)
        
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
        sample_contours = self.resample_contour(outer_contours[0], 128)
        print(type(outer_contours[0]))
        return self.terrace_map

def test_terrace_map():
    """
    Demonstrate the usage of TerraceMapGenerator
    """
    try:
        import torchvision.transforms as T
    except ImportError:
        print("torchvision library not installed, cannot run this example")
        return
    
    # Load test images
    rgb_img_path = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_4030P_0.png"
    mask_img_path = "/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/touch_images_mask_process/gel_image_4030P_0.png"
    
    from config import EXPANSION_SIZE
    # Load RGB image and mask
    rgb_img = Image.open(rgb_img_path).convert('RGB')
    test_mask = Image.open(mask_img_path).convert('L')
    
    # Create terrace map generator
    terrace_generator = TerraceMapGenerator(
        intensity_scaling=[0.2, 0.6, 0.8, 1.0],
        edge_enhancement=1.0,
        expansion_size=EXPANSION_SIZE,
        debug=True
    )
    
    transorms = T.Compose([
        T.ToTensor(),
    ])
    # Apply transformation
    terrace_map = terrace_generator(test_mask, serial='4024P')
    terrace_tensor = transorms(terrace_map)
    # Print shape
    print(f"Terrace Map Tensor Shape: {terrace_tensor.shape}")
    # Show RGB with terrace map overlay
    visualize_rgb_with_terrace(rgb_img, terrace_map)
    
    # # Print shape
    # print(f"Terrace Map Tensor Shape: {terrace_map.shape}")
    
    return terrace_map


def visualize_rgb_with_terrace(rgb_img, terrace_map):
    """
    Visualize RGB image with terrace map overlay
    
    Parameters:
        rgb_img: RGB image (PIL.Image)
        terrace_map: Terrace map (numpy.ndarray)
    """
    # Convert to numpy arrays
    rgb_np = np.array(rgb_img)
    terrace_map = np.array(terrace_map)
    # Create colored representation of the terrace map
    terrace_colored = cv2.applyColorMap(terrace_map, cv2.COLORMAP_VIRIDIS)
    # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
    terrace_colored = cv2.cvtColor(terrace_colored, cv2.COLOR_BGR2RGB)
    
    # Create a copy of RGB image for overlay
    overlay_rgb = rgb_np.copy()
    
    # Calculate overlay image - blend RGB image and terrace map in mask region
    alpha = 0.6  # Transparency, can be adjusted
    overlay_rgb = (alpha * overlay_rgb + (1-alpha) * terrace_colored).astype(np.uint8)
    
    # Prepare visualization

    from utils.VisualizeParam import visualize_images
    
    # Prepare images to display
    images = [
        terrace_colored,  # Colored terrace map
        overlay_rgb  # Overlay result
    ]
    
    # Image titles
    titles = [
        'Colored Terrace Map',
        'RGB with Terrace Map Overlay'
    ]
    

    # Use visualize_images function to display all images
    visualize_images(
        images=images,
        titles=titles,
        figsize=(18, 9),
        display=True,
    )

if __name__ == "__main__":
    # 测试梯田图生成
    print("\n测试梯田图生成:")
    try:
        test_terrace_map()
    except ImportError:
        print("无法运行梯田图生成测试: torchvision 未安装")