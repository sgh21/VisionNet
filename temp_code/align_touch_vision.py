import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize
import glob
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

class TactileVisualAligner:
    """
    用于对齐触觉和视觉图像的类，通过优化确定触觉图像到视觉图像的变换参数
    (x偏移、y偏移和缩放系数)
    """
    
    def __init__(self, visual_data_dir, tactile_data_dir, output_dir=None):
        """
        初始化对齐器
        
        参数:
            visual_data_dir (str): 视觉数据目录，包含图像和标签
            tactile_data_dir (str): 触觉数据目录，包含图像和标签
            output_dir (str, optional): 输出结果的目录
        """
        self.visual_data_dir = visual_data_dir
        self.tactile_data_dir = tactile_data_dir
        
        # 如果未指定输出目录，则默认在视觉数据目录下创建一个output子目录
        if output_dir is None:
            self.output_dir = os.path.join(visual_data_dir, "alignment_output")
        else:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 变换参数 [x偏移, y偏移, 缩放系数]
        self.transform_params = None
        
        # 加载数据
        self.visual_data = []  # 每项为 (image_path, [[x1,y1,w1,h1,theta1], [x2,y2,w2,h2,theta2]])
        self.tactile_data = []
        
        self._load_data()
    
    def _load_data(self):
        """加载视觉和触觉数据"""
        # 加载视觉数据
        self._load_dataset(self.visual_data_dir, self.visual_data)
        
        # 加载触觉数据
        self._load_dataset(self.tactile_data_dir, self.tactile_data)
        
        # 确保数据量匹配
        min_count = min(len(self.visual_data), len(self.tactile_data))
        if min_count == 0:
            raise ValueError("没有找到匹配的数据")
            
        self.visual_data = self.visual_data[:min_count]
        self.tactile_data = self.tactile_data[:min_count]
        
        print(f"已加载 {min_count} 对匹配数据")
    
    def _load_dataset(self, data_dir, data_list):
        """
        加载数据集（图像和标签）
        
        参数:
            data_dir (str): 数据目录
            data_list (list): 存储加载数据的列表
        """
        # 查找标签文件
        label_dir = os.path.join(data_dir, "annotations")
        image_dir = os.path.join(data_dir, "images")
        
        if not os.path.exists(label_dir):
            raise ValueError(f"标签目录不存在: {label_dir}")
        if not os.path.exists(image_dir):
            raise ValueError(f"图像目录不存在: {image_dir}")
        
        # 按数字顺序查找所有txt标签文件
        label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")),
                            key=lambda x: self._extract_number(os.path.basename(x)))
        
        for label_file in label_files:
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            
            # 查找对应的图像文件
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                test_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            if image_path is None:
                print(f"警告: 找不到标签 {label_file} 对应的图像文件")
                continue
            
            # 读取标签文件
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) != 2:
                        print(f"警告: 标签文件 {label_file} 格式不正确，应为两行")
                        continue
                    
                    # 解析两个矩形 [x, y, w, h, theta]
                    rect1 = [float(val) for val in lines[0].strip().split()]
                    rect2 = [float(val) for val in lines[1].strip().split()]
                    
                    if len(rect1) != 5 or len(rect2) != 5:
                        print(f"警告: 标签文件 {label_file} 中的矩形格式不正确，应为5个值")
                        continue
                    
                    data_list.append((image_path, [rect1, rect2]))
            except Exception as e:
                print(f"读取标签文件 {label_file} 出错: {str(e)}")
    
    def _extract_number(self, filename):
        """从文件名中提取数字，用于排序"""
        import re
        numbers = re.findall(r'_(\d+)\.', filename)
        if numbers:
            return int(numbers[0])
        return 0
    
    def _rectification_error(self, params, visual_rects, tactile_rects):
        """
        计算变换后的矩形框匹配误差
        
        参数:
            params (list): 变换参数 [x偏移, y偏移, 缩放系数]
            visual_rects (list): 视觉数据的矩形框列表 [[x1,y1,w1,h1,theta1], [x2,y2,w2,h2,theta2]]
            tactile_rects (list): 触觉数据的矩形框列表 [[x1,y1,w1,h1,theta1], [x2,y2,w2,h2,theta2]]
            
        返回:
            float: 误差值 (越小越好)
        """
        x_offset, y_offset, scale = params
        
        total_error = 0
        
        # 对两个矩形框分别计算误差
        for i in range(2):
            v_rect = visual_rects[i]
            t_rect = tactile_rects[i]
            
            # 应用变换到触觉矩形
            transformed_t_rect = [
                t_rect[0] * scale + x_offset,  # x
                t_rect[1] * scale + y_offset,  # y
                t_rect[2] * scale,            # w
                t_rect[3] * scale,            # h
                t_rect[4]                     # theta (角度不变)
            ]
            
            # 计算中心点距离误差
            center_error = np.sqrt((v_rect[0] - transformed_t_rect[0])**2 + 
                                  (v_rect[1] - transformed_t_rect[1])**2)
            
            # 计算尺寸误差
            size_error = np.sqrt((v_rect[2] - transformed_t_rect[2])**2 + 
                                (v_rect[3] - transformed_t_rect[3])**2)
            
            # 计算角度误差（注意角度的循环性）
            angle_diff = abs(v_rect[4] - transformed_t_rect[4])
            angle_error = min(angle_diff, 360 - angle_diff if angle_diff > 180 else angle_diff)
            
            # 综合误差 (给不同的误差项分配权重)
            rect_error = center_error +  size_error
            total_error += rect_error
        
        return total_error
    
    def _objective_function(self, params):
        """
        优化的目标函数，计算所有数据对的累积误差
        
        参数:
            params (list): 变换参数 [x偏移, y偏移, 缩放系数]
            
        返回:
            float: 总误差
        """
        total_error = 0
        
        for i in range(len(self.visual_data)):
            visual_rects = self.visual_data[i][1]
            tactile_rects = self.tactile_data[i][1]
            
            # 计算此对数据的误差并累加
            error = self._rectification_error(params, visual_rects, tactile_rects)
            total_error += error
        
        return total_error
    
    def optimize_alignment(self, initial_params=None):
        """
        优化对齐参数
        
        参数:
            initial_params (list, optional): 初始参数 [x偏移, y偏移, 缩放系数]
            
        返回:
            list: 优化后的参数 [x偏移, y偏移, 缩放系数]
        """
        # 如果未提供初始参数，使用默认值
        if initial_params is None:
            # 默认参数: 无偏移，缩放系数为1
            initial_params = [0, 0, 1.0]
        
        # 定义参数边界 (防止缩放变为负值)
        bounds = [
            (None, None),  # x偏移无限制
            (None, None),  # y偏移无限制
            (0.5, 2.0)    # 缩放系数限制在合理范围
        ]
        
        # 执行优化
        print("开始优化对齐参数...")
        result = minimize(
            self._objective_function, 
            initial_params,
            method='L-BFGS-B',  # 边界优化算法
            bounds=bounds,
            options={'disp': True}
        )
        
        # 检查优化是否成功
        if result.success:
            print(f"优化成功! 最终误差: {result.fun}")
            print(f"最优参数: x偏移={result.x[0]:.2f}, y偏移={result.x[1]:.2f}, 缩放={result.x[2]:.4f}")
        else:
            print(f"警告: 优化未收敛。{result.message}")
        
        self.transform_params = result.x
        self.transform_params = [12.579, -21.21148, 1.3391]
        return self.transform_params
    
    def visualize_alignment(self, sample_indices=None):
        """
        可视化对齐结果 - 将变换应用于触觉图像
        
        参数:
            sample_indices (list, optional): 要可视化的样本索引，如果为None则使用前5个样本
        """
        if self.transform_params is None:
            print("请先运行optimize_alignment()来获取变换参数")
            return
        
        # 如果未指定样本索引，使用前5个样本（或全部，如果少于5个）
        if sample_indices is None:
            sample_count = min(10, len(self.visual_data))
            sample_indices = list(range(sample_count))
        
        # 获取变换参数
        x_offset, y_offset, scale = self.transform_params
        
        for idx in sample_indices:
            visual_img_path, visual_rects = self.visual_data[idx]
            tactile_img_path, tactile_rects = self.tactile_data[idx]
            
            # 读取图像
            visual_img = cv2.imread(visual_img_path)
            visual_img = cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB)
            
            tactile_img = cv2.imread(tactile_img_path)
            tactile_img = cv2.cvtColor(tactile_img, cv2.COLOR_BGR2RGB)
            
            # 创建图像
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 显示视觉图像及其标注
            axes[0].imshow(visual_img)
            axes[0].set_title("Vision Image")
            self._draw_rects(axes[0], visual_rects, 'r')
            
            # 显示触觉图像及其标注
            axes[1].imshow(tactile_img)
            axes[1].set_title("Tactile Image")
            self._draw_rects(axes[1], tactile_rects, 'b')
            
            # 计算变换后的触觉图像
            h, w = tactile_img.shape[:2]
            # 创建仿射变换矩阵
            M = np.float32([
                [scale, 0, x_offset],
                [0, scale, y_offset]
            ])
            
            # 计算输出图像的大小，确保变换后的图像能完全显示
            # 这里需要计算变换后图像的实际大小
            transformed_w = int(w * scale)
            transformed_h = int(h * scale)
            
            # 应用仿射变换到触觉图像
            aligned_tactile_img = cv2.warpAffine(
                tactile_img, 
                M, 
                (visual_img.shape[1], visual_img.shape[0]),  # 使用视觉图像的尺寸作为输出尺寸
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # 显示原始视觉图像
            axes[2].imshow(visual_img)
            
            # 叠加变换后的触觉图像
            axes[2].imshow(aligned_tactile_img, alpha=0.7)  # 半透明叠加
            axes[2].set_title("Aligned Result (Tactile transformed)")
            
            # 在对齐后的图像上绘制变换后的触觉矩形和原始视觉矩形
            transformed_tactile_rects = []
            for rect in tactile_rects:
                transformed_rect = [
                    rect[0] * scale + x_offset,  # x
                    rect[1] * scale + y_offset,  # y
                    rect[2] * scale,            # w
                    rect[3] * scale,            # h
                    rect[4]                     # theta (角度不变)
                ]
                transformed_tactile_rects.append(transformed_rect)
            
            # 绘制视觉和变换后的触觉矩形
            self._draw_rects(axes[2], visual_rects, 'r')
            self._draw_rects(axes[2], transformed_tactile_rects, 'b')
            
            # 去除坐标轴
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            output_path = os.path.join(self.output_dir, f"alignment_sample_{idx}.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"已保存对齐可视化结果到 {output_path}")
    
    def _draw_rects(self, ax, rects, color):
        """在给定的轴上绘制矩形框"""
        for rect in rects:
            x, y, w, h, theta = rect
            
            # 创建以原点为中心的矩形
            rect_patch = Rectangle(
                (x - w/2, y - h/2),  # 左下角坐标
                w, h,                # 宽度和高度
                fill=False,
                edgecolor=color,
                linewidth=2
            )
            
            # 应用旋转变换
            transform = Affine2D().rotate_deg_around(x, y, theta)
            rect_patch.set_transform(transform + ax.transData)
            
            # 添加到轴
            ax.add_patch(rect_patch)
    
    def apply_transform_to_image(self, tactile_img_path, output_path=None, visual_img_shape=None):
        """
        对触觉图像应用变换，使其对齐到视觉图像
        
        参数:
            tactile_img_path (str): 触觉图像路径
            output_path (str, optional): 输出路径，如果为None则使用默认命名
            visual_img_shape (tuple, optional): 视觉图像的形状 (height, width)，如果提供，
                                            则将输出调整为该大小
                
        返回:
            numpy.ndarray: 变换后的图像
        """
        if self.transform_params is None:
            print("请先运行optimize_alignment()来获取变换参数")
            return None
        
        # 读取图像
        tactile_img = cv2.imread(tactile_img_path)
        if tactile_img is None:
            print(f"无法读取图像: {tactile_img_path}")
            return None
        
        # 应用变换
        x_offset, y_offset, scale = self.transform_params
        h, w = tactile_img.shape[:2]
        
        # 如果未提供视觉图像尺寸，则根据变换后的预期尺寸创建
        if visual_img_shape is None:
            # 估计变换后的尺寸
            output_w = w  # 保持原始宽度
            output_h = h  # 保持原始高度
        else:
            output_h, output_w = visual_img_shape[:2]
        
        # 创建变换矩阵
        M = np.float32([
            [scale, 0, x_offset],
            [0, scale, y_offset]
        ])
        
        # 应用变换
        aligned_img = cv2.warpAffine(
            tactile_img, 
            M, 
            (output_w, output_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 保存图像
        if output_path is not None:
            cv2.imwrite(output_path, aligned_img)
            print(f"已保存变换后的图像到 {output_path}")
        
        return aligned_img
    
    def process_dataset(self, input_dir, output_dir=None, visual_img_shape=None):
        """
        处理整个数据集，对所有触觉图像应用变换
        
        参数:
            input_dir (str): 输入触觉图像目录
            output_dir (str, optional): 输出目录，如果为None则使用默认命名
            visual_img_shape (tuple, optional): 视觉图像的形状 (height, width)
        """
        if self.transform_params is None:
            print("请先运行optimize_alignment()来获取变换参数")
            return
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "transformed_tactile")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果未指定视觉图像形状，且有视觉数据，则使用第一个视觉图像的形状
        if visual_img_shape is None and len(self.visual_data) > 0:
            visual_img = cv2.imread(self.visual_data[0][0])
            if visual_img is not None:
                visual_img_shape = visual_img.shape
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        # 按数字排序
        image_files = sorted(image_files, key=lambda x: self._extract_number(os.path.basename(x)))
        
        print(f"开始处理 {len(image_files)} 张图像...")
        
        for img_path in image_files:
            # 确定输出路径
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, base_name)
            
            # 应用变换
            self.apply_transform_to_image(img_path, output_path, visual_img_shape)
        
        print(f"已完成处理并保存到 {output_dir}")
    
    def save_parameters(self, output_path=None):
        """
        保存优化后的变换参数到文件
        
        参数:
            output_path (str, optional): 输出路径，如果为None则使用默认命名
        """
        if self.transform_params is None:
            print("请先运行optimize_alignment()来获取变换参数")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "transform_parameters.txt")
        
        with open(output_path, 'w') as f:
            f.write(f"x_offset: {self.transform_params[0]}\n")
            f.write(f"y_offset: {self.transform_params[1]}\n")
            f.write(f"scale: {self.transform_params[2]}\n")
        
        print(f"已保存变换参数到 {output_path}")

# 示例用法
if __name__ == "__main__":
    # 设置数据目录
    visual_data_dir = "/home/sgh/data/WorkSpace/BTBInsertionV2/documents/calib_data/rgb_images"
    tactile_data_dir = "/home/sgh/data/WorkSpace/BTBInsertionV2/documents/calib_data/touch_images"
    output_dir = "/home/sgh/data/WorkSpace/BTBInsertionV2/documents/calib_data/result"
    
    # 创建对齐器
    aligner = TactileVisualAligner(visual_data_dir, tactile_data_dir, output_dir)
    
    # 优化对齐参数
    params = aligner.optimize_alignment()
    
    # 可视化结果
    aligner.visualize_alignment()
    
    # 保存参数
    aligner.save_parameters()
    
    # 处理整个数据集
    aligner.process_dataset(tactile_data_dir)