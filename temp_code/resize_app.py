import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QFileDialog, QCheckBox,
                            QGroupBox, QTextEdit, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

class DeNormalize(object):
    """反归一化变换"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): 要反归一化的张量
        Returns:
            Tensor: 反归一化后的张量
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def calculate_error_metrics(original, reconstructed):
    """计算原始图像和重建图像之间的误差指标"""
    # 确保两个图像都是相同大小的张量
    assert original.shape == reconstructed.shape, "图像尺寸不匹配"
    
    # 计算像素绝对误差
    abs_diff = torch.abs(original - reconstructed)
    
    # 各种误差指标
    mae = torch.mean(abs_diff).item()  # 平均绝对误差
    mse = torch.mean((original - reconstructed) ** 2).item()  # 均方误差
    rmse = torch.sqrt(torch.mean((original - reconstructed) ** 2)).item()  # 均方根误差
    max_error = torch.max(abs_diff).item()  # 最大误差
    
    # 计算PSNR (Peak Signal-to-Noise Ratio)
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item() if mse > 0 else float('inf')
    
    # 计算各通道的误差
    channel_mae = [torch.mean(abs_diff[c]).item() for c in range(abs_diff.shape[0])]
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_error': max_error,
        'psnr': psnr,
        'channel_mae': channel_mae
    }

def save_tensor_as_image(tensor, filename):
    """将tensor保存为图像文件"""
    # 确保张量在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    # 转换为numpy数组
    img = tensor.permute(1, 2, 0).numpy() * 255.0
    img = img.astype(np.uint8)
    # 保存图像
    Image.fromarray(img).save(filename)

class TransformWorker(QThread):
    """图像处理工作线程"""
    update_log = pyqtSignal(str)
    complete = pyqtSignal(dict, dict)  # 结果, 图像张量
    
    def __init__(self, image_path, size, original_size, use_norm=True, interp_mode="BICUBIC"):
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.original_size = original_size
        self.use_norm = use_norm
        self.interp_mode = interp_mode
        
    def run(self):
        self.update_log.emit("开始处理图像...")
        
        # 加载图像
        self.update_log.emit(f"加载图像: {self.image_path}")
        try:
            original_image = Image.open(self.image_path).convert('RGB')
        except Exception as e:
            self.update_log.emit(f"错误: 无法加载图像 - {str(e)}")
            return
            
        # 定义均值和标准差 (ImageNet标准)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # 获取插值模式
        interp_mode_map = {
            "NEAREST": transforms.InterpolationMode.NEAREST,
            "BILINEAR": transforms.InterpolationMode.BILINEAR,
            "BICUBIC": transforms.InterpolationMode.BICUBIC,
            "LANCZOS": transforms.InterpolationMode.LANCZOS
        }
        interp_mode = interp_mode_map.get(self.interp_mode, transforms.InterpolationMode.BICUBIC)
        # 原始大小的图像转张量
        transform_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 缩小和标准化变换
        transform_list = [transforms.Resize((self.size, self.size)), transforms.ToTensor()]
        if self.use_norm:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transform_list)
        
        # 反向变换
        inverse_list = []
        if self.use_norm:
            inverse_list.append(DeNormalize(mean=mean, std=std))
        inverse_list.append(transforms.Resize((self.original_size, self.original_size),interpolation=interp_mode))
        inverse_transform = transforms.Compose(inverse_list)
        
        # 应用变换
        self.update_log.emit("应用缩小和标准化变换...")
        original_tensor = transform_tensor(original_image)
        normalized_tensor = transform(original_image)
        
        # 应用反向变换
        self.update_log.emit("应用反归一化和放大变换...")
        reconstructed_tensor = inverse_transform(normalized_tensor)
        
        # 计算误差
        self.update_log.emit("计算误差指标...")
        error_metrics = calculate_error_metrics(original_tensor, reconstructed_tensor)
        
        # 为可视化准备图像张量
        tensors = {
            'original': original_tensor,
            'small': normalized_tensor.clone(),
            'reconstructed': reconstructed_tensor
        }
        
        # 如果使用了标准化，将小图反归一化用于可视化
        if self.use_norm:
            tensors['small'] = DeNormalize(mean=mean, std=std)(tensors['small'])
        
        # 创建差异张量
        diff_tensor = torch.abs(original_tensor - reconstructed_tensor) * 10.0  # 放大10倍便于观察
        tensors['diff'] = diff_tensor
        
        self.update_log.emit("处理完成!")
        self.complete.emit(error_metrics, tensors)

class ImageTransformApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像变换质量评估工具")
        self.setGeometry(100, 100, 1200, 800)
        
        self.image_path = ""
        self.current_tensors = {}
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图像选择控件
        img_group = QGroupBox("输入图像")
        img_layout = QVBoxLayout()
        
        self.image_path_label = QLabel("未选择图像")
        self.image_path_label.setWordWrap(True)
        img_layout.addWidget(self.image_path_label)
        
        self.select_image_btn = QPushButton("选择图像")
        self.select_image_btn.clicked.connect(self.select_image)
        img_layout.addWidget(self.select_image_btn)
        
        # 添加创建测试图像按钮
        self.create_test_img_btn = QPushButton("创建测试图像")
        self.create_test_img_btn.clicked.connect(self.create_test_image)
        img_layout.addWidget(self.create_test_img_btn)
        
        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)
        
        # 参数设置控件
        param_group = QGroupBox("变换参数")
        param_layout = QVBoxLayout()
        
        # 缩小尺寸参数
        small_size_layout = QHBoxLayout()
        small_size_layout.addWidget(QLabel("缩小尺寸:"))
        self.small_size_spin = QSpinBox()
        self.small_size_spin.setRange(16, 1024)
        self.small_size_spin.setSingleStep(8)
        self.small_size_spin.setValue(224)
        small_size_layout.addWidget(self.small_size_spin)
        param_layout.addLayout(small_size_layout)
        
        # 原始/放大尺寸参数
        orig_size_layout = QHBoxLayout()
        orig_size_layout.addWidget(QLabel("放大尺寸:"))
        self.orig_size_spin = QSpinBox()
        self.orig_size_spin.setRange(16, 4096)
        self.orig_size_spin.setSingleStep(8)
        self.orig_size_spin.setValue(560)  # 默认224*2.5
        orig_size_layout.addWidget(self.orig_size_spin)
        param_layout.addLayout(orig_size_layout)
        
        # 添加插值模式选择下拉框
        from PyQt5.QtWidgets import QComboBox
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("插值模式:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"])
        self.interp_combo.setCurrentText("BICUBIC")  # 默认使用BICUBIC
        interp_layout.addWidget(self.interp_combo)
        param_layout.addLayout(interp_layout)

        # 是否使用标准化
        self.use_norm_check = QCheckBox("应用标准化")
        self.use_norm_check.setChecked(True)
        param_layout.addWidget(self.use_norm_check)
        
        # 放大倍数滑块（用于差异可视化）
        diff_scale_layout = QHBoxLayout()
        diff_scale_layout.addWidget(QLabel("差异放大倍数:"))
        self.diff_scale_slider = QSlider(Qt.Horizontal)
        self.diff_scale_slider.setRange(1, 50)
        self.diff_scale_slider.setValue(10)
        self.diff_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.diff_scale_slider.setTickInterval(5)
        self.diff_scale_slider.valueChanged.connect(self.update_diff_image)
        diff_scale_layout.addWidget(self.diff_scale_slider)
        self.diff_scale_label = QLabel("10x")
        diff_scale_layout.addWidget(self.diff_scale_label)
        param_layout.addLayout(diff_scale_layout)
        
        # 运行按钮
        self.run_btn = QPushButton("运行变换")
        self.run_btn.clicked.connect(self.run_transform)
        param_layout.addWidget(self.run_btn)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # 日志显示
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 右侧显示面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像显示区域
        image_view_group = QGroupBox("图像比较")
        image_view_layout = QVBoxLayout()
        
        # 使用matplotlib画布显示图像
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        image_view_layout.addWidget(self.canvas)
        
        image_view_group.setLayout(image_view_layout)
        right_layout.addWidget(image_view_group, 3)
        
        # 结果显示区域
        results_group = QGroupBox("评估结果")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier New", 10))
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group, 1)
        
        # 添加到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        
        # 添加欢迎消息
        self.log_message("欢迎使用图像变换质量评估工具。请选择一张图像开始。")
        
        # 初始化图像显示
        self.update_figures(None)
    
    def log_message(self, message):
        """向日志区域添加消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def select_image(self):
        """选择图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.image_path = file_path
            self.image_path_label.setText(os.path.basename(file_path))
            self.log_message(f"已选择图像: {os.path.basename(file_path)}")
    
    def create_test_image(self):
        """创建棋盘格测试图像"""
        size = 560  # 默认大小与原始尺寸相同
        grid_size = size // 8
        
        # 创建棋盘格
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, grid_size):
            for j in range(0, size, grid_size):
                if (i // grid_size + j // grid_size) % 2 == 0:
                    img[i:i+grid_size, j:j+grid_size] = 255
        
        # 添加红色对角线
        for i in range(size):
            if 0 <= i < size and 0 <= i < size:
                img[i, i] = [255, 0, 0]
            if 0 <= i < size and 0 <= size-i-1 < size:
                img[i, size-i-1] = [255, 0, 0]
        
        # 添加彩色中心标记
        center = size // 2
        marker_size = size // 16
        
        # 横线（绿色）
        img[center-marker_size//2:center+marker_size//2, center-marker_size*2:center+marker_size*2] = [0, 255, 0]
        
        # 竖线（蓝色）
        img[center-marker_size*2:center+marker_size*2, center-marker_size//2:center+marker_size//2] = [0, 0, 255]
        
        # 保存为临时文件
        temp_dir = os.path.join(os.path.expanduser("~"), ".image_transform_tool")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "test_image.png")
        
        Image.fromarray(img).save(temp_file)
        
        self.image_path = temp_file
        self.image_path_label.setText("测试棋盘格图像")
        self.log_message("已创建测试棋盘格图像")
    
    def run_transform(self):
        """运行图像变换"""
        if not self.image_path:
            self.log_message("错误: 请先选择图像")
            return
        
        # 禁用运行按钮
        self.run_btn.setEnabled(False)
        
        # 获取参数
        small_size = self.small_size_spin.value()
        orig_size = self.orig_size_spin.value()
        use_norm = self.use_norm_check.isChecked()
        interp_mode = self.interp_combo.currentText()

        # 创建并启动处理线程
        self.worker = TransformWorker(
            self.image_path, small_size, orig_size, use_norm
        )
        self.worker.update_log.connect(self.log_message)
        self.worker.complete.connect(self.on_transform_complete)
        self.worker.start()
    
    def on_transform_complete(self, error_metrics, tensors):
        """变换完成后的回调"""
        # 保存当前张量
        self.current_tensors = tensors
        
        # 更新图像显示
        self.update_figures(tensors)
        
        # 显示评估结果
        self.show_results(error_metrics)
        
        # 重新启用运行按钮
        self.run_btn.setEnabled(True)
    
    def update_figures(self, tensors):
        """更新图像显示"""
        self.figure.clear()
        
        if tensors is None:
            # 无数据时显示提示
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please select an image to transform.", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14)
            ax.axis('off')
            self.canvas.draw()
            return
        
        # 设置子图布局
        titles = ['original', 'small', 'reconstructed', f'diff (x{self.diff_scale_slider.value()})']
        
        for i, (key, title) in enumerate(zip(['original', 'small', 'reconstructed', 'diff'], titles)):
            ax = self.figure.add_subplot(2, 2, i+1)
            
            # 转换张量为可显示的图像
            img_np = tensors[key].permute(1, 2, 0).numpy()
            
            # 确保在[0,1]范围内
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.set_title(title)
            ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_diff_image(self):
        """更新差异图像的放大倍数"""
        scale = self.diff_scale_slider.value()
        self.diff_scale_label.setText(f"{scale}x")
        
        # 如果没有当前张量，直接返回
        if not self.current_tensors:
            return
            
        # 重新计算差异张量
        diff_tensor = torch.abs(self.current_tensors['original'] - self.current_tensors['reconstructed']) * scale
        self.current_tensors['diff'] = diff_tensor
        
        # 更新图像显示
        self.update_figures(self.current_tensors)
    
    def show_results(self, metrics):
        """显示评估结果"""
        result_text = "# 图像变换质量评估结果\n\n"
        
        result_text += f"## 误差指标\n"
        result_text += f"平均绝对误差 (MAE): {metrics['mae']:.8f}\n"
        result_text += f"均方误差 (MSE): {metrics['mse']:.8f}\n"
        result_text += f"均方根误差 (RMSE): {metrics['rmse']:.8f}\n"
        result_text += f"最大像素误差: {metrics['max_error']:.8f}\n"
        result_text += f"插值模式: {self.interp_combo.currentText()}\n"
        result_text += f"峰值信噪比 (PSNR): {metrics['psnr']:.2f} dB\n\n"
        
        result_text += f"## 各通道误差\n"
        result_text += f"红色通道 (R): {metrics['channel_mae'][0]:.8f}\n"
        result_text += f"绿色通道 (G): {metrics['channel_mae'][1]:.8f}\n"
        result_text += f"蓝色通道 (B): {metrics['channel_mae'][2]:.8f}\n\n"
        
        result_text += f"## 质量评估\n"
        if metrics['mae'] < 0.001:
            result_text += "图像质量损失极小: 变换过程导致的信息损失几乎不可见。\n"
        elif metrics['mae'] < 0.01:
            result_text += "图像质量损失较小: 变换过程导致的信息损失轻微，一般应用中可以接受。\n"
        elif metrics['mae'] < 0.05:
            result_text += "图像质量损失中等: 变换过程导致明显的信息损失，在某些应用中可能产生影响。\n"
        else:
            result_text += "图像质量损失严重: 变换过程导致显著的信息损失，可能不适合对图像精度有严格要求的应用。\n"
        
        self.results_text.setText(result_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTransformApp()
    window.show()
    sys.exit(app.exec_())