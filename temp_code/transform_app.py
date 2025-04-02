import sys
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog,
                             QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torchvision.transforms as transforms
from PIL import Image

class TransformVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initUI()
        self.img1 = None
        self.img2 = None
        self.img1_tensor = None
        self.img2_tensor = None
        self._slider_updating = False  # 防止循环更新
        
    def initUI(self):
        self.setWindowTitle('图像变换可视化工具')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 添加图像显示区域
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)
        
        # 创建更精细的控制区域布局
        controls_layout = QGridLayout()
        
        # 创建验证器，限制输入范围
        angle_validator = QDoubleValidator(-180.0, 180.0, 2)
        coord_validator = QDoubleValidator(-1.0, 1.0, 4)
        
        # 旋转角度控制
        self.rotation_label = QLabel('旋转角度:')
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(45)
        self.rotation_input = QLineEdit('0.0')
        self.rotation_input.setValidator(angle_validator)
        self.rotation_input.setMaximumWidth(80)
        
        # 旋转中心X控制
        self.cx_label = QLabel('旋转中心X:')
        self.cx_slider = QSlider(Qt.Horizontal)
        self.cx_slider.setRange(-100, 100)
        self.cx_slider.setValue(0)
        self.cx_input = QLineEdit('0.0')
        self.cx_input.setValidator(coord_validator)
        self.cx_input.setMaximumWidth(80)
        
        # 旋转中心Y控制
        self.cy_label = QLabel('旋转中心Y:')
        self.cy_slider = QSlider(Qt.Horizontal)
        self.cy_slider.setRange(-100, 100)
        self.cy_slider.setValue(0)
        self.cy_input = QLineEdit('0.0')
        self.cy_input.setValidator(coord_validator)
        self.cy_input.setMaximumWidth(80)
        
        # 平移X控制
        self.tx_label = QLabel('平移X:')
        self.tx_slider = QSlider(Qt.Horizontal)
        self.tx_slider.setRange(-100, 100)
        self.tx_slider.setValue(0)
        self.tx_input = QLineEdit('0.0')
        self.tx_input.setValidator(coord_validator)
        self.tx_input.setMaximumWidth(80)
        
        # 平移Y控制
        self.ty_label = QLabel('平移Y:')
        self.ty_slider = QSlider(Qt.Horizontal)
        self.ty_slider.setRange(-100, 100)
        self.ty_slider.setValue(0)
        self.ty_input = QLineEdit('0.0')
        self.ty_input.setValidator(coord_validator)
        self.ty_input.setMaximumWidth(80)
        
        # 添加到布局
        row = 0
        # 旋转角度
        controls_layout.addWidget(self.rotation_label, row, 0)
        controls_layout.addWidget(self.rotation_slider, row, 1)
        controls_layout.addWidget(self.rotation_input, row, 2)
        row += 1
        
        # 旋转中心X
        controls_layout.addWidget(self.cx_label, row, 0)
        controls_layout.addWidget(self.cx_slider, row, 1)
        controls_layout.addWidget(self.cx_input, row, 2)
        row += 1
        
        # 旋转中心Y
        controls_layout.addWidget(self.cy_label, row, 0)
        controls_layout.addWidget(self.cy_slider, row, 1)
        controls_layout.addWidget(self.cy_input, row, 2)
        row += 1
        
        # 平移X
        controls_layout.addWidget(self.tx_label, row, 0)
        controls_layout.addWidget(self.tx_slider, row, 1)
        controls_layout.addWidget(self.tx_input, row, 2)
        row += 1
        
        # 平移Y
        controls_layout.addWidget(self.ty_label, row, 0)
        controls_layout.addWidget(self.ty_slider, row, 1)
        controls_layout.addWidget(self.ty_input, row, 2)
        
        # 连接信号和槽
        self.rotation_slider.valueChanged.connect(self.slider_value_changed)
        self.cx_slider.valueChanged.connect(self.slider_value_changed)
        self.cy_slider.valueChanged.connect(self.slider_value_changed)
        self.tx_slider.valueChanged.connect(self.slider_value_changed)
        self.ty_slider.valueChanged.connect(self.slider_value_changed)
        
        self.rotation_input.editingFinished.connect(self.input_value_changed)
        self.cx_input.editingFinished.connect(self.input_value_changed)
        self.cy_input.editingFinished.connect(self.input_value_changed)
        self.tx_input.editingFinished.connect(self.input_value_changed)
        self.ty_input.editingFinished.connect(self.input_value_changed)
        
        main_layout.addLayout(controls_layout)
        
        # 添加按钮区域
        buttons_layout = QHBoxLayout()
        
        self.load_img1_btn = QPushButton('加载图像1')
        self.load_img1_btn.clicked.connect(self.load_image1)
        buttons_layout.addWidget(self.load_img1_btn)
        
        self.load_img2_btn = QPushButton('加载图像2')
        self.load_img2_btn.clicked.connect(self.load_image2)
        buttons_layout.addWidget(self.load_img2_btn)
        
        self.reset_btn = QPushButton('重置参数')
        self.reset_btn.clicked.connect(self.reset_params)
        buttons_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # 设置主窗口部件
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 初始化图像显示
        self.init_plots()
    
    def slider_value_changed(self):
        """当滑动条值改变时更新输入框并执行变换"""
        if self._slider_updating:
            return
            
        self._slider_updating = True
        
        # 获取滑动条值并更新输入框
        rotation_value = self.rotation_slider.value()
        cx_value = self.cx_slider.value() / 100.0
        cy_value = self.cy_slider.value() / 100.0
        tx_value = self.tx_slider.value() / 100.0
        ty_value = self.ty_slider.value() / 100.0
        
        self.rotation_input.setText(f"{rotation_value:.1f}")
        self.cx_input.setText(f"{cx_value:.2f}")
        self.cy_input.setText(f"{cy_value:.2f}")
        self.tx_input.setText(f"{tx_value:.2f}")
        self.ty_input.setText(f"{ty_value:.2f}")
        
        # 执行变换
        self.update_transform()
        
        self._slider_updating = False
    
    def input_value_changed(self):
        """当输入框值改变时更新滑动条并执行变换"""
        if self._slider_updating:
            return
            
        self._slider_updating = True
        
        # 获取输入框值并更新滑动条
        try:
            rotation_value = float(self.rotation_input.text())
            cx_value = float(self.cx_input.text())
            cy_value = float(self.cy_input.text())
            tx_value = float(self.tx_input.text())
            ty_value = float(self.ty_input.text())
            
            # 确保值在有效范围内
            rotation_value = max(-180, min(180, rotation_value))
            cx_value = max(-1, min(1, cx_value))
            cy_value = max(-1, min(1, cy_value))
            tx_value = max(-1, min(1, tx_value))
            ty_value = max(-1, min(1, ty_value))
            
            # 更新滑动条
            self.rotation_slider.setValue(int(rotation_value))
            self.cx_slider.setValue(int(cx_value * 100))
            self.cy_slider.setValue(int(cy_value * 100))
            self.tx_slider.setValue(int(tx_value * 100))
            self.ty_slider.setValue(int(ty_value * 100))
            
            # 更新输入框（确保显示合法值）
            self.rotation_input.setText(f"{rotation_value:.1f}")
            self.cx_input.setText(f"{cx_value:.2f}")
            self.cy_input.setText(f"{cy_value:.2f}")
            self.tx_input.setText(f"{tx_value:.2f}")
            self.ty_input.setText(f"{ty_value:.2f}")
            
            # 执行变换
            self.update_transform()
            
        except ValueError:
            # 输入值无效，重置为当前滑动条值
            self.slider_value_changed()
            
        self._slider_updating = False
        
    def init_plots(self):
        # [代码保持不变]...
        self.fig.clear()
        
        # 创建子图
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        self.ax1.set_title('原始图像1')
        self.ax2.set_title('目标图像2')
        self.ax3.set_title('变换后图像1')
        self.ax4.set_title('差值图像')
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.axis('off')
        
        self.canvas.draw()
        
    def load_image1(self):
        # [代码保持不变]...
        fname, _ = QFileDialog.getOpenFileName(self, '选择图像1', '', '图像文件 (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.img1 = Image.open(fname).convert('RGB')
            self.img1 = self.img1.resize((224, 224))
            self.img1_tensor = transforms.ToTensor()(self.img1).unsqueeze(0).to(self.device)
            self.update_transform()
            
    def load_image2(self):
        # [代码保持不变]...
        fname, _ = QFileDialog.getOpenFileName(self, '选择图像2', '', '图像文件 (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.img2 = Image.open(fname).convert('RGB')
            self.img2 = self.img2.resize((224, 224))
            self.img2_tensor = transforms.ToTensor()(self.img2).unsqueeze(0).to(self.device)
            self.update_transform()
    
    def reset_params(self):
        """重置所有参数为默认值"""
        self._slider_updating = True
        
        # 重置滑动条
        self.rotation_slider.setValue(0)
        self.cx_slider.setValue(0)
        self.cy_slider.setValue(0)
        self.tx_slider.setValue(0)
        self.ty_slider.setValue(0)
        
        # 重置输入框
        self.rotation_input.setText("0.0")
        self.cx_input.setText("0.0")
        self.cy_input.setText("0.0")
        self.tx_input.setText("0.0")
        self.ty_input.setText("0.0")
        
        self._slider_updating = False
        self.update_transform()
        
    def update_transform(self):
        """执行变换并更新显示"""
        # 如果图像已加载，执行变换
        if self.img1_tensor is not None:
            try:
                # 从输入框读取值（更精确）
                rotation_value = float(self.rotation_input.text())
                cx_value = float(self.cx_input.text())
                cy_value = float(self.cy_input.text())
                tx_value = float(self.tx_input.text())
                ty_value = float(self.ty_input.text())
                
                # 计算旋转矩阵参数
                angle_rad = rotation_value * np.pi / 180.0
                a = np.cos(angle_rad)
                b = -np.sin(angle_rad)
                c = np.sin(angle_rad)
                d = np.cos(angle_rad)
                
                # 创建变换参数tensor
                T = torch.tensor([[a, b, c, d, cx_value, cy_value, tx_value, ty_value]], 
                                dtype=torch.float32).to(self.device)
                
                # 执行变换
                transformed_img = self.forward_transfer(self.img1_tensor, T)
                
                # 更新图表
                self.ax1.clear()
                self.ax1.imshow(self.img1)
                self.ax1.set_title('原始图像1')
                self.ax1.axis('off')
                
                if self.img2_tensor is not None:
                    self.ax2.clear()
                    self.ax2.imshow(self.img2)
                    self.ax2.set_title('目标图像2')
                    self.ax2.axis('off')
                    
                    # 计算差值
                    diff_img = torch.abs(transformed_img - self.img2_tensor)
                    diff_np = diff_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    self.ax4.clear()
                    self.ax4.imshow(np.clip(diff_np, 0, 1))
                    self.ax4.set_title(f'差值图像 (均值: {np.mean(diff_np):.4f})')
                    self.ax4.axis('off')
                    
                # 显示变换后图像
                transformed_np = transformed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                self.ax3.clear()
                self.ax3.imshow(np.clip(transformed_np, 0, 1))
                self.ax3.set_title('变换后图像1')
                self.ax3.axis('off')
                
                self.canvas.draw()
            
            except (ValueError, TypeError) as e:
                print(f"变换参数错误: {e}")
    
    def forward_transfer(self, x, T):
        # [代码保持不变]...
        # 此函数逻辑保持不变
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
        # 先平移后旋转的逆变换 = 先逆旋转后逆平移
        
        # 1. 将坐标相对于旋转中心
        x_centered = grid_x - cx - tx
        y_centered = grid_y - cy - ty
        
        # 2. 应用旋转的逆变换
        x_unrotated = inv_a * x_centered + inv_b * y_centered
        y_unrotated = inv_c * x_centered + inv_d * y_centered
        
        # 3. 加回旋转中心
        x_after_rot = x_unrotated + cx
        y_after_rot = y_unrotated + cy
        
        # 4. 应用平移的逆变换
        x_in = x_after_rot 
        y_in = y_after_rot 
        
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

def main():
    app = QApplication(sys.argv)
    window = TransformVisualizer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()