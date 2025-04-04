import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QSlider, QLineEdit, 
                            QGridLayout, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
import torch
from PIL import Image
import torchvision.transforms as transforms
import time

class TransformVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.weight_map = None  # 存储权重图
        self.sigma = 0.5  # 高斯权重的sigma参数

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initUI()
        self.img1 = None
        self.img2 = None
        self.img1_tensor = None
        self.img2_tensor = None
        self._slider_updating = False  # 防止循环更新

        
    def initUI(self):
        # 确保必要的属性已被初始化
        if not hasattr(self, 'sigma'):
            self.sigma = 0.5
        if not hasattr(self, 'weight_map'):
            self.weight_map = None
        # 设置窗口标题和大小
        self.setWindowTitle('图像变换可视化工具')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 创建图表和控制区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 创建图表
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas)
        
        # 设置图表初始标题
        self.ax1.set_title('origin1')
        self.ax2.set_title('dst2')
        self.ax3.set_title('aff1')
        self.ax4.set_title('diff')
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')
        self.ax4.axis('off')
        
        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 图像加载控制
        load_layout = QVBoxLayout()
        self.load_img1_btn = QPushButton('加载图像1')
        self.load_img2_btn = QPushButton('加载图像2')
        load_layout.addWidget(self.load_img1_btn)
        load_layout.addWidget(self.load_img2_btn)
        
        # 连接信号和槽
        self.load_img1_btn.clicked.connect(lambda: self.load_image(1))
        self.load_img2_btn.clicked.connect(lambda: self.load_image(2))
        
        right_layout.addLayout(load_layout)
        right_layout.addSpacing(20)
        
        # 变换参数控制
        controls_layout = QGridLayout()
        
        # 旋转控制
        self.rotation_label = QLabel('旋转角度:')
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_input = QLineEdit('0.0')
        self.rotation_input.setValidator(QDoubleValidator(-180.0, 180.0, 5))
        self.rotation_input.setMaximumWidth(80)
        
        # 旋转中心控制
        self.cx_label = QLabel('旋转中心 X:')
        self.cx_slider = QSlider(Qt.Horizontal)
        self.cx_slider.setRange(-100, 100)
        self.cx_slider.setValue(0)
        self.cx_input = QLineEdit('0.0')
        self.cx_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        self.cx_input.setMaximumWidth(80)
        
        self.cy_label = QLabel('旋转中心 Y:')
        self.cy_slider = QSlider(Qt.Horizontal)
        self.cy_slider.setRange(-100, 100)
        self.cy_slider.setValue(0)
        self.cy_input = QLineEdit('0.0')
        self.cy_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        self.cy_input.setMaximumWidth(80)
        
        # 平移控制
        self.tx_label = QLabel('平移 X:')
        self.tx_slider = QSlider(Qt.Horizontal)
        self.tx_slider.setRange(-100, 100)
        self.tx_slider.setValue(0)
        self.tx_input = QLineEdit('0.0')
        self.tx_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        self.tx_input.setMaximumWidth(80)
        
        self.ty_label = QLabel('平移 Y:')
        self.ty_slider = QSlider(Qt.Horizontal)
        self.ty_slider.setRange(-100, 100)
        self.ty_slider.setValue(0)
        self.ty_input = QLineEdit('0.0')
        self.ty_input.setValidator(QDoubleValidator(-1.0, 1.0, 5))
        self.ty_input.setMaximumWidth(80)
        
        # 添加 sigma 控制滑块
        self.sigma_label = QLabel('高斯权重 Sigma:')
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(10, 200)  # 0.1 到 2.0
        self.sigma_slider.setValue(int(self.sigma * 100))
        self.sigma_input = QLineEdit(f"{self.sigma:.5f}")
        self.sigma_input.setValidator(QDoubleValidator(0.1, 2.0, 5))
        self.sigma_input.setMaximumWidth(80)
        
        # 添加到网格布局
        controls_layout.addWidget(self.rotation_label, 0, 0)
        controls_layout.addWidget(self.rotation_slider, 0, 1)
        controls_layout.addWidget(self.rotation_input, 0, 2)
        
        controls_layout.addWidget(self.cx_label, 1, 0)
        controls_layout.addWidget(self.cx_slider, 1, 1)
        controls_layout.addWidget(self.cx_input, 1, 2)
        
        controls_layout.addWidget(self.cy_label, 2, 0)
        controls_layout.addWidget(self.cy_slider, 2, 1)
        controls_layout.addWidget(self.cy_input, 2, 2)
        
        controls_layout.addWidget(self.tx_label, 3, 0)
        controls_layout.addWidget(self.tx_slider, 3, 1)
        controls_layout.addWidget(self.tx_input, 3, 2)
        
        controls_layout.addWidget(self.ty_label, 4, 0)
        controls_layout.addWidget(self.ty_slider, 4, 1)
        controls_layout.addWidget(self.ty_input, 4, 2)
        
        controls_layout.addWidget(self.sigma_label, 5, 0)
        controls_layout.addWidget(self.sigma_slider, 5, 1)
        controls_layout.addWidget(self.sigma_input, 5, 2)
        
        right_layout.addLayout(controls_layout)
        
        # 连接信号和槽
        self.rotation_slider.valueChanged.connect(self.rotation_slider_changed)
        self.rotation_input.editingFinished.connect(self.rotation_input_changed)
        
        self.cx_slider.valueChanged.connect(self.cx_slider_changed)
        self.cx_input.editingFinished.connect(self.cx_input_changed)
        
        self.cy_slider.valueChanged.connect(self.cy_slider_changed)
        self.cy_input.editingFinished.connect(self.cy_input_changed)
        
        self.tx_slider.valueChanged.connect(self.tx_slider_changed)
        self.tx_input.editingFinished.connect(self.tx_input_changed)
        
        self.ty_slider.valueChanged.connect(self.ty_slider_changed)
        self.ty_input.editingFinished.connect(self.ty_input_changed)
        
        self.sigma_slider.valueChanged.connect(self.sigma_slider_changed)
        self.sigma_input.editingFinished.connect(self.sigma_input_changed)
        
        # 添加按钮来应用变换
        self.apply_btn = QPushButton('应用变换')
        self.apply_btn.clicked.connect(self.update_transform)
        right_layout.addWidget(self.apply_btn)
        
        # 添加优化按钮(可选)
        self.optimize_btn = QPushButton('优化变换参数')
        self.optimize_btn.clicked.connect(self.optimize_parameters)
        right_layout.addWidget(self.optimize_btn)
        self.optimize_btn.setEnabled(False)  # 初始时禁用优化按钮
        
        # 设置右侧部件的最小宽度
        right_widget.setMinimumWidth(300)
        
        # 添加到主布局
        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(right_widget, 1)
        
        # 设置中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def load_image(self, img_num):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(self, f'选择图像{img_num}', '', '图像文件 (*.png *.jpg *.jpeg *.bmp)')
        if not file_path:
            return
        
        try:
            # 加载并预处理图像
            img = Image.open(file_path).convert('RGB')
            img = img.resize((224, 224))  # 调整大小
            
            # 转换为张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
               
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
            
            # 存储图像
            if img_num == 1:
                self.img1 = np.array(img) / 255.0
                self.img1_tensor = img_tensor
                self.ax1.clear()
                self.ax1.imshow(self.img1)
                self.ax1.set_title('origin1')
                self.ax1.axis('off')
            else:
                self.img2 = np.array(img) / 255.0
                self.img2_tensor = img_tensor
                self.ax2.clear()
                self.ax2.imshow(self.img2)
                self.ax2.set_title('dst2')
                self.ax2.axis('off')
            
            self.canvas.draw()
            
            # 如果两张图都已加载，启用优化按钮
            if self.img1_tensor is not None and self.img2_tensor is not None:
                self.optimize_btn.setEnabled(True)
                
        except Exception as e:
            print(f"加载图像出错: {e}")
    
    def rotation_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        rotation_value = self.rotation_slider.value()
        self.rotation_input.setText(f"{rotation_value:.5f}")
        self.update_transform()
        self._slider_updating = False
    
    def rotation_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            rotation_value = float(self.rotation_input.text())
            rotation_value = max(-180, min(180, rotation_value))
            self.rotation_slider.setValue(int(rotation_value))
            self.update_transform()
        except ValueError:
            self.rotation_input.setText(f"{self.rotation_slider.value():.5f}")
        
        self._slider_updating = False
    
    def cx_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        cx_value = self.cx_slider.value() / 100.0
        self.cx_input.setText(f"{cx_value:.5f}")
        self.update_transform()
        self._slider_updating = False
    
    def cx_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            cx_value = float(self.cx_input.text())
            cx_value = max(-1, min(1, cx_value))
            self.cx_slider.setValue(int(cx_value * 100))
            self.update_transform()
        except ValueError:
            self.cx_input.setText(f"{self.cx_slider.value() / 100.0:.5f}")
        
        self._slider_updating = False
    
    def cy_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        cy_value = self.cy_slider.value() / 100.0
        self.cy_input.setText(f"{cy_value:.5f}")
        self.update_transform()
        self._slider_updating = False
    
    def cy_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            cy_value = float(self.cy_input.text())
            cy_value = max(-1, min(1, cy_value))
            self.cy_slider.setValue(int(cy_value * 100))
            self.update_transform()
        except ValueError:
            self.cy_input.setText(f"{self.cy_slider.value() / 100.0:.5f}")
        
        self._slider_updating = False
    
    def tx_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        tx_value = self.tx_slider.value() / 100.0
        self.tx_input.setText(f"{tx_value:.5f}")
        self.update_transform()
        self._slider_updating = False
    
    def tx_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            tx_value = float(self.tx_input.text())
            tx_value = max(-1, min(1, tx_value))
            self.tx_slider.setValue(int(tx_value * 100))
            self.update_transform()
        except ValueError:
            self.tx_input.setText(f"{self.tx_slider.value() / 100.0:.5f}")
        
        self._slider_updating = False
    
    def ty_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        ty_value = self.ty_slider.value() / 100.0
        self.ty_input.setText(f"{ty_value:.5f}")
        self.update_transform()
        self._slider_updating = False
    
    def ty_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            ty_value = float(self.ty_input.text())
            ty_value = max(-1, min(1, ty_value))
            self.ty_slider.setValue(int(ty_value * 100))
            self.update_transform()
        except ValueError:
            self.ty_input.setText(f"{self.ty_slider.value() / 100.0:.5f}")
        
        self._slider_updating = False
    
    def sigma_slider_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        sigma_value = self.sigma_slider.value() / 100.0
        self.sigma = sigma_value
        self.sigma_input.setText(f"{sigma_value:.5f}")
        self.weight_map = None  # 重置权重图
        self.update_transform()
        self._slider_updating = False
    
    def sigma_input_changed(self):
        if self._slider_updating:
            return
        
        self._slider_updating = True
        try:
            sigma_value = float(self.sigma_input.text())
            sigma_value = max(0.1, min(2.0, sigma_value))
            self.sigma = sigma_value
            self.sigma_slider.setValue(int(sigma_value * 100))
            self.weight_map = None  # 重置权重图
            self.update_transform()
        except ValueError:
            self.sigma_input.setText(f"{self.sigma:.5f}")
        
        self._slider_updating = False

    def calculate_loss(self, img1, img2_trans):
        """
        计算两个图像之间的加权MSE损失，权重从中心到边缘逐渐减小
        
        Args:
            img1: 原始图像，形状为[1, C, H, W]
            img2_trans: 变换后图像，形状为[1, C, H, W]
            
        Returns:
            float: 加权MSE损失值
        """
        if not hasattr(self, 'sigma'):
            self.sigma = 0.5
        if img1 is None or img2_trans is None:
            return 0.0
            
        B, C, H, W = img1.shape
        device = img1.device
        
        # 预计算网格坐标 (只需计算一次，存为类变量)
        if self.weight_map is None or self.weight_map.shape[2:] != (H, W) or self.weight_map.device != device:
            # 创建高斯权重图，中心权重高，边缘权重低
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            
            # 计算每个像素到图像中心的距离
            dist_squared = x_grid.pow(2) + y_grid.pow(2)
            
            # 使用高斯函数生成权重图
            weights = torch.exp(-dist_squared / (2 * self.sigma**2))
            
            # 归一化权重，使权重总和为像素数量
            weights = weights * (H * W) / weights.sum()
            
            # 保存为类变量，避免重复计算
            self.weight_map = weights.unsqueeze(0).unsqueeze(0)
        
        # 扩展维度以匹配当前批次大小
        weights = self.weight_map.expand(B, C, H, W)
        
        # 计算MSE损失并应用权重
        squared_diff = torch.nn.functional.mse_loss(img1, img2_trans, reduction='none')
        
        # 加权平均
        loss = (squared_diff * weights).sum() / (B * C * H * W)
        
        return loss.item()
    
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
                
                # 转换为弧度
                rotation_rad = rotation_value * np.pi / 180.0
                
                # 创建5参数向量 [theta, cx, cy, tx, ty]
                params = torch.tensor([[rotation_rad, cx_value, cy_value, tx_value, ty_value]], 
                                   dtype=torch.float32).to(self.device)
                
                # 执行变换
                transformed_img = self.forward_transfer(self.img1_tensor, params)
                
                # 计算损失
                loss_value = 0
                if self.img2_tensor is not None:
                    loss_value = self.calculate_loss(self.img2_tensor, transformed_img)
                
                # 更新图表
                self.ax1.clear()
                self.ax1.imshow(self.img1)
                self.ax1.set_title('origin1')
                self.ax1.axis('off')
                
                # 显示变换后图像
                transformed_np = transformed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                self.ax3.clear()
                self.ax3.imshow(np.clip(transformed_np, 0, 1))
                self.ax3.set_title(f'trans1')
                self.ax3.axis('off')
                
                if self.img2_tensor is not None:
                    self.ax2.clear()
                    self.ax2.imshow(self.img2)
                    self.ax2.set_title('dst2')
                    self.ax2.axis('off')
                    
                    # 计算差值图
                    diff_img = torch.abs(transformed_img - self.img2_tensor)
                    diff_np = diff_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    self.ax4.clear()
                    self.ax4.imshow(np.clip(diff_np, 0, 1))
                    self.ax4.set_title(f'diff (Loss: {loss_value:.6f})')
                    self.ax4.axis('off')
                
                self.canvas.draw()
            
            except (ValueError, TypeError) as e:
                print(f"变换参数错误: {e}")
                
    def optimize_parameters(self):
        """优化变换参数"""
        if self.img1_tensor is None or self.img2_tensor is None:
            return
        
        print("正在优化变换参数...")
        
        # 从输入框读取当前值作为初始值
        init_rotation = float(self.rotation_input.text()) * np.pi / 180.0
        init_cx = float(self.cx_input.text())
        init_cy = float(self.cy_input.text())
        init_tx = float(self.tx_input.text())
        init_ty = float(self.ty_input.text())
        
        # 初始参数
        params = torch.tensor([[init_rotation, init_cx, init_cy, init_tx, init_ty]], 
                            requires_grad=True, device=self.device)
        
        # 创建优化器
        optimizer = torch.optim.Adam([params], lr=0.01)
        
        # 存储初始损失
        with torch.no_grad():
            transformed_img = self.forward_transfer(self.img1_tensor, params)
            initial_loss = self.calculate_loss(self.img2_tensor, transformed_img)
        
        print(f"初始损失: {initial_loss:.6f}")
        
        # 优化循环
        n_iterations = 100
        best_loss = initial_loss
        best_params = params.clone().detach()
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 前向传播
            transformed_img = self.forward_transfer(self.img1_tensor, params)
            
            # 计算损失
            loss = self.calculate_loss(self.img2_tensor, transformed_img)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 保存最佳参数
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.clone().detach()
            
            if (i+1) % 10 == 0:
                print(f"迭代 {i+1}/{n_iterations}, 损失: {current_loss:.6f}")
        
        print(f"优化完成! 最终损失: {best_loss:.6f}")
        
        # 更新UI
        with torch.no_grad():
            best_rotation = best_params[0, 0].item() * 180.0 / np.pi
            best_cx = best_params[0, 1].item()
            best_cy = best_params[0, 2].item()
            best_tx = best_params[0, 3].item()
            best_ty = best_params[0, 4].item()
            
            self._slider_updating = True
            
            self.rotation_slider.setValue(int(best_rotation))
            self.rotation_input.setText(f"{best_rotation:.5f}")
            
            self.cx_slider.setValue(int(best_cx * 100))
            self.cx_input.setText(f"{best_cx:.5f}")
            
            self.cy_slider.setValue(int(best_cy * 100))
            self.cy_input.setText(f"{best_cy:.5f}")
            
            self.tx_slider.setValue(int(best_tx * 100))
            self.tx_input.setText(f"{best_tx:.5f}")
            
            self.ty_slider.setValue(int(best_ty * 100))
            self.ty_input.setText(f"{best_ty:.5f}")
            
            self._slider_updating = False
            
            self.update_transform()
            
    def forward_transfer(self, x, params):
        """
        使用5参数[theta,cx,cy,tx,ty]应用仿射变换到输入图像
        
        Args:
            x (Tensor): 输入数据，[B, C, H, W]
            params (Tensor): 变换参数，[B, 5] (theta, cx, cy, tx, ty)
                
        Returns:
            Tensor: 变换后的图像，[B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 提取参数
        theta = params[:, 0]  # 旋转角度
        cx = params[:, 1]  # 旋转中心x
        cy = params[:, 2]  # 旋转中心y
        tx = params[:, 3]  # x方向平移
        ty = params[:, 4]  # y方向平移
        
        # 构建旋转矩阵
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 旋转矩阵元素
        a = cos_theta
        b_ = -sin_theta
        c = sin_theta
        d = cos_theta
        
        # 创建归一化网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
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
        
        # 将参数调整为[B,1,1]形状，方便广播
        inv_a = inv_a.view(B, 1, 1)
        inv_b = inv_b.view(B, 1, 1)
        inv_c = inv_c.view(B, 1, 1)
        inv_d = inv_d.view(B, 1, 1)
        cx = cx.view(B, 1, 1)
        cy = cy.view(B, 1, 1)
        tx = tx.view(B, 1, 1)
        ty = ty.view(B, 1, 1)
        
        # 扩展网格坐标到批次维度
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        
        # 逆向映射坐标计算（从输出找输入）:
        # 先平移后旋转的逆变换 = 先逆旋转后逆平移
        
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TransformVisualizer()
    window.show()
    sys.exit(app.exec_())