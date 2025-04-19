import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2

class TouchVisionRegistrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Touch-Vision Image Registration Tool")
        self.root.geometry("1200x800")
        
        # 设置Tkinter风格主题
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():  # clam主题通常兼容性更好
            self.style.theme_use('clam')
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.touch_img = None
        self.vision_img = None
        self.touch_tensor = None
        self.vision_tensor = None
        self.overlay_alpha = 0.5  # 叠加透明度
        
        # 矩形框选择
        self.rect_touch = None      # 触觉图像上的选择框
        self.rect_vision = None     # 视觉图像上的选择框
        self.rs_touch = None        # 触觉图像的RectangleSelector对象
        self.rs_vision = None       # 视觉图像的RectangleSelector对象
        self.active_selector = None # 当前激活的选择器
        
        self.create_widgets()
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左右分割布局
        self.paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧显示区域
        left_frame = ttk.Frame(self.paned, width=800)
        self.paned.add(left_frame, weight=3)
        
        # 右侧控制面板
        right_frame = ttk.Frame(self.paned, width=300)
        self.paned.add(right_frame, weight=1)
        
        # 创建matplotlib图形
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # 设置图表初始标题
        self.ax1.set_title('Tactile Image')
        self.ax2.set_title('Visual Image')
        self.ax3.set_title('Transformed Tactile')
        self.ax4.set_title('Overlay Result')
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')
        self.ax4.axis('off')
        
        # 在tkinter窗口中放置matplotlib图形
        canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas
        
        # 图像加载控制
        load_frame = ttk.LabelFrame(right_frame, text="Image Loading")
        load_frame.pack(fill=tk.X, pady=5, padx=5)
        
        load_touch_btn = ttk.Button(load_frame, text="Load Tactile Image", command=lambda: self.load_image(1))
        load_touch_btn.pack(fill=tk.X, pady=5, padx=5)
        
        load_vision_btn = ttk.Button(load_frame, text="Load Visual Image", command=lambda: self.load_image(2))
        load_vision_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # 矩形框控制
        rect_frame = ttk.LabelFrame(right_frame, text="Rectangle Selection")
        rect_frame.pack(fill=tk.X, pady=5, padx=5)
        
        rect_touch_btn = ttk.Button(rect_frame, text="Select on Tactile Image", 
                                   command=lambda: self.activate_rectangle_selector(1))
        rect_touch_btn.pack(fill=tk.X, pady=5, padx=5)
        
        rect_vision_btn = ttk.Button(rect_frame, text="Select on Visual Image", 
                                    command=lambda: self.activate_rectangle_selector(2))
        rect_vision_btn.pack(fill=tk.X, pady=5, padx=5)
        
        clear_rect_btn = ttk.Button(rect_frame, text="Clear Rectangles", 
                                   command=self.clear_rectangles)
        clear_rect_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # 创建控制参数框架
        controls_frame = ttk.LabelFrame(right_frame, text="Transform Parameters")
        controls_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # X方向平移控制
        x_frame = ttk.Frame(controls_frame)
        x_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(x_frame, text="X Translation:").pack(side=tk.LEFT, padx=5)
        self.tx_var = tk.DoubleVar(value=0.0)
        tx_scale = ttk.Scale(x_frame, from_=-1.0, to=1.0, variable=self.tx_var, 
                           command=lambda x: self.update_scale_label('tx', x))
        tx_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.tx_label = ttk.Label(x_frame, text="0.00")
        self.tx_label.pack(side=tk.RIGHT, padx=5)
        
        # Y方向平移控制
        y_frame = ttk.Frame(controls_frame)
        y_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(y_frame, text="Y Translation:").pack(side=tk.LEFT, padx=5)
        self.ty_var = tk.DoubleVar(value=0.0)
        ty_scale = ttk.Scale(y_frame, from_=-1.0, to=1.0, variable=self.ty_var,
                           command=lambda y: self.update_scale_label('ty', y))
        ty_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.ty_label = ttk.Label(y_frame, text="0.00")
        self.ty_label.pack(side=tk.RIGHT, padx=5)
        
        # 等比例缩放控制
        s_frame = ttk.Frame(controls_frame)
        s_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(s_frame, text="Scale:").pack(side=tk.LEFT, padx=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_scale = ttk.Scale(s_frame, from_=0.5, to=2.0, variable=self.scale_var,
                              command=lambda s: self.update_scale_label('scale', s))
        scale_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.scale_label = ttk.Label(s_frame, text="1.00")
        self.scale_label.pack(side=tk.RIGHT, padx=5)
        
        # 透明度控制
        a_frame = ttk.Frame(controls_frame)
        a_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(a_frame, text="Transparency:").pack(side=tk.LEFT, padx=5)
        self.alpha_var = tk.DoubleVar(value=self.overlay_alpha)
        alpha_scale = ttk.Scale(a_frame, from_=0.0, to=1.0, variable=self.alpha_var,
                              command=lambda a: self.update_scale_label('alpha', a))
        alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.alpha_label = ttk.Label(a_frame, text="0.50")
        self.alpha_label.pack(side=tk.RIGHT, padx=5)
        
        # 显示边缘选项
        edge_frame = ttk.Frame(controls_frame)
        edge_frame.pack(fill=tk.X, pady=5, padx=10)
        self.edge_var = tk.BooleanVar(value=False)
        edge_check = ttk.Checkbutton(edge_frame, text="Show Edges", 
                                   variable=self.edge_var,
                                   command=self.update_transform)
        edge_check.pack(fill=tk.X, pady=5)
        
        # 按钮控制
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # 自动对齐框按钮
        align_rect_btn = ttk.Button(buttons_frame, text="Align Rectangles", 
                                   command=self.align_rectangles)
        align_rect_btn.pack(fill=tk.X, pady=5)
        
        apply_btn = ttk.Button(buttons_frame, text="Apply Transform", command=self.update_transform)
        apply_btn.pack(fill=tk.X, pady=5)
        
        save_btn = ttk.Button(buttons_frame, text="Save Results", command=self.save_results)
        save_btn.pack(fill=tk.X, pady=5)
        
        # 设置默认值标签
        self.update_scale_label('tx', 0.0)
        self.update_scale_label('ty', 0.0)
        self.update_scale_label('scale', 1.0)
        self.update_scale_label('alpha', 0.5)
    
    def update_scale_label(self, scale_type, value):
        """更新滑动条旁边的标签显示"""
        try:
            value = float(value)
            if scale_type == 'tx':
                self.tx_label.config(text=f"{value:.2f}")
            elif scale_type == 'ty':
                self.ty_label.config(text=f"{value:.2f}")
            elif scale_type == 'scale':
                self.scale_label.config(text=f"{value:.2f}")
            elif scale_type == 'alpha':
                self.alpha_label.config(text=f"{value:.2f}")
                self.overlay_alpha = value
            
            # 更新变换
            self.update_transform()
        except:
            pass
    
    def load_image(self, img_num):
        """加载图像"""
        file_path = filedialog.askopenfilename(title='Select Image', 
                                               filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        
        try:
            # 加载并预处理图像
            img = Image.open(file_path).convert('RGB')
            
            # 转换为张量
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
            
            # 存储图像
            if img_num == 1:
                self.touch_img = np.array(transforms.Resize((224, 224))(img)) / 255.0
                self.touch_tensor = img_tensor
                self.ax1.clear()
                self.ax1.imshow(self.touch_img)
                self.ax1.set_title('Tactile Image')
                self.ax1.axis('off')
                
                # 重新创建矩形选择器
                if self.rs_touch:
                    self.rs_touch.set_active(False)
                self.rs_touch = RectangleSelector(
                    self.ax1, self.on_rectangle_select,
                    useblit=True,
                    button=[1],
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='red', edgecolor='red', alpha=0.5, fill=True)
                )
                self.rs_touch.set_active(False)
                
            else:
                self.vision_img = np.array(transforms.Resize((224, 224))(img)) / 255.0
                self.vision_tensor = img_tensor
                self.ax2.clear()
                self.ax2.imshow(self.vision_img)
                self.ax2.set_title('Visual Image')
                self.ax2.axis('off')
                
                # 重新创建矩形选择器
                if self.rs_vision:
                    self.rs_vision.set_active(False)
                self.rs_vision = RectangleSelector(
                    self.ax2, self.on_rectangle_select,
                    useblit=True,
                    button=[1],
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='red', edgecolor='red', alpha=0.5, fill=True)
                )
                self.rs_vision.set_active(False)
            
            self.canvas.draw()
            
            # 如果两张图都已加载，自动应用变换
            if self.touch_tensor is not None and self.vision_tensor is not None:
                self.update_transform()
                
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def activate_rectangle_selector(self, img_num):
        """激活矩形选择器"""
        if img_num == 1:
            if self.touch_img is None:
                messagebox.showinfo("Info", "Please load tactile image first")
                return
            
            # 停用其他选择器
            if self.rs_vision:
                self.rs_vision.set_active(False)
            
            # 激活触觉图像选择器
            if self.rs_touch:
                self.rs_touch.set_active(True)
                self.active_selector = self.rs_touch
                messagebox.showinfo("Info", "Draw a rectangle on the tactile image")
        else:
            if self.vision_img is None:
                messagebox.showinfo("Info", "Please load visual image first")
                return
            
            # 停用其他选择器
            if self.rs_touch:
                self.rs_touch.set_active(False)
            
            # 激活视觉图像选择器
            if self.rs_vision:
                self.rs_vision.set_active(True)
                self.active_selector = self.rs_vision
                messagebox.showinfo("Info", "Draw a rectangle on the visual image")
    
    def on_rectangle_select(self, eclick, erelease):
        """矩形选择事件处理"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # 确保x1,y1是左上角，x2,y2是右下角
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        rect = (x1, y1, x2, y2)  # 存储为(左,上,右,下)格式
        
        if self.active_selector == self.rs_touch:
            self.rect_touch = rect
            print(f"Selected rectangle on tactile image: {rect}")
        elif self.active_selector == self.rs_vision:
            self.rect_vision = rect
            print(f"Selected rectangle on visual image: {rect}")
            
        self.canvas.draw()
    
    def clear_rectangles(self):
        """清除所有矩形选择"""
        if self.rs_touch:
            self.rs_touch.set_active(False)
            if hasattr(self.rs_touch, 'to_draw'):
                self.rs_touch.to_draw.set_visible(False)
        
        if self.rs_vision:
            self.rs_vision.set_active(False)
            if hasattr(self.rs_vision, 'to_draw'):
                self.rs_vision.to_draw.set_visible(False)
            
        self.rect_touch = None
        self.rect_vision = None
        self.active_selector = None
        self.canvas.draw()
    
    def align_rectangles(self):
        """根据选择的矩形框自动计算对齐参数"""
        if not self.rect_touch or not self.rect_vision:
            messagebox.showinfo("Info", "Please select rectangles on both images first")
            return
        
        # 提取矩形的尺寸
        x1_t, y1_t, x2_t, y2_t = self.rect_touch  # 触觉图像上的矩形
        x1_v, y1_v, x2_v, y2_v = self.rect_vision  # 视觉图像上的矩形
        
        # 计算矩形中心点
        center_t = ((x1_t + x2_t) / 2, (y1_t + y2_t) / 2)
        center_v = ((x1_v + x2_v) / 2, (y1_v + y2_v) / 2)
        
        # 计算矩形尺寸
        width_t = x2_t - x1_t
        height_t = y2_t - y1_t
        width_v = x2_v - x1_v
        height_v = y2_v - y1_v
        
        # 计算缩放因子 (取宽高的平均缩放比)
        scale_factor = ((width_v / width_t) + (height_v / height_t)) / 2 if width_t > 0 and height_t > 0 else 1.0
        
        # 图像坐标系范围是[0,1]，而归一化坐标系范围是[-1,1]
        # 计算在[-1,1]范围内的归一化平移量
        tx = (center_v[0] - center_t[0] * scale_factor) / 112  # 除以图像半宽
        ty = (center_v[1] - center_t[1] * scale_factor) / 112  # 除以图像半高
        
        # 限制在合理范围内
        scale_factor = max(0.5, min(2.0, scale_factor))
        tx = max(-1.0, min(1.0, tx))
        ty = max(-1.0, min(1.0, ty))
        
        # 更新UI控件
        self.scale_var.set(scale_factor)
        self.tx_var.set(tx)
        self.ty_var.set(ty)
        
        # 更新标签
        self.scale_label.config(text=f"{scale_factor:.2f}")
        self.tx_label.config(text=f"{tx:.2f}")
        self.ty_label.config(text=f"{ty:.2f}")
        
        # 应用变换
        self.update_transform()
        
        messagebox.showinfo("Info", f"Aligned with Scale={scale_factor:.2f}, X={tx:.2f}, Y={ty:.2f}")
    
    def update_transform(self):
        """执行变换并更新显示"""
        # 如果触觉图像已加载，执行变换
        if self.touch_tensor is not None:
            try:
                # 获取参数值
                tx_value = self.tx_var.get()
                ty_value = self.ty_var.get()
                scale_value = self.scale_var.get()
                
                # 创建3参数向量 [scale, tx, ty]
                params = torch.tensor([[scale_value, tx_value, ty_value]], 
                                     dtype=torch.float32).to(self.device)
                
                # 执行变换
                transformed_img = self.forward_transfer(self.touch_tensor, params)
                
                # 更新触觉图像显示
                self.ax1.clear()
                self.ax1.imshow(self.touch_img)
                self.ax1.set_title('Tactile Image')
                self.ax1.axis('off')
                
                # 如果有矩形框，重新绘制
                if self.rect_touch:
                    x1, y1, x2, y2 = self.rect_touch
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      fill=False, edgecolor='red', linewidth=2)
                    self.ax1.add_patch(rect)
                
                # 显示变换后的触觉图像
                transformed_np = transformed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                transformed_np = np.clip(transformed_np, 0, 1)
                self.ax3.clear()
                self.ax3.imshow(transformed_np)
                self.ax3.set_title('Transformed Tactile')
                self.ax3.axis('off')
                
                # 如果视觉图像也已加载，创建叠加图像
                if self.vision_tensor is not None:
                    self.ax2.clear()
                    self.ax2.imshow(self.vision_img)
                    self.ax2.set_title('Visual Image')
                    self.ax2.axis('off')
                    
                    # 如果有矩形框，重新绘制
                    if self.rect_vision:
                        x1, y1, x2, y2 = self.rect_vision
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          fill=False, edgecolor='red', linewidth=2)
                        self.ax2.add_patch(rect)
                    
                    # 显示叠加结果
                    self.ax4.clear()
                    
                    # 检查是否显示边缘
                    if self.edge_var.get():
                        # 转换为灰度图并检测边缘
                        gray_touch = cv2.cvtColor((transformed_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray_touch, 50, 150)
                        
                        # 创建彩色叠加图
                        vision_rgb = self.vision_img.copy()
                        
                        # 在边缘位置绘制绿色
                        edge_overlay = vision_rgb.copy()
                        edge_overlay[edges > 0] = [0, 1, 0]  # 绿色边缘
                        
                        self.ax4.imshow(edge_overlay)
                        self.ax4.set_title('Edge Overlay')
                    else:
                        # 正常的混合叠加
                        alpha = self.overlay_alpha
                        overlay_img = (1-alpha) * self.vision_img + alpha * transformed_np
                        self.ax4.imshow(overlay_img)
                        self.ax4.set_title(f'Overlay Result (α={alpha:.2f})')
                    
                    self.ax4.axis('off')
                
                # 恢复矩形选择器的激活状态
                if self.rs_touch:
                    self.rs_touch.set_active(self.active_selector == self.rs_touch)
                if self.rs_vision:
                    self.rs_vision.set_active(self.active_selector == self.rs_vision)
                
                self.canvas.draw()
            
            except (ValueError, TypeError) as e:
                print(f"Transform parameter error: {e}")
    
    def save_results(self):
        """保存当前显示的图像结果"""
        if not (self.touch_tensor is not None and self.vision_tensor is not None):
            print("Please load both images first")
            return
            
        file_path = filedialog.asksaveasfilename(
            title='Save Results',
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if not file_path:
            return
            
        try:
            # 临时停用矩形选择器用于保存
            touch_active = False
            vision_active = False
            
            if self.rs_touch:
                touch_active = self.rs_touch.active
                self.rs_touch.set_active(False)
            
            if self.rs_vision:
                vision_active = self.rs_vision.active
                self.rs_vision.set_active(False)
            
            # 获取当前的图像数据
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {file_path}")
            
            # 恢复矩形选择器状态
            if self.rs_touch and touch_active:
                self.rs_touch.set_active(True)
            if self.rs_vision and vision_active:
                self.rs_vision.set_active(True)
            
        except Exception as e:
            print(f"Error saving results: {e}")
            
    def forward_transfer(self, x, params):
        """
        使用3参数[scale, tx, ty]应用仿射变换到输入图像
        
        Args:
            x (Tensor): 输入数据，[B, C, H, W]
            params (Tensor): 变换参数，[B, 3] (scale, tx, ty)
                
        Returns:
            Tensor: 变换后的图像，[B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 提取参数
        scale = params[:, 0]  # 缩放比例
        tx = params[:, 1]     # x方向平移
        ty = params[:, 2]     # y方向平移
        
        # 创建归一化网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 扩展网格坐标到批次维度
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        
        # 参数调整为[B,1,1]形状，方便广播
        scale = scale.view(B, 1, 1)
        tx = tx.view(B, 1, 1)
        ty = ty.view(B, 1, 1)
        
        # 逆向映射坐标计算
        # 缩放的逆变换是除以缩放比例
        # 平移的逆变换是减去平移量
        x_in = (grid_x - tx) / scale
        y_in = (grid_y - ty) / scale
        
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
    root = tk.Tk()
    root.update()  # 在创建应用前先更新一次root窗口
    app = TouchVisionRegistrationTool(root)
    root.mainloop()