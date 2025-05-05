import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import argparse
from functools import partial

def rgb_to_hsv(img):
    """将RGB图像转换为HSV"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(img):
    """将HSV图像转换为RGB"""
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def rgb_to_lab(img):
    """将RGB图像转换为LAB"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def lab_to_rgb(img):
    """将LAB图像转换为RGB"""
    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    对图像应用双边滤波器
    
    参数:
        img: 输入RGB图像
        d: 滤波器直径，表示参与计算的像素邻域大小
        sigma_color: 颜色空间的标准差
        sigma_space: 坐标空间的标准差
    
    返回:
        滤波后的图像
    """
    # OpenCV的双边滤波需要BGR格式
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
    # 转换回RGB
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

def align_channel_statistics(source, target, channel_index, match_mean=True, match_std=True):
    """
    将源图像的指定通道统计特性对齐到目标图像
    
    参数:
        source: 源图像
        target: 目标图像
        channel_index: 要对齐的通道索引
        match_mean: 是否匹配均值
        match_std: 是否匹配标准差
    
    返回:
        对齐后的源图像
    """
    # 创建输出图像的副本
    aligned = source.copy().astype(np.float32)
    
    # 计算源图像和目标图像指定通道的均值和标准差
    source_mean = np.mean(source[:, :, channel_index])
    source_std = np.std(source[:, :, channel_index])
    
    target_mean = np.mean(target[:, :, channel_index])
    target_std = np.std(target[:, :, channel_index])
    
    # 仅对指定通道进行调整
    if match_std and target_std > 0 and source_std > 0:
        # 标准化并重新缩放
        factor = target_std / source_std if match_std else 1.0
        aligned[:, :, channel_index] = (aligned[:, :, channel_index] - source_mean) * factor
        
        if match_mean:
            aligned[:, :, channel_index] += target_mean
        else:
            aligned[:, :, channel_index] += source_mean
    elif match_mean:
        # 只匹配均值
        aligned[:, :, channel_index] += (target_mean - source_mean)
    
    # 确保值在合适的范围内
    aligned = np.clip(aligned, 0, 255)
    
    return aligned.astype(np.uint8)

def get_channel_stats(image, colorspace="RGB"):
    """
    获取图像各通道的统计信息
    
    参数:
        image: 原始RGB图像
        colorspace: 色彩空间 ('RGB', 'HSV', 'LAB')
    
    返回:
        各通道的均值和标准差
    """
    if colorspace == "RGB":
        img = image
        channel_names = ["R", "G", "B"]
    elif colorspace == "HSV":
        img = rgb_to_hsv(image)
        channel_names = ["H", "S", "V"]
    elif colorspace == "LAB":
        img = rgb_to_lab(image)
        channel_names = ["L", "A", "B"]
    else:
        raise ValueError("不支持的色彩空间")
    
    stats = []
    for i in range(3):
        mean = np.mean(img[:, :, i])
        std = np.std(img[:, :, i])
        stats.append((mean, std))
    
    return dict(zip(channel_names, stats))

def plot_channel_histograms(img1, img2, img1_aligned=None, img1_filtered=None, colorspace="RGB", title="Channel Histograms"):
    """
    绘制图像通道的直方图
    
    参数:
        img1: 第一张RGB图像
        img2: 第二张RGB图像
        img1_aligned: 对齐后的第一张图像
        img1_filtered: 滤波后的图像
        colorspace: 色彩空间 ('RGB', 'HSV', 'LAB')
        title: 图表标题
    """
    # 转换色彩空间
    if colorspace == "RGB":
        img1_conv = img1
        img2_conv = img2
        img1_aligned_conv = img1_aligned if img1_aligned is not None else None
        img1_filtered_conv = img1_filtered if img1_filtered is not None else None
        channel_names = ["Red", "Green", "Blue"]
    elif colorspace == "HSV":
        img1_conv = rgb_to_hsv(img1)
        img2_conv = rgb_to_hsv(img2)
        img1_aligned_conv = rgb_to_hsv(img1_aligned) if img1_aligned is not None else None
        img1_filtered_conv = rgb_to_hsv(img1_filtered) if img1_filtered is not None else None
        channel_names = ["Hue", "Saturation", "Value"]
    elif colorspace == "LAB":
        img1_conv = rgb_to_lab(img1)
        img2_conv = rgb_to_lab(img2)
        img1_aligned_conv = rgb_to_lab(img1_aligned) if img1_aligned is not None else None
        img1_filtered_conv = rgb_to_lab(img1_filtered) if img1_filtered is not None else None
        channel_names = ["Lightness", "A (Green-Red)", "B (Blue-Yellow)"]
    
    # 设置图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # 为每个通道生成直方图
    for i, (channel_name, ax) in enumerate(zip(channel_names, axes)):
        # 确定当前通道的范围
        if colorspace == "HSV" and i == 0:
            range_vals = (0, 180)  # OpenCV中H通道范围为0-180
        else:
            range_vals = (0, 255)
        
        # 计算直方图
        hist_img1, _ = np.histogram(img1_conv[:,:,i].flatten(), bins=50, range=range_vals)
        hist_img2, _ = np.histogram(img2_conv[:,:,i].flatten(), bins=50, range=range_vals)
        
        # 绘制直方图
        ax.plot(hist_img1, label=f"Image 1 ({channel_names[i]})", color='blue', alpha=0.7)
        ax.plot(hist_img2, label=f"Image 2 ({channel_names[i]})", color='red', alpha=0.7)
        
        if img1_aligned_conv is not None:
            hist_aligned, _ = np.histogram(img1_aligned_conv[:,:,i].flatten(), bins=50, range=range_vals)
            ax.plot(hist_aligned, label=f"Aligned ({channel_names[i]})", color='green', alpha=0.7)
        
        if img1_filtered_conv is not None:
            hist_filtered, _ = np.histogram(img1_filtered_conv[:,:,i].flatten(), bins=50, range=range_vals)
            ax.plot(hist_filtered, label=f"Filtered ({channel_names[i]})", color='purple', alpha=0.7)
        
        # 获取统计信息
        mean_img1 = np.mean(img1_conv[:,:,i])
        std_img1 = np.std(img1_conv[:,:,i])
        mean_img2 = np.mean(img2_conv[:,:,i])
        std_img2 = np.std(img2_conv[:,:,i])
        
        # 在图中添加统计信息文本
        stats_text = f"Img1: mean={mean_img1:.2f}, std={std_img1:.2f}\nImg2: mean={mean_img2:.2f}, std={std_img2:.2f}"
        
        if img1_aligned_conv is not None:
            mean_aligned = np.mean(img1_aligned_conv[:,:,i])
            std_aligned = np.std(img1_aligned_conv[:,:,i])
            stats_text += f"\nAligned: mean={mean_aligned:.2f}, std={std_aligned:.2f}"
            
        if img1_filtered_conv is not None:
            mean_filtered = np.mean(img1_filtered_conv[:,:,i])
            std_filtered = np.std(img1_filtered_conv[:,:,i])
            stats_text += f"\nFiltered: mean={mean_filtered:.2f}, std={std_filtered:.2f}"
        
        ax.text(0.95, 0.95, stats_text, 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # 添加标题和图例
        ax.set_title(f"{channel_name} Channel")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

class ImageAlignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像通道对齐工具")
        self.root.geometry("1300x900")
        
        # 初始化变量
        self.img1_path = None
        self.img2_path = None
        self.img1 = None
        self.img2 = None
        self.img1_aligned = None
        self.img1_filtered = None  # 添加滤波后的图像
        self.img1_display = None
        self.img2_display = None
        self.img_aligned_display = None
        self.img_filtered_display = None  # 添加滤波后图像显示
        
        # 当前的色彩空间
        self.colorspace = tk.StringVar(value="RGB")
        
        # 对齐设置
        self.align_channel = tk.IntVar(value=2)  # 默认对齐第三个通道 (如RGB中的B或HSV中的V)
        self.match_mean = tk.BooleanVar(value=True)
        self.match_std = tk.BooleanVar(value=True)
        
        # 滤波设置
        self.filter_enabled = tk.BooleanVar(value=False)
        self.filter_diameter = tk.IntVar(value=9)
        self.filter_sigma_color = tk.IntVar(value=75)
        self.filter_sigma_space = tk.IntVar(value=75)
        
        # 创建界面
        self.create_ui()
        
    def create_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部控制栏
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 图像选择区域
        image_select_frame = ttk.LabelFrame(controls_frame, text="图像选择", padding="5")
        image_select_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(image_select_frame, text="选择图像1", command=lambda: self.load_image(1)).grid(row=0, column=0, padx=5, pady=5)
        self.img1_label = ttk.Label(image_select_frame, text="未选择")
        self.img1_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(image_select_frame, text="选择图像2", command=lambda: self.load_image(2)).grid(row=1, column=0, padx=5, pady=5)
        self.img2_label = ttk.Label(image_select_frame, text="未选择")
        self.img2_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 颜色空间选择
        colorspace_frame = ttk.LabelFrame(controls_frame, text="色彩空间", padding="5")
        colorspace_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Radiobutton(colorspace_frame, text="RGB", variable=self.colorspace, value="RGB", command=self.update_colorspace).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(colorspace_frame, text="HSV", variable=self.colorspace, value="HSV", command=self.update_colorspace).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(colorspace_frame, text="LAB", variable=self.colorspace, value="LAB", command=self.update_colorspace).pack(anchor="w", padx=5, pady=2)
        
        # 对齐设置
        align_frame = ttk.LabelFrame(controls_frame, text="对齐设置", padding="5")
        align_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Label(align_frame, text="对齐通道:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.channel_combo = ttk.Combobox(align_frame, width=5)
        self.channel_combo.grid(row=0, column=1, padx=5, pady=2)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_selected)
        self.update_channel_combo()  # 初始化通道选择
        
        ttk.Checkbutton(align_frame, text="匹配均值", variable=self.match_mean).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(align_frame, text="匹配标准差", variable=self.match_std).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(controls_frame, text="操作", padding="5")
        action_frame.pack(side=tk.LEFT, fill=tk.X, padx=(5, 0))
        
        ttk.Button(action_frame, text="执行对齐", command=self.align_images).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="保存对齐图像", command=self.save_aligned_image).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="分析统计", command=self.analyze_stats).pack(fill=tk.X, padx=5, pady=5)
        
        # 添加滤波控制面板
        filter_frame = ttk.LabelFrame(main_frame, text="双边滤波设置", padding="5")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 启用/禁用滤波
        ttk.Checkbutton(filter_frame, text="启用双边滤波", variable=self.filter_enabled, 
                    command=self.apply_filter).pack(side=tk.LEFT, padx=5)
        
        # 滤波参数设置
        params_frame = ttk.Frame(filter_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT, expand=True)
        
        # 直径设置
        ttk.Label(params_frame, text="直径:").grid(row=0, column=0, padx=5, pady=2)
        diameter_slider = ttk.Scale(params_frame, from_=3, to=25, orient="horizontal", variable=self.filter_diameter,
                               command=partial(self.on_filter_param_change, "diameter"))
        diameter_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.diameter_label = ttk.Label(params_frame, text="9")
        self.diameter_label.grid(row=0, column=2, padx=5, pady=2)
        
        # 颜色标准差设置
        ttk.Label(params_frame, text="颜色标准差:").grid(row=1, column=0, padx=5, pady=2)
        sigma_color_slider = ttk.Scale(params_frame, from_=10, to=150, orient="horizontal", variable=self.filter_sigma_color,
                                   command=partial(self.on_filter_param_change, "sigma_color"))
        sigma_color_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.sigma_color_label = ttk.Label(params_frame, text="75")
        self.sigma_color_label.grid(row=1, column=2, padx=5, pady=2)
        
        # 空间标准差设置
        ttk.Label(params_frame, text="空间标准差:").grid(row=2, column=0, padx=5, pady=2)
        sigma_space_slider = ttk.Scale(params_frame, from_=10, to=150, orient="horizontal", variable=self.filter_sigma_space,
                                   command=partial(self.on_filter_param_change, "sigma_space"))
        sigma_space_slider.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.sigma_space_label = ttk.Label(params_frame, text="75")
        self.sigma_space_label.grid(row=2, column=2, padx=5, pady=2)
        
        # 应用滤波按钮
        ttk.Button(filter_frame, text="应用滤波", command=self.apply_filter).pack(side=tk.RIGHT, padx=10)
        
        # 设置列权重
        params_frame.columnconfigure(1, weight=1)
        
        # 图像显示区域
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # 显示图像1
        img1_frame = ttk.LabelFrame(images_frame, text="图像1", padding="5")
        img1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.img1_canvas = tk.Canvas(img1_frame, bg="#eee")
        self.img1_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 显示图像2
        img2_frame = ttk.LabelFrame(images_frame, text="图像2", padding="5")
        img2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.img2_canvas = tk.Canvas(img2_frame, bg="#eee")
        self.img2_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 显示对齐结果
        aligned_frame = ttk.LabelFrame(images_frame, text="对齐结果", padding="5")
        aligned_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.aligned_canvas = tk.Canvas(aligned_frame, bg="#eee")
        self.aligned_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 显示滤波结果
        filtered_frame = ttk.LabelFrame(images_frame, text="滤波结果", padding="5")
        filtered_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.filtered_canvas = tk.Canvas(filtered_frame, bg="#eee")
        self.filtered_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 配置权重使得图像显示区域可以扩展
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.columnconfigure(2, weight=1)
        images_frame.rowconfigure(0, weight=1)
        images_frame.rowconfigure(1, weight=1)  # 为滤波结果行添加权重
        
    def on_filter_param_change(self, param, value):
        """处理滤波参数变化"""
        # 更新标签
        if param == "diameter":
            self.diameter_label.config(text=str(int(float(value))))
        elif param == "sigma_color":
            self.sigma_color_label.config(text=str(int(float(value))))
        elif param == "sigma_space":
            self.sigma_space_label.config(text=str(int(float(value))))
        
        # 如果启用了滤波，自动应用新参数
        if self.filter_enabled.get() and self.img1_aligned is not None:
            # 使用防抖措施，防止滑块快速滑动时产生过多计算
            if hasattr(self, "_filter_timer"):
                self.root.after_cancel(self._filter_timer)
            self._filter_timer = self.root.after(300, self.apply_filter)  # 等待300ms后应用过滤
    
    def apply_filter(self):
        """应用双边滤波"""
        if self.img1_aligned is None:
            if self.filter_enabled.get():
                messagebox.showinfo("提示", "请先执行图像对齐")
                self.filter_enabled.set(False)
            return
        
        try:
            if self.filter_enabled.get():
                # 应用双边滤波
                d = self.filter_diameter.get()
                sigma_color = self.filter_sigma_color.get()
                sigma_space = self.filter_sigma_space.get()
                
                self.img1_filtered = apply_bilateral_filter(
                    self.img1_aligned, 
                    d=d, 
                    sigma_color=sigma_color, 
                    sigma_space=sigma_space
                )
                
                # 显示滤波结果
                self.display_image(self.img1_filtered, self.filtered_canvas, 'img_filtered_display')
                
                self.status_var.set(f"已应用双边滤波: 直径={d}, 颜色σ={sigma_color}, 空间σ={sigma_space}")
            else:
                # 清除滤波结果
                self.img1_filtered = None
                self.filtered_canvas.delete("all")
                self.status_var.set("双边滤波已禁用")
        except Exception as e:
            messagebox.showerror("错误", f"应用滤波时出错: {str(e)}")
    
    def update_colorspace(self):
        """更新色彩空间并重新设置通道选择下拉菜单"""
        self.update_channel_combo()
        
        # 如果已经加载了图像，则更新显示
        if self.img1 is not None and self.img2 is not None:
            self.status_var.set(f"当前色彩空间: {self.colorspace.get()}")
    
    def update_channel_combo(self):
        """根据当前色彩空间更新通道选择下拉菜单"""
        colorspace = self.colorspace.get()
        
        if colorspace == "RGB":
            channels = ["R", "G", "B"]
        elif colorspace == "HSV":
            channels = ["H", "S", "V"]
        elif colorspace == "LAB":
            channels = ["L", "A", "B"]
        
        # 更新下拉菜单
        self.channel_combo['values'] = channels
        self.channel_combo.current(2)  # 默认选择第三个通道
    
    def on_channel_selected(self, event):
        """通道选择变更时的响应函数"""
        selected_index = self.channel_combo.current()
        if selected_index >= 0:
            self.align_channel.set(selected_index)
    
    def load_image(self, img_num):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title=f"选择图像 {img_num}",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif")])
        
        if not file_path:
            return
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("错误", "无法读取图像文件")
                return
                
            # 转换BGR到RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img_num == 1:
                self.img1_path = file_path
                self.img1 = img
                self.img1_label.config(text=os.path.basename(file_path))
                self.display_image(self.img1, self.img1_canvas, 'img1_display')
            else:
                self.img2_path = file_path
                self.img2 = img
                self.img2_label.config(text=os.path.basename(file_path))
                self.display_image(self.img2, self.img2_canvas, 'img2_display')
                
            # 如果两张图片尺寸不同，调整第二张图片大小以匹配第一张
            if self.img1 is not None and self.img2 is not None:
                if self.img1.shape != self.img2.shape:
                    self.img2 = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))
                    self.display_image(self.img2, self.img2_canvas, 'img2_display')
                    self.status_var.set("已调整图像2大小以匹配图像1")
            
            # 清除之前的结果
            self.img1_aligned = None
            self.img1_filtered = None
            self.aligned_canvas.delete("all")
            self.filtered_canvas.delete("all")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
    
    def display_image(self, img, canvas, attr_name):
        """在画布上显示图像"""
        # 将numpy数组转换为PIL图像
        pil_img = Image.fromarray(img)
        
        # 调整大小以适应画布
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 如果画布尺寸尚未确定，使用合理的默认值
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 300
        
        # 保持纵横比缩放图像
        width, height = pil_img.size
        ratio = min(canvas_width/width, canvas_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 调整图像大小
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建PhotoImage
        tk_img = ImageTk.PhotoImage(resized_img)
        
        # 保存引用以避免被垃圾收集
        setattr(self, attr_name, tk_img)
        
        # 清除画布并显示图像
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=tk_img)
    
    def align_images(self):
        """执行图像对齐"""
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("错误", "请先加载两张图像")
            return
        
        try:
            colorspace = self.colorspace.get()
            channel_index = self.align_channel.get()
            
            # 转换到目标色彩空间
            if colorspace == "RGB":
                img1_conv = self.img1
                img2_conv = self.img2
                to_rgb = lambda x: x  # 恒等函数
            elif colorspace == "HSV":
                img1_conv = rgb_to_hsv(self.img1)
                img2_conv = rgb_to_hsv(self.img2)
                to_rgb = hsv_to_rgb
            elif colorspace == "LAB":
                img1_conv = rgb_to_lab(self.img1)
                img2_conv = rgb_to_lab(self.img2)
                to_rgb = lab_to_rgb
            
            # 执行对齐
            aligned_conv = align_channel_statistics(
                img1_conv, 
                img2_conv, 
                channel_index,
                self.match_mean.get(),
                self.match_std.get()
            )
            
            # 转换回RGB
            self.img1_aligned = to_rgb(aligned_conv)
            
            # 显示对齐结果
            self.display_image(self.img1_aligned, self.aligned_canvas, 'img_aligned_display')
            
            # 清除滤波结果
            self.img1_filtered = None
            self.filtered_canvas.delete("all")
            
            # 如果启用了滤波，立即应用
            if self.filter_enabled.get():
                self.apply_filter()
            
            # 更新状态
            channel_name = self.channel_combo.get()
            self.status_var.set(f"已对齐 {colorspace} 色彩空间的 {channel_name} 通道")
            
        except Exception as e:
            messagebox.showerror("错误", f"对齐过程中发生错误: {str(e)}")
    
    def save_aligned_image(self):
        """保存对齐后或滤波后的图像"""
        # 确定要保存的图像
        if self.filter_enabled.get() and self.img1_filtered is not None:
            save_img = self.img1_filtered
            img_type = "滤波后的图像"
        elif self.img1_aligned is not None:
            save_img = self.img1_aligned
            img_type = "对齐后的图像"
        else:
            messagebox.showerror("错误", "请先执行图像对齐")
            return
        
        file_path = filedialog.asksaveasfilename(
            title=f"保存{img_type}",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 转换RGB到BGR（OpenCV格式）
                img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                self.status_var.set(f"{img_type}已保存至: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存图像时出错: {str(e)}")
    
    def analyze_stats(self):
        """分析并显示图像的统计数据"""
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("错误", "请先加载两张图像")
            return
        
        colorspace = self.colorspace.get()
        img1_filtered = self.img1_filtered if self.filter_enabled.get() else None
        
        try:
            # 生成统计数据可视化
            fig = plot_channel_histograms(
                self.img1, 
                self.img2, 
                self.img1_aligned,
                img1_filtered,
                colorspace,
                f"{colorspace} Channel Statistics"
            )
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            messagebox.showerror("错误", f"分析统计数据时出错: {str(e)}")

def main():
    """主函数，处理命令行参数并启动应用"""
    parser = argparse.ArgumentParser(description="图像通道对齐与滤波工具")
    parser.add_argument("--img1", type=str, help="第一张图片路径")
    parser.add_argument("--img2", type=str, help="第二张图片路径")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ImageAlignmentApp(root)
    
    # 如果指定了命令行图像路径，加载它们
    if args.img1 and os.path.exists(args.img1):
        img = cv2.imread(args.img1)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            app.img1 = img
            app.img1_path = args.img1
            app.img1_label.config(text=os.path.basename(args.img1))
            app.display_image(img, app.img1_canvas, 'img1_display')
    
    if args.img2 and os.path.exists(args.img2):
        img = cv2.imread(args.img2)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            app.img2 = img
            app.img2_path = args.img2
            app.img2_label.config(text=os.path.basename(args.img2))
            app.display_image(img, app.img2_canvas, 'img2_display')
            
            # 调整大小如需要
            if app.img1 is not None and app.img1.shape != app.img2.shape:
                app.img2 = cv2.resize(app.img2, (app.img1.shape[1], app.img1.shape[0]))
                app.display_image(app.img2, app.img2_canvas, 'img2_display')
    
    # 设置窗口的响应函数，当窗口大小改变时更新图像
    root.bind("<Configure>", lambda e: app.root.after(100, app.apply_filter) 
              if hasattr(app, 'img1_aligned') and app.img1_aligned is not None and app.filter_enabled.get() else None)
    
    # 启动应用
    root.mainloop()

if __name__ == "__main__":
    main()