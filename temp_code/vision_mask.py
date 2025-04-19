#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("视觉-触觉掩码查看器")
        self.root.geometry("1100x750")
        
        # 初始化变量
        self.current_index = 0
        self.visual_images = []  # 视觉图像路径列表
        self.mask_images = []    # 掩码图像路径列表
        self.serial_numbers = [] # 序列号列表
        self.current_visual = None
        self.current_mask = None
        self.base_dir = ""
        self.overlay_alpha = 0.4  # 默认透明度
        
        # 设置主题
        self.set_theme()
        
        # 创建UI
        self.create_ui()
        
        # 绑定键盘事件
        self.root.bind('<a>', lambda e: self.prev_image())
        self.root.bind('<d>', lambda e: self.next_image())
        
        # 显示欢迎信息
        self.update_status("欢迎使用! 请选择数据集根目录，按A/D键浏览图像")
    
    def set_theme(self):
        """设置应用程序主题颜色"""
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_color = "#F5F5F5"
        fg_color = "#333333"
        accent_color = "#4CAF50"
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=bg_color, foreground=fg_color)
        style.map('TButton',
                 background=[('active', accent_color)],
                 foreground=[('active', 'white')])
        
        self.root.configure(background=bg_color)
    
    def create_ui(self):
        # 主布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题栏 - 显示当前图片信息
        self.title_frame = ttk.Frame(main_frame)
        self.title_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.title_var = tk.StringVar()
        self.title_label = ttk.Label(self.title_frame, textvariable=self.title_var, 
                                     font=('Arial', 12, 'bold'), anchor=tk.CENTER)
        self.title_label.pack(fill=tk.X)
        
        # 进度条
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(self.progress_frame, text="进度:")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_text = ttk.Label(self.progress_frame, text="0/0")
        self.progress_text.pack(side=tk.LEFT, padx=(5, 0))
        
        # 内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 图像显示区域
        self.image_frame = ttk.Frame(content_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg='#F0F0F0')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 设置面板
        settings_frame = ttk.Frame(content_frame, width=200)
        settings_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 叠加设置
        overlay_frame = ttk.LabelFrame(settings_frame, text="叠加设置")
        overlay_frame.pack(fill=tk.X, pady=5)
        
        # 透明度滑块
        alpha_frame = ttk.Frame(overlay_frame)
        alpha_frame.pack(fill=tk.X, pady=10)
        
        alpha_label = ttk.Label(alpha_frame, text="透明度:")
        alpha_label.pack(side=tk.LEFT, padx=5)
        
        self.alpha_var = tk.DoubleVar(value=self.overlay_alpha)
        alpha_slider = ttk.Scale(alpha_frame, from_=0.0, to=1.0, 
                                orient=tk.HORIZONTAL, 
                                variable=self.alpha_var,
                                command=self.update_alpha)
        alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 目录选择
        dir_frame = ttk.LabelFrame(settings_frame, text="数据集设置")
        dir_frame.pack(fill=tk.X, pady=5)
        
        select_dir_btn = ttk.Button(dir_frame, text="选择数据集根目录", 
                                    command=self.select_directory)
        select_dir_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 目录信息显示
        self.dir_info_var = tk.StringVar(value="未选择目录")
        dir_info_label = ttk.Label(dir_frame, textvariable=self.dir_info_var,
                                  wraplength=180)
        dir_info_label.pack(padx=5, pady=5, fill=tk.X)
        
        # 图像信息
        image_info_frame = ttk.LabelFrame(settings_frame, text="图像信息")
        image_info_frame.pack(fill=tk.X, pady=5)
        
        self.visual_info_var = tk.StringVar(value="视觉图像: 未加载")
        visual_info_label = ttk.Label(image_info_frame, textvariable=self.visual_info_var,
                                     wraplength=180)
        visual_info_label.pack(padx=5, pady=2, fill=tk.X)
        
        self.mask_info_var = tk.StringVar(value="触觉掩码: 未加载")
        mask_info_label = ttk.Label(image_info_frame, textvariable=self.mask_info_var,
                                   wraplength=180)
        mask_info_label.pack(padx=5, pady=2, fill=tk.X)
        
        # 控制栏
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # 上一张按钮
        self.prev_btn = ttk.Button(control_frame, text="上一张 (A)", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        # 下一张按钮
        self.next_btn = ttk.Button(control_frame, text="下一张 (D)", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.status_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)
    
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_alpha(self, *args):
        """更新叠加透明度"""
        self.overlay_alpha = self.alpha_var.get()
        if self.current_visual is not None and self.current_mask is not None:
            self.display_overlay()
    
    def select_directory(self):
        """选择数据集根目录"""
        dir_path = filedialog.askdirectory(title="选择数据集根目录")
        if not dir_path:
            return
            
        self.base_dir = dir_path
        
        # 检查目录结构
        rgb_dir = os.path.join(dir_path, "rgb_images")
        mask_dir = os.path.join(dir_path, "touch_images_mask_process")
        
        if not os.path.isdir(rgb_dir) or not os.path.isdir(mask_dir):
            messagebox.showerror("错误", "找不到 rgb_images 或 touch_images_mask 子目录！")
            return
        
        # 加载图像列表
        self.load_image_lists(rgb_dir, mask_dir)
    
    def load_image_lists(self, rgb_dir, mask_dir):
        """加载视觉和掩码图像列表，并建立匹配关系"""
        self.visual_images = []
        self.mask_images = []
        self.serial_numbers = []
        
        # 获取所有视觉图像
        rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 获取所有掩码图像
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 解析序列号和建立对应关系
        serial_to_visual = {}
        serial_to_mask = {}
        
        for file in rgb_files:
            # 从 image_serial_number.png 格式中提取序列号
            if file.startswith("image_"):
                serial = file[6:].split(".")[0]  # 获取serial_number部分
                serial_to_visual[serial] = os.path.join(rgb_dir, file)
        
        for file in mask_files:
            # 从 gel_image_serial_number.png 格式中提取序列号
            if file.startswith("gel_image_"):
                serial = file[10:].split(".")[0]  # 获取serial_number部分
                serial_to_mask[serial] = os.path.join(mask_dir, file)
        
        # 保留两个列表中都有的序列号
        common_serials = sorted(list(set(serial_to_visual.keys()) & set(serial_to_mask.keys())))
        
        if not common_serials:
            messagebox.showwarning("警告", "找不到匹配的视觉和掩码图像对！")
            return
        
        # 按序列号排序并保存对应的文件路径
        for serial in common_serials:
            self.serial_numbers.append(serial)
            self.visual_images.append(serial_to_visual[serial])
            self.mask_images.append(serial_to_mask[serial])
        
        # 更新目录信息显示
        self.dir_info_var.set(f"当前目录: {self.base_dir}\n共找到 {len(self.serial_numbers)} 对匹配图像")
        
        # 加载第一对图像
        if self.serial_numbers:
            self.current_index = 0
            self.load_current_images()
            self.update_status(f"已加载 {len(self.serial_numbers)} 对图像")
    
    def load_current_images(self):
        """加载当前选中的图像对"""
        if not self.serial_numbers:
            return
        
        # 加载视觉图像
        visual_path = self.visual_images[self.current_index]
        self.current_visual = cv2.imread(visual_path)
        if self.current_visual is not None:
            self.current_visual = cv2.cvtColor(self.current_visual, cv2.COLOR_BGR2RGB)
            visual_h, visual_w = self.current_visual.shape[:2]
            self.visual_info_var.set(f"视觉图像: 已加载\n尺寸: {visual_w}x{visual_h}")
        else:
            self.visual_info_var.set(f"视觉图像: 加载失败")
        
        # 加载掩码图像
        mask_path = self.mask_images[self.current_index]
        self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.current_mask is not None:
            mask_h, mask_w = self.current_mask.shape[:2]
            self.mask_info_var.set(f"触觉掩码: 已加载\n尺寸: {mask_w}x{mask_h}")
        else:
            self.mask_info_var.set(f"触觉掩码: 加载失败")
        
        # 显示叠加图像
        if self.current_visual is not None and self.current_mask is not None:
            self.display_overlay()
        
        # 更新标题、进度条等
        self.update_title()
    
    def update_title(self):
        """更新标题和进度显示"""
        if not self.serial_numbers:
            self.title_var.set("未加载图像")
            self.progress_bar['value'] = 0
            self.progress_text.config(text="0/0")
            return
            
        # 更新标题
        serial = self.serial_numbers[self.current_index]
        self.title_var.set(f"序列号: {serial}")
        
        # 更新进度条
        total = len(self.serial_numbers)
        current = self.current_index + 1
        progress_pct = (current / total) * 100
        self.progress_bar['value'] = progress_pct
        self.progress_text.config(text=f"{current}/{total}")
    
    def display_overlay(self):
        """将掩码叠加在视觉图像上显示"""
        if self.current_visual is None or self.current_mask is None:
            return
        
        # 确保尺寸匹配，必要时调整掩码尺寸
        if self.current_visual.shape[:2] != self.current_mask.shape:
            resized_mask = cv2.resize(self.current_mask, 
                                     (self.current_visual.shape[1], self.current_visual.shape[0]))
        else:
            resized_mask = self.current_mask.copy()
        
        # 创建彩色掩码 - 使用浅色调
        colored_mask = self.create_colored_mask(resized_mask)
        
        # 根据透明度叠加
        alpha = self.overlay_alpha
        overlay = cv2.addWeighted(self.current_visual, 1-alpha, colored_mask, alpha, 0)
        
        # 显示叠加结果
        self.display_image(overlay)
    
    def create_colored_mask(self, mask):
        """创建蓝色系的彩色掩码"""
        # 创建空白的彩色掩码 (RGB格式)
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 在所有非零区域应用蓝色 (B通道)
        # 使用渐变蓝色，使强度与原始掩码值成比例
        colored_mask[:,:,0] = 0                 # R通道设为0
        colored_mask[:,:,1] = mask // 2         # G通道设为mask值的一半，添加一些青色调
        colored_mask[:,:,2] = mask              # B通道设为完整mask值
        
        # 可选：增加亮度，使颜色更鲜明
        brightness_factor = 1.5
        colored_mask = np.clip(colored_mask * brightness_factor, 0, 255).astype(np.uint8)
        
        return colored_mask
    
    def display_image(self, image):
        """在画布上显示图像"""
        # 转换为PIL图像并调整大小
        pil_img = Image.fromarray(image)
        
        # 获取画布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布尚未完全初始化，使用默认尺寸
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 500
        
        # 计算缩放比例，保持纵横比
        img_width, img_height = pil_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 调整图像大小
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 保存引用以防止垃圾回收
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.photo)
    
    def next_image(self):
        """显示下一张图像"""
        if self.serial_numbers and self.current_index < len(self.serial_numbers) - 1:
            self.current_index += 1
            self.load_current_images()
            self.update_status(f"当前图像: {self.serial_numbers[self.current_index]}")
    
    def prev_image(self):
        """显示上一张图像"""
        if self.serial_numbers and self.current_index > 0:
            self.current_index -= 1
            self.load_current_images()
            self.update_status(f"当前图像: {self.serial_numbers[self.current_index]}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()