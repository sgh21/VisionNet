#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pygame.mixer  # 用于音效
from datetime import datetime

class MaskEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("掩码编辑器")
        self.root.geometry("1100x750")
        
        # 设置应用程序主题颜色 - 使用浅色主题
        self.set_theme()
        
        # 初始化音效
        self.init_sounds()
        
        # 初始化变量
        self.current_index = 0
        self.image_files = []
        self.current_mask = None
        self.current_image = None  # 存储对应的原始图像
        self.mask_folder_path = ""
        self.image_folder_path = ""  # 原始图像文件夹路径
        self.history = []
        self.history_index = 0
        self.overlay_enabled = True  # 是否启用叠加显示
        self.overlay_alpha = 0.3  # 叠加透明度 - 改为更低的默认值
        self.saved_filenames = []  # 文件名缓存
        self.filter_filenames = []  # 从文件加载的文件名过滤列表
        
        # 创建UI
        self.create_ui()
        
        # 绑定键盘事件
        self.root.bind('<a>', lambda e: self.prev_image())
        self.root.bind('<d>', lambda e: self.next_image())
        self.root.bind('<s>', lambda e: self.save_mask())
        self.root.bind('<z>', lambda e: self.undo_edit())
        self.root.bind('<q>', lambda e: self.add_filename_to_cache())
        self.root.bind('<space>', lambda e: self.toggle_overlay())
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 显示欢迎信息
        self.update_status("欢迎使用掩码编辑器! 按A/D切换图片，点击移除区域，按S保存，按Q记录当前文件名")
    
    def set_theme(self):
        """设置应用程序主题颜色"""
        style = ttk.Style()
        style.theme_use('clam')  # 使用clam主题，它支持更多自定义
        
        # 设置各种元素的颜色
        bg_color = "#F5F5F5"       # 浅灰色背景
        fg_color = "#333333"       # 深灰色文本
        accent_color = "#4CAF50"   # 绿色强调色
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=bg_color, foreground=fg_color)
        style.map('TButton',
                 background=[('active', accent_color)],
                 foreground=[('active', 'white')])
        
        # 设置根窗口颜色
        self.root.configure(background=bg_color)
    
    def init_sounds(self):
        """初始化音效"""
        try:
            pygame.mixer.init()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 尝试加载音效文件，如果不存在则使用默认系统声音
            sound_path = os.path.join(script_dir, 'sounds', 'save.wav')
            if os.path.exists(sound_path):
                self.save_sound = pygame.mixer.Sound(sound_path)
            else:
                # 如果音效文件不存在，尝试使用系统声音或内置声音
                self.save_sound = None
                print("提示: 音效文件未找到，可以在程序目录下创建 'sounds/save.wav' 文件")
        except:
            self.save_sound = None
            print("提示: 初始化音效失败，将不会播放音效")
        
    def create_ui(self):
        # 主布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题栏 - 显示当前图片名称
        self.title_frame = ttk.Frame(main_frame)
        self.title_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.title_var = tk.StringVar()
        self.title_label = ttk.Label(self.title_frame, textvariable=self.title_var, 
                                     font=('Arial', 12, 'bold'), anchor=tk.CENTER)
        self.title_label.pack(fill=tk.X)
        
        # 进度条 - 显示当前浏览进度
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(self.progress_frame, text="进度:")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_text = ttk.Label(self.progress_frame, text="0/0")
        self.progress_text.pack(side=tk.LEFT, padx=(5, 0))
        
        # 图像和设置布局（水平分割）
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 图像显示区域
        self.image_frame = ttk.Frame(content_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg='#F0F0F0')  # 更浅的灰色背景
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 设置面板
        settings_frame = ttk.Frame(content_frame, width=200)
        settings_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 叠加设置
        overlay_frame = ttk.LabelFrame(settings_frame, text="叠加设置")
        overlay_frame.pack(fill=tk.X, pady=5)
        
        # 叠加切换
        overlay_toggle_frame = ttk.Frame(overlay_frame)
        overlay_toggle_frame.pack(fill=tk.X, pady=5)
        
        self.overlay_var = tk.BooleanVar(value=self.overlay_enabled)
        overlay_check = ttk.Checkbutton(overlay_toggle_frame, text="启用叠加显示", 
                                      variable=self.overlay_var, 
                                      command=self.toggle_overlay)
        overlay_check.pack(side=tk.LEFT, padx=5)
        
        # 透明度滑块
        alpha_frame = ttk.Frame(overlay_frame)
        alpha_frame.pack(fill=tk.X, pady=5)
        
        alpha_label = ttk.Label(alpha_frame, text="透明度:")
        alpha_label.pack(side=tk.LEFT, padx=5)
        
        self.alpha_var = tk.DoubleVar(value=self.overlay_alpha)
        alpha_slider = ttk.Scale(alpha_frame, from_=0.0, to=1.0, 
                                orient=tk.HORIZONTAL, 
                                variable=self.alpha_var,
                                command=self.update_alpha)
        alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 文件夹选择框
        folders_frame = ttk.LabelFrame(settings_frame, text="文件夹设置")
        folders_frame.pack(fill=tk.X, pady=5)
        
        mask_folder_btn = ttk.Button(folders_frame, text="选择掩码文件夹", 
                                     command=self.open_mask_folder)
        mask_folder_btn.pack(fill=tk.X, padx=5, pady=5)
        
        image_folder_btn = ttk.Button(folders_frame, text="选择图像文件夹", 
                                      command=self.open_image_folder)
        image_folder_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加文件过滤功能
        filenames_filter_frame = ttk.LabelFrame(settings_frame, text="文件名过滤")
        filenames_filter_frame.pack(fill=tk.X, pady=5)
        
        load_filter_btn = ttk.Button(filenames_filter_frame, text="从文本文件加载文件名列表", 
                                     command=self.load_filenames_filter)
        load_filter_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示已加载的过滤器信息
        self.filter_info_var = tk.StringVar(value="未加载过滤列表")
        filter_info_label = ttk.Label(filenames_filter_frame, textvariable=self.filter_info_var)
        filter_info_label.pack(padx=5, pady=5)
        
        # 重置过滤器按钮
        reset_filter_btn = ttk.Button(filenames_filter_frame, text="重置文件名过滤", 
                                     command=self.reset_filenames_filter)
        reset_filter_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 文件名保存框
        self.filename_frame = ttk.LabelFrame(settings_frame, text="文件名记录")
        self.filename_frame.pack(fill=tk.X, pady=5)
        
        # 添加当前文件到缓存
        add_name_btn = ttk.Button(self.filename_frame, text="记录当前文件名 (Q)", 
                                 command=self.add_filename_to_cache)
        add_name_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 保存所有已记录的文件名
        save_names_btn = ttk.Button(self.filename_frame, text="保存所有记录的文件名", 
                                   command=self.save_cached_filenames)
        save_names_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 已保存文件计数
        self.saved_count_var = tk.StringVar(value="已记录: 0 个文件")
        saved_count_label = ttk.Label(self.filename_frame, textvariable=self.saved_count_var)
        saved_count_label.pack(padx=5, pady=5)
        
        # 控制栏
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # 上一张按钮
        self.prev_btn = ttk.Button(control_frame, text="上一张 (A)", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        # 下一张按钮
        self.next_btn = ttk.Button(control_frame, text="下一张 (D)", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存按钮
        self.save_btn = ttk.Button(control_frame, text="保存 (S)", command=self.save_mask)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 撤销按钮
        self.undo_btn = ttk.Button(control_frame, text="撤销 (Z)", command=self.undo_edit)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
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
        # 刷新UI，确保状态栏立即更新
        self.root.update_idletasks()
    
    def update_title(self):
        """更新标题和进度显示"""
        if not self.image_files:
            self.title_var.set("未加载图片")
            self.progress_bar['value'] = 0
            self.progress_text.config(text="0/0")
            return
            
        # 更新标题
        self.title_var.set(f"{self.image_files[self.current_index]}")
        
        # 更新进度条
        total = len(self.image_files)
        current = self.current_index + 1
        progress_pct = (current / total) * 100
        self.progress_bar['value'] = progress_pct
        self.progress_text.config(text=f"{current}/{total}")
    
    def load_filenames_filter(self):
        """从文本文件加载文件名过滤列表"""
        filter_path = filedialog.askopenfilename(
            title="选择文件名列表文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filter_path:
            return
            
        try:
            # 从文本文件加载文件名列表
            with open(filter_path, 'r') as f:
                lines = f.readlines()
            
            # 处理文件名列表，移除注释和空行
            self.filter_filenames = []
            for line in lines:
                # 去掉注释部分
                if '#' in line:
                    line = line.split('#')[0]
                
                # 清理空白字符
                line = line.strip()
                
                # 如果不是空行，添加到过滤列表
                if line:
                    self.filter_filenames.append(line)
            
            # 更新过滤器信息显示
            self.filter_info_var.set(f"已加载: {len(self.filter_filenames)} 个文件名")
            
            # 如果掩码文件夹已经设置，重新加载图像
            if self.mask_folder_path:
                self.load_images()
                self.update_status(f"已从文件加载 {len(self.filter_filenames)} 个文件名用于过滤")
            else:
                self.update_status("过滤列表已加载，请选择掩码文件夹")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载文件名列表失败: {str(e)}")
    
    def reset_filenames_filter(self):
        """重置文件名过滤器，显示所有文件"""
        self.filter_filenames = []
        self.filter_info_var.set("未加载过滤列表")
        
        # 如果掩码文件夹已经设置，重新加载所有图像
        if self.mask_folder_path:
            self.load_images()
            self.update_status("已重置过滤器，显示所有文件")
        else:
            self.update_status("已重置过滤器")
    
    def open_mask_folder(self):
        """打开掩码文件夹并加载图像"""
        folder_path = filedialog.askdirectory(title="选择掩码图像文件夹")
        if folder_path:
            self.mask_folder_path = folder_path
            self.load_images()
    
    def open_image_folder(self):
        """打开原始图像文件夹"""
        folder_path = filedialog.askdirectory(title="选择原始图像文件夹")
        if folder_path:
            self.image_folder_path = folder_path
            # 如果掩码已经加载，重新加载当前图像以显示叠加效果
            if self.current_mask is not None:
                # 清除当前图像以确保重新加载
                self.current_image = None  
                self.load_current_image()
                self.update_status(f"已设置原始图像文件夹: {folder_path}")
    
    def load_images(self):
        """加载文件夹中的所有图像，根据过滤列表筛选"""
        if not self.mask_folder_path:
            messagebox.showwarning("警告", "请先选择掩码文件夹")
            return
            
        # 获取所有图像文件
        all_files = [f for f in os.listdir(self.mask_folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 应用过滤
        if self.filter_filenames:
            # 只保留过滤列表中的文件
            self.image_files = [f for f in all_files if f in self.filter_filenames]
            
            # 显示过滤结果
            filtered_count = len(self.image_files)
            total_count = len(all_files)
            self.update_status(f"应用过滤: 显示 {filtered_count}/{total_count} 个文件")
        else:
            # 没有过滤器，显示所有文件
            self.image_files = all_files
        
        if not self.image_files:
            messagebox.showwarning("警告", "没有找到符合条件的图像文件")
            return
        
        # 重置当前索引
        self.current_index = 0
        self.load_current_image()
        
        # 更新状态栏和标题
        if self.filter_filenames:
            self.update_status(f"已加载 {len(self.image_files)}/{len(all_files)} 个文件 (已过滤)")
        else:
            self.update_status(f"已加载 {len(self.image_files)} 个文件")
        
        self.update_title()
    
    def load_current_image(self):
        """加载当前选中的图像"""
        if not self.image_files:
            return
        
        # 加载当前掩码
        mask_path = os.path.join(self.mask_folder_path, self.image_files[self.current_index])
        self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.current_mask is None:
            messagebox.showerror("错误", f"无法加载掩码图像: {mask_path}")
            return
            
        # 尝试加载对应的原始图像（如果图像文件夹已设置）
        self.current_image = None  # 确保清除旧的图像
        if self.image_folder_path:
            image_path = os.path.join(self.image_folder_path, self.image_files[self.current_index])
            if os.path.exists(image_path):
                try:
                    self.current_image = cv2.imread(image_path)
                    if self.current_image is not None:
                        # 转换为RGB格式（OpenCV读取的是BGR格式）
                        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                        print(f"成功加载原始图像: {image_path}, 形状: {self.current_image.shape}")
                    else:
                        print(f"无法解码原始图像: {image_path}")
                except Exception as e:
                    print(f"加载原始图像时出错: {str(e)}")
                    self.current_image = None
            else:
                print(f"原始图像不存在: {image_path}")
        
        # 创建编辑历史
        self.history = [self.current_mask.copy()]
        self.history_index = 0
        
        # 显示图像
        self.display_mask()
        
        # 更新标题和进度条
        self.update_title()
        
        # 更新状态栏
        msg = f"文件 {self.current_index + 1}/{len(self.image_files)}: {self.image_files[self.current_index]}"
        if self.current_image is not None:
            msg += " (已加载原始图像)"
        self.update_status(msg)
    
    def display_mask(self):
        """将掩码显示在画布上，如果启用了叠加模式且有原始图像，则显示叠加效果"""
        if self.current_mask is None:
            return
        
        # 根据当前设置决定显示方式
        if self.overlay_enabled and self.current_image is not None:
            # 创建叠加图像
            display_img = self.create_overlay_image()
        else:
            # 创建彩色图像用于显示 - 使用更柔和的颜色方案
            colored_mask = self.create_colored_mask()
            display_img = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像并调整大小
        pil_img = Image.fromarray(display_img)
        
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
        
        # 保存当前显示图像的尺寸信息，用于点击坐标转换
        self.display_info = {
            'orig_width': img_width,
            'orig_height': img_height,
            'disp_width': new_width,
            'disp_height': new_height,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }
    
    def create_colored_mask(self):
        """创建一个彩色掩码，使用更柔和的颜色方案"""
        # 首先，创建一个调整后的掩码 - 确保值在有效范围内
        adjusted_mask = self.current_mask.copy()
        
        # 使用浅色调的热力图 - PINK/BONE是比较柔和的方案
        colored_mask = cv2.applyColorMap(adjusted_mask, cv2.COLORMAP_PINK)
        
        # 增加亮度，使颜色更浅
        brightness_factor = 2.0  # 亮度增加因子 - 增大这个值使颜色更浅
        colored_mask = np.clip(colored_mask * brightness_factor, 0, 255).astype(np.uint8)
        
        return colored_mask
    
    def create_overlay_image(self):
        """创建一个叠加显示的图像，将掩码叠加在原始图像上"""
        if self.current_image is None:
            print("错误：没有可用的原始图像进行叠加")
            return self.create_colored_mask()  # 如果没有原始图像，返回彩色掩码
            
        # 确保原始图像和掩码尺寸匹配
        try:
            if self.current_image.shape[:2] != self.current_mask.shape:
                print(f"调整图像大小 - 原始图像: {self.current_image.shape[:2]}, 掩码: {self.current_mask.shape}")
                resized_image = cv2.resize(self.current_image, 
                                          (self.current_mask.shape[1], self.current_mask.shape[0]))
            else:
                resized_image = self.current_image.copy()
            
            # 创建彩色掩码 - 使用更柔和的颜色方案
            colored_mask = self.create_colored_mask()
            
            # 根据透明度叠加
            alpha = self.alpha_var.get()
            overlay = cv2.addWeighted(resized_image, 1-alpha, colored_mask, alpha, 0)
            
            return overlay
            
        except Exception as e:
            print(f"创建叠加图像时出错: {str(e)}")
            # 出错时仍返回彩色掩码
            return cv2.cvtColor(self.create_colored_mask(), cv2.COLOR_BGR2RGB)
    
    def toggle_overlay(self, *args):
        """切换叠加显示模式"""
        self.overlay_enabled = self.overlay_var.get()
        self.display_mask()
        
        mode = "启用" if self.overlay_enabled else "禁用"
        has_image = "可用" if self.current_image is not None else "不可用"
        self.update_status(f"叠加显示模式: {mode}, 原始图像: {has_image}")
    
    def update_alpha(self, *args):
        """更新叠加透明度"""
        self.overlay_alpha = self.alpha_var.get()
        if self.overlay_enabled and self.current_image is not None:
            self.display_mask()
    
    def next_image(self):
        """显示下一张图像"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def prev_image(self):
        """显示上一张图像"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def save_mask(self):
        """保存当前编辑的掩码"""
        if self.current_mask is None or not self.image_files:
            return
        
        # 保存当前掩码
        save_path = os.path.join(self.mask_folder_path, self.image_files[self.current_index])
        cv2.imwrite(save_path, self.current_mask)
        
        # 播放保存音效
        if self.save_sound:
            try:
                self.save_sound.play()
            except:
                pass
        
        # 显示保存成功的提示
        self.flash_save_indicator()
        
        # 在状态栏显示保存成功的提示
        self.update_status(f"✓ 已保存到 {save_path}")
    
    def add_filename_to_cache(self, *args):
        """将当前文件名添加到缓存中"""
        if not self.image_files or self.current_index < 0 or self.current_index >= len(self.image_files):
            self.update_status("没有可记录的文件名")
            return
            
        filename = self.image_files[self.current_index]
        
        # 检查是否已经在缓存中
        if filename in self.saved_filenames:
            self.update_status(f"文件名 '{filename}' 已经记录")
            return
            
        # 添加到缓存
        self.saved_filenames.append(filename)
        
        # 更新计数
        self.saved_count_var.set(f"已记录: {len(self.saved_filenames)} 个文件")
        
        # 播放保存音效
        if self.save_sound:
            try:
                self.save_sound.play()
            except:
                pass
                
        # 显示提示
        self.update_status(f"✓ 文件名 '{filename}' 已添加到记录 (总计: {len(self.saved_filenames)})")
    
    def save_cached_filenames(self):
        """保存所有缓存的文件名到文本文件"""
        if not self.saved_filenames:
            messagebox.showinfo("提示", "没有已记录的文件名")
            return
            
        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="selected_filenames.txt"
        )
        
        if not save_path:
            return
            
        # 保存所有记录的文件名
        try:
            with open(save_path, "w") as f:
                # 添加标题行
                f.write("# 保存的文件名列表\n")
                f.write(f"# 保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 总计: {len(self.saved_filenames)} 个文件\n\n")
                
                # 写入所有文件名，每行一个
                for filename in self.saved_filenames:
                    f.write(f"{filename}\n")
            
            # 播放保存音效
            if self.save_sound:
                try:
                    self.save_sound.play()
                except:
                    pass
                    
            # 显示提示
            self.update_status(f"✓ 已保存 {len(self.saved_filenames)} 个文件名到 {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存文件名失败: {str(e)}")
    
    def on_closing(self):
        """关闭窗口时的处理"""
        # 如果有缓存的文件名，提示用户保存
        if self.saved_filenames:
            answer = messagebox.askyesnocancel(
                "保存记录", 
                f"您有 {len(self.saved_filenames)} 个已记录的文件名，是否要保存？",
                icon=messagebox.QUESTION
            )
            
            if answer is None:  # 取消关闭
                return
                
            if answer:  # 选择保存
                self.save_cached_filenames()
        
        # 关闭窗口
        self.root.destroy()
    
    def flash_save_indicator(self):
        """显示醒目的保存成功提示"""
        # 创建一个临时的保存成功提示框
        flash_frame = tk.Frame(self.root, bg='#4CAF50')  # 绿色背景
        flash_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, 
                         width=200, height=80)
        
        save_label = tk.Label(flash_frame, text="✓ 保存成功", 
                              font=('Arial', 16, 'bold'),
                              bg='#4CAF50', fg='white')
        save_label.pack(fill=tk.BOTH, expand=True)
        
        # 短暂显示后移除
        self.root.after(800, flash_frame.destroy)
        
        # 同时高亮保存按钮
        orig_bg = self.save_btn.cget('background')
        self.save_btn.configure(background='lightgreen')
        self.root.after(800, lambda: self.save_btn.configure(background=orig_bg))
    
    def undo_edit(self):
        """撤销上一次编辑操作"""
        if len(self.history) > 1 and self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            self.display_mask()
            self.update_status(f"已撤销操作 ({self.history_index+1}/{len(self.history)})")
    
    def on_canvas_click(self, event):
        """处理画布点击事件"""
        if self.current_mask is None:
            return
        
        if not hasattr(self, 'display_info'):
            return
            
        # 计算画布中心点
        canvas_center_x = self.display_info['canvas_width'] // 2
        canvas_center_y = self.display_info['canvas_height'] // 2
        
        # 计算图像左上角在画布上的坐标
        img_left = canvas_center_x - self.display_info['disp_width'] // 2
        img_top = canvas_center_y - self.display_info['disp_height'] // 2
        
        # 计算点击在图像上的坐标
        rel_x = event.x - img_left
        rel_y = event.y - img_top
        
        # 检查点击是否在图像范围内
        if (rel_x < 0 or rel_x >= self.display_info['disp_width'] or 
            rel_y < 0 or rel_y >= self.display_info['disp_height']):
            return
        
        # 转换为原始图像上的坐标
        orig_x = int(rel_x * (self.display_info['orig_width'] / self.display_info['disp_width']))
        orig_y = int(rel_y * (self.display_info['orig_height'] / self.display_info['disp_height']))
        
        # 确保坐标在有效范围内
        if orig_x >= self.current_mask.shape[1] or orig_y >= self.current_mask.shape[0]:
            return
            
        # 获取点击位置的像素值
        seed_value = self.current_mask[orig_y, orig_x]
        
        # 如果点击的是已经为0的区域，则不进行处理
        if seed_value == 0:
            self.update_status("点击位置已经是背景区域")
            return
        
        # 创建洪水填充的掩码
        h, w = self.current_mask.shape[:2]
        flood_fill_mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        # 进行洪水填充，找到连通区域并填充为0
        cv2.floodFill(self.current_mask, flood_fill_mask, (orig_x, orig_y), 
                      0, 0, 0, flags=4)  # 4连通性
        
        # 保存当前状态到历史记录
        self.history = self.history[:self.history_index+1]  # 删除当前位置之后的历史
        self.history.append(self.current_mask.copy())
        self.history_index = len(self.history) - 1
        
        # 更新显示
        self.display_mask()
        
        # 更新状态栏
        self.update_status(f"已移除坐标 ({orig_x}, {orig_y}) 的连通区域，点击值: {seed_value}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskEditorApp(root)
    root.mainloop()