import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
import matplotlib.path as mpath
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image
import math
import pandas as pd

class RotatedBoxAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Rotated Bounding Box Annotator")
        self.root.geometry("1200x800")
        
        # 设置Tkinter风格主题
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        
        # 图像和标注数据
        self.image_folder = ""         # 输入图像文件夹路径
        self.output_folder = ""        # 输出文件夹路径
        self.image_files = []          # 图像文件列表
        self.current_index = -1        # 当前图像索引
        self.current_image = None      # 当前图像数据
        self.csv_data = None           # CSV数据
        
        # 标注相关变量
        self.annotations = []          # 当前图像的所有标注框 [(x,y,w,h,theta), ...]
        self.drawing = False           # 是否正在绘制标注框
        self.rect_start = None         # 标注框起始点
        self.rect_current = None       # 标注框当前点
        self.rotation = 0.0            # 当前标注框旋转角度
        self.current_rect = None       # 当前正在绘制的矩形对象
        self.selected_rect_idx = -1    # 当前选中的矩形索引
        
        # 矩形编辑状态
        self.edit_mode = False         # 是否处于编辑模式
        self.drag_start = None         # 拖动起始点
        self.resize_handle = None      # 当前正在调整的控制点
        self.edit_rect_idx = -1        # 正在编辑的矩形索引
        
        # 创建界面
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
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title("No Image Loaded")
        self.ax.axis('off')
        
        # 在tkinter窗口中放置matplotlib图形
        canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 文件夹选择框架
        folder_frame = ttk.LabelFrame(right_frame, text="Folder Selection")
        folder_frame.pack(fill=tk.X, pady=5, padx=5)
        
        input_btn = ttk.Button(folder_frame, text="Select Input Folder", 
                              command=self.select_input_folder)
        input_btn.pack(fill=tk.X, pady=5, padx=5)
        
        self.input_label = ttk.Label(folder_frame, text="No folder selected", 
                                   wraplength=250)
        self.input_label.pack(fill=tk.X, pady=5, padx=5)
        
        output_btn = ttk.Button(folder_frame, text="Select Output Folder", 
                               command=self.select_output_folder)
        output_btn.pack(fill=tk.X, pady=5, padx=5)
        
        self.output_label = ttk.Label(folder_frame, text="No folder selected", 
                                    wraplength=250)
        self.output_label.pack(fill=tk.X, pady=5, padx=5)
        
        # CSV文件选择
        csv_btn = ttk.Button(folder_frame, text="Load CSV Reference File", 
                           command=self.load_csv_reference)
        csv_btn.pack(fill=tk.X, pady=5, padx=5)
        
        self.csv_label = ttk.Label(folder_frame, text="No CSV file loaded", 
                                 wraplength=250)
        self.csv_label.pack(fill=tk.X, pady=5, padx=5)
        
        # 导航控制框架
        nav_frame = ttk.LabelFrame(right_frame, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=5, padx=5)
        
        nav_btns = ttk.Frame(nav_frame)
        nav_btns.pack(fill=tk.X, pady=5, padx=5)
        
        prev_btn = ttk.Button(nav_btns, text="Previous", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        next_btn = ttk.Button(nav_btns, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.filename_label = ttk.Label(nav_frame, text="No file loaded", 
                                      wraplength=250)
        self.filename_label.pack(fill=tk.X, pady=5, padx=5)
        
        self.counter_label = ttk.Label(nav_frame, text="0 / 0")
        self.counter_label.pack(fill=tk.X, pady=5, padx=5)
        
        # 标注控制框架
        annot_frame = ttk.LabelFrame(right_frame, text="Annotation Controls")
        annot_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # 角度输入框 - 新增
        angle_input_frame = ttk.Frame(annot_frame)
        angle_input_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(angle_input_frame, text="Angle:").pack(side=tk.LEFT, padx=5)
        
        vcmd = (self.root.register(self.validate_angle_input), '%P')
        self.angle_entry = ttk.Entry(angle_input_frame, validate='key', validatecommand=vcmd, width=8)
        self.angle_entry.pack(side=tk.LEFT, padx=5)
        self.angle_entry.insert(0, "0.00")
        
        apply_angle_btn = ttk.Button(angle_input_frame, text="Apply", 
                                   command=self.apply_angle_from_entry)
        apply_angle_btn.pack(side=tk.LEFT, padx=5)
        
        # 角度控制 - 范围改为-10到10度
        angle_frame = ttk.Frame(annot_frame)
        angle_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(angle_frame, text="Rotation (degrees):").pack(side=tk.LEFT, padx=5)
        
        self.angle_var = tk.DoubleVar(value=0.0)
        angle_scale = ttk.Scale(angle_frame, from_=-10, to=10, variable=self.angle_var,
                              command=lambda a: self.update_rotation_value(float(a)))
        angle_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.angle_label = ttk.Label(angle_frame, text="0.00°")
        self.angle_label.pack(side=tk.RIGHT, padx=5)
        
        # 添加精确调整按钮
        fine_adjust_frame = ttk.Frame(annot_frame)
        fine_adjust_frame.pack(fill=tk.X, pady=5, padx=5)
        
        minus_01_btn = ttk.Button(fine_adjust_frame, text="-0.1°", 
                                command=lambda: self.adjust_angle(-0.1))
        minus_01_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        minus_001_btn = ttk.Button(fine_adjust_frame, text="-0.01°", 
                                 command=lambda: self.adjust_angle(-0.01))
        minus_001_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        plus_001_btn = ttk.Button(fine_adjust_frame, text="+0.01°", 
                                command=lambda: self.adjust_angle(0.01))
        plus_001_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        plus_01_btn = ttk.Button(fine_adjust_frame, text="+0.1°", 
                               command=lambda: self.adjust_angle(0.1))
        plus_01_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # 矩形参数输入框框架
        rect_param_frame = ttk.LabelFrame(annot_frame, text="Rectangle Parameters")
        rect_param_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # X坐标输入
        x_frame = ttk.Frame(rect_param_frame)
        x_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(x_frame, text="X:").pack(side=tk.LEFT, padx=5)
        
        self.x_entry = ttk.Entry(x_frame, width=8)
        self.x_entry.pack(side=tk.LEFT, padx=5)
        self.x_entry.insert(0, "0.0")
        
        x_btn_frame = ttk.Frame(x_frame)
        x_btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(x_btn_frame, text="-1", command=lambda: self.adjust_rect_param('x', -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(x_btn_frame, text="-0.1", command=lambda: self.adjust_rect_param('x', -0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(x_btn_frame, text="+0.1", command=lambda: self.adjust_rect_param('x', 0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(x_btn_frame, text="+1", command=lambda: self.adjust_rect_param('x', 1)).pack(side=tk.LEFT, padx=2)
        
        # Y坐标输入
        y_frame = ttk.Frame(rect_param_frame)
        y_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(y_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        
        self.y_entry = ttk.Entry(y_frame, width=8)
        self.y_entry.pack(side=tk.LEFT, padx=5)
        self.y_entry.insert(0, "0.0")
        
        y_btn_frame = ttk.Frame(y_frame)
        y_btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(y_btn_frame, text="-1", command=lambda: self.adjust_rect_param('y', -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(y_btn_frame, text="-0.1", command=lambda: self.adjust_rect_param('y', -0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(y_btn_frame, text="+0.1", command=lambda: self.adjust_rect_param('y', 0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(y_btn_frame, text="+1", command=lambda: self.adjust_rect_param('y', 1)).pack(side=tk.LEFT, padx=2)
        
        # 宽度输入
        w_frame = ttk.Frame(rect_param_frame)
        w_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(w_frame, text="W:").pack(side=tk.LEFT, padx=5)
        
        self.w_entry = ttk.Entry(w_frame, width=8)
        self.w_entry.pack(side=tk.LEFT, padx=5)
        self.w_entry.insert(0, "0.0")
        
        w_btn_frame = ttk.Frame(w_frame)
        w_btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(w_btn_frame, text="-1", command=lambda: self.adjust_rect_param('w', -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(w_btn_frame, text="-0.1", command=lambda: self.adjust_rect_param('w', -0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(w_btn_frame, text="+0.1", command=lambda: self.adjust_rect_param('w', 0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(w_btn_frame, text="+1", command=lambda: self.adjust_rect_param('w', 1)).pack(side=tk.LEFT, padx=2)
        
        # 高度输入
        h_frame = ttk.Frame(rect_param_frame)
        h_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(h_frame, text="H:").pack(side=tk.LEFT, padx=5)
        
        self.h_entry = ttk.Entry(h_frame, width=8)
        self.h_entry.pack(side=tk.LEFT, padx=5)
        self.h_entry.insert(0, "0.0")
        
        h_btn_frame = ttk.Frame(h_frame)
        h_btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(h_btn_frame, text="-1", command=lambda: self.adjust_rect_param('h', -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(h_btn_frame, text="-0.1", command=lambda: self.adjust_rect_param('h', -0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(h_btn_frame, text="+0.1", command=lambda: self.adjust_rect_param('h', 0.1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(h_btn_frame, text="+1", command=lambda: self.adjust_rect_param('h', 1)).pack(side=tk.LEFT, padx=2)
        
        # 应用按钮
        apply_params_btn = ttk.Button(rect_param_frame, text="Apply Parameters", 
                                    command=self.apply_rect_params)
        apply_params_btn.pack(fill=tk.X, pady=5, padx=5)

        # 编辑模式切换按钮
        edit_mode_frame = ttk.Frame(annot_frame)
        edit_mode_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.edit_mode_var = tk.BooleanVar(value=False)
        edit_mode_check = ttk.Checkbutton(edit_mode_frame, text="Edit Mode (Move/Resize)",
                                        variable=self.edit_mode_var,
                                        command=self.toggle_edit_mode)
        edit_mode_check.pack(fill=tk.X)
        
        # 删除当前标注按钮
        delete_btn = ttk.Button(annot_frame, text="Delete Selected Annotation", 
                               command=self.delete_selected_annotation)
        delete_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # 清除所有标注按钮
        clear_btn = ttk.Button(annot_frame, text="Clear All Annotations", 
                              command=self.clear_annotations)
        clear_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # 标注列表框架
        list_frame = ttk.LabelFrame(right_frame, text="Annotations List")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # 创建一个滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建ListBox并关联滚动条
        self.annot_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        selectmode=tk.SINGLE, height=10)
        self.annot_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.annot_listbox.yview)
        
        # ListBox选择事件绑定
        self.annot_listbox.bind('<<ListboxSelect>>', self.on_annotation_selected)
        
        # 保存按钮
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=10, padx=5)
        
        save_btn = ttk.Button(save_frame, text="Save Annotations", 
                             command=self.save_annotations)
        save_btn.pack(fill=tk.X)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(right_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, pady=5)
        
        # 显示操作提示
        self.show_instructions()
    # 添加以下方法到类中

    def adjust_rect_param(self, param, delta):
        """调整矩形参数（相对变化）"""
        if self.selected_rect_idx < 0 or self.selected_rect_idx >= len(self.annotations):
            return
        
        # 获取当前矩形参数
        x, y, w, h, theta = self.annotations[self.selected_rect_idx]
        
        # 根据参数类型调整
        if param == 'x':
            x += delta
            self.x_entry.delete(0, tk.END)
            self.x_entry.insert(0, f"{x:.1f}")
        elif param == 'y':
            y += delta
            self.y_entry.delete(0, tk.END)
            self.y_entry.insert(0, f"{y:.1f}")
        elif param == 'w':
            w = max(5, w + delta)  # 确保宽度不小于5
            self.w_entry.delete(0, tk.END)
            self.w_entry.insert(0, f"{w:.1f}")
        elif param == 'h':
            h = max(5, h + delta)  # 确保高度不小于5
            self.h_entry.delete(0, tk.END)
            self.h_entry.insert(0, f"{h:.1f}")
        
        # 更新标注
        self.annotations[self.selected_rect_idx] = (x, y, w, h, theta)
        self.update_annotations_display()
        self.status_var.set(f"Adjusted {param} by {delta}")

    def apply_rect_params(self):
        """应用输入框中的矩形参数"""
        if self.selected_rect_idx < 0 or self.selected_rect_idx >= len(self.annotations):
            messagebox.showwarning("Warning", "No annotation selected")
            return
        
        try:
            # 读取输入框中的值
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            w = float(self.w_entry.get())
            h = float(self.h_entry.get())
            
            # 验证宽高有效性
            if w < 5 or h < 5:
                messagebox.showwarning("Warning", "Width and height must be at least 5")
                return
            
            # 获取当前角度
            _, _, _, _, theta = self.annotations[self.selected_rect_idx]
            
            # 更新标注
            self.annotations[self.selected_rect_idx] = (x, y, w, h, theta)
            
            # 更新显示
            self.update_annotations_display()
            self.status_var.set("Applied rectangle parameters")
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter valid numbers.")
    def validate_angle_input(self, value):
        """验证输入的角度值"""
        # 允许空输入
        if value == "":
            return True
            
        # 允许负号在开头
        if value == "-":
            return True
            
        # 检查是否为有效浮点数格式
        try:
            if value.count('.') <= 1:  # 最多只能有一个小数点
                # 尝试转换为浮点数
                if value[-1] == '.':
                    float(value[:-1])
                else:
                    float(value)
                    
                # 检查范围 (-10 ~ 10)
                if value != "-" and value != ".":
                    angle = float(value)
                    if angle < -10 or angle > 10:
                        return False
                return True
        except ValueError:
            pass
            
        return False
    
    def apply_angle_from_entry(self):
        """从输入框应用角度值"""
        try:
            angle_text = self.angle_entry.get()
            if angle_text:
                angle = float(angle_text)
                
                # 限制在-10到10度范围内
                angle = max(-10, min(10, angle))
                
                # 更新角度值
                self.angle_var.set(angle)
                self.update_rotation_value(angle)
                
                # 同步输入框内容
                self.angle_entry.delete(0, tk.END)
                self.angle_entry.insert(0, f"{angle:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Invalid angle value. Please enter a number between -10 and 10.")
    
    def toggle_edit_mode(self):
        """切换编辑模式"""
        self.edit_mode = self.edit_mode_var.get()
        mode = "Edit" if self.edit_mode else "Draw"
        self.status_var.set(f"Mode: {mode}")
    
    def show_instructions(self):
        """显示操作指南"""
        instructions = (
            "Instructions:\n"
            "1. Select input & output folders\n"
            "2. Load CSV reference file (optional)\n"
            "3. Draw rectangles by dragging mouse\n"
            "4. Enable 'Edit Mode' to move/resize rectangles\n"
            "5. Enter angle directly in the input box or use slider\n"
            "6. Use fine adjustment buttons for precise rotation\n"
            "7. Press 'Delete' key to delete selected box\n"
            "8. Press 's' to save annotations\n"
            "9. Use arrow keys or buttons to navigate\n"
            "10. Click on annotations list to select"
        )
        messagebox.showinfo("Instructions", instructions)
    
    def adjust_angle(self, delta):
        """精确调整角度"""
        if self.selected_rect_idx < 0 or self.selected_rect_idx >= len(self.annotations):
            return
            
        current = self.angle_var.get()
        new_angle = current + delta
        
        # 限制在-10到10度范围内
        new_angle = max(-10, min(10, new_angle))
        
        self.angle_var.set(new_angle)
        self.update_rotation_value(new_angle)
        
        # 同步输入框内容
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{new_angle:.2f}")
    
    def load_csv_reference(self):
        """加载CSV参考文件"""
        file_path = filedialog.askopenfilename(title="Select CSV Reference File", 
                                             filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
            
        try:
            self.csv_data = pd.read_csv(file_path)
            
            # 检查CSV是否包含必要的列
            if 'tcp_rz' not in self.csv_data.columns:
                raise ValueError("CSV file must contain 'tcp_rz' column")
                
            # 检查是否有图像名对应的列
            if 'image_name' not in self.csv_data.columns:
                # 尝试通过索引或行号关联
                self.status_var.set("Warning: No 'image_name' column found, will try to match by index")
            
            self.csv_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.status_var.set(f"CSV reference loaded: {len(self.csv_data)} entries")
            
            # 如果当前已加载图像，尝试应用参考角度
            if self.current_image is not None:
                current_file = self.image_files[self.current_index]
                ref_angle = self.check_csv_reference(current_file)
                if ref_angle is not None:
                    self.status_var.set(f"Reference angle from CSV: {ref_angle:.4f}°")
                    # 显示在角度输入框中
                    self.angle_entry.delete(0, tk.END)
                    self.angle_entry.insert(0, f"{ref_angle:.2f}")
            
        except Exception as e:
            self.csv_data = None
            self.csv_label.config(text="No CSV file loaded")
            self.status_var.set(f"Error loading CSV: {str(e)}")
            messagebox.showerror("CSV Load Error", str(e))
    
    def select_input_folder(self):
        """选择输入文件夹"""
        folder_path = filedialog.askdirectory(title="Select Input Image Folder")
        if not folder_path:
            return
        
        self.image_folder = folder_path
        self.input_label.config(text=self.image_folder)
        
        # 加载文件夹中的图像文件
        self.image_files = []
        image_files_temp = []
        for file in os.listdir(self.image_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files_temp.append(file)
        
        # 按照文件名中的数字进行排序
        def extract_number(filename):
            # 提取文件名中的数字部分
            import re
            numbers = re.findall(r'_(\d+)\.', filename)
            if numbers:
                return int(numbers[0])
            return 0  # 如果没有找到数字，则返回0
        
        # 按数字排序
        self.image_files = sorted(image_files_temp, key=extract_number)
        
        # 更新计数器
        self.counter_label.config(text=f"0 / {len(self.image_files)}")
        
        if self.image_files:
            self.current_index = 0
            self.load_current_image()
        else:
            self.status_var.set("No images found in the selected folder")
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if not folder_path:
            return
        
        self.output_folder = folder_path
        self.output_label.config(text=self.output_folder)
        
        # 确保输出文件夹及子文件夹存在
        self.ensure_output_folders()
    
    def ensure_output_folders(self):
        """确保输出文件夹及其子文件夹存在"""
        if not self.output_folder:
            return False
            
        # 确保主输出文件夹存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        # 确保images子文件夹存在
        images_dir = os.path.join(self.output_folder, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            
        # 确保annotations子文件夹存在
        annot_dir = os.path.join(self.output_folder, "annotations")
        if not os.path.exists(annot_dir):
            os.makedirs(annot_dir)
            
        return True
    
    def load_current_image(self):
        """加载当前索引的图像"""
        if not self.image_files or self.current_index < 0 or self.current_index >= len(self.image_files):
            return
        
        # 如果有未保存的标注，提示保存
        if self.annotations:
            if messagebox.askyesno("Save Annotations", "Save current annotations before loading next image?"):
                self.save_annotations()
        
        # 清除当前标注
        self.clear_annotations(False)  # 不需要确认
        
        # 加载图像
        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, current_file)
        
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not read image")
                
            # OpenCV读取的是BGR格式，转换为RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # 更新图像显示
            self.ax.clear()
            self.ax.imshow(self.current_image)
            self.ax.set_title(current_file)
            self.ax.axis('off')
            
            # 尝试从CSV加载旋转角度参考
            ref_angle = self.check_csv_reference(current_file)
            if ref_angle is not None:
                self.angle_entry.delete(0, tk.END)
                self.angle_entry.insert(0, f"{ref_angle:.2f}")
            
            # 加载已有标注（如果存在）
            self.load_existing_annotations(current_file)
            
            # 更新界面信息
            self.counter_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
            self.filename_label.config(text=current_file)
            self.status_var.set(f"Loaded {current_file}")
            
            self.canvas.draw()
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def check_csv_reference(self, image_filename):
        """从CSV文件中查找当前图像对应的旋转角度"""
        if self.csv_data is None:
            return None
            
        try:
            # 去除扩展名，以便匹配
            base_name = os.path.splitext(image_filename)[0]
            
            # 优先尝试通过image_name列匹配
            if 'image_name' in self.csv_data.columns:
                # 尝试精确匹配
                match = self.csv_data[self.csv_data['image_name'] == image_filename]
                
                # 如果没找到，尝试匹配不带扩展名的文件名
                if len(match) == 0:
                    match = self.csv_data[self.csv_data['image_name'] == base_name]
                
                if len(match) > 0:
                    angle = match['tcp_rz'].iloc[0]
                    self.status_var.set(f"Found reference angle: {angle:.4f}°")
                    # 这里将CSV中的角度作为参考返回
                    return angle
            
            # 如果通过名称未找到匹配，尝试使用索引匹配
            if self.current_index < len(self.csv_data):
                angle = self.csv_data['tcp_rz'].iloc[self.current_index]
                self.status_var.set(f"Using index match: reference angle = {angle:.4f}°")
                return angle
                
        except Exception as e:
            self.status_var.set(f"Error finding CSV reference: {str(e)}")
            
        return None
    
    def load_existing_annotations(self, image_filename):
        """加载已存在的标注文件"""
        if not self.output_folder:
            return
            
        base_name = os.path.splitext(image_filename)[0]
        annotation_file = os.path.join(self.output_folder, "annotations", f"{base_name}.txt")
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            x, y, w, h, theta = map(float, parts)
                            self.annotations.append((x, y, w, h, theta))
                
                # 更新标注显示
                self.update_annotations_display()
                self.status_var.set(f"Loaded {len(self.annotations)} annotations")
            except Exception as e:
                self.status_var.set(f"Error loading annotations: {str(e)}")
    
    def next_image(self):
        """切换到下一张图像"""
        if not self.image_files:
            return
            
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def prev_image(self):
        """切换到上一张图像"""
        if not self.image_files:
            return
            
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def update_rotation_value(self, value):
        """更新旋转角度值"""
        self.rotation = value
        self.angle_label.config(text=f"{value:.2f}°")
        
        # 更新输入框
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{value:.2f}")
        
        # 如果有选中的标注框，更新其旋转角度
        if self.selected_rect_idx >= 0 and self.selected_rect_idx < len(self.annotations):
            x, y, w, h, _ = self.annotations[self.selected_rect_idx]
            self.annotations[self.selected_rect_idx] = (x, y, w, h, value)
            self.update_annotations_display()
    
    def on_mouse_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes != self.ax or self.current_image is None:
            return
        
        if self.edit_mode:
            # 编辑模式下：检查是否点击了已有标注框
            self.handle_edit_press(event)
        else:
            # 绘制模式下：开始绘制新矩形
            self.drawing = True
            self.rect_start = (event.xdata, event.ydata)
            
            # 如果已经有一个临时矩形，删除它
            if self.current_rect is not None:
                self.current_rect.remove()
                self.current_rect = None
            
            # 获取当前CSV参考角度（如果有）
            ref_angle = None
            if self.csv_data is not None:
                current_file = self.image_files[self.current_index]
                ref_angle = self.check_csv_reference(current_file)
            
            # 设置旋转角度（如果有CSV参考则使用，否则为0）
            if ref_angle is not None and -10 <= ref_angle <= 10:
                self.rotation = ref_angle
                self.angle_var.set(ref_angle)
                self.angle_label.config(text=f"{ref_angle:.2f}°")
                self.angle_entry.delete(0, tk.END)
                self.angle_entry.insert(0, f"{ref_angle:.2f}")
            else:
                self.rotation = 0.0
                self.angle_var.set(0.0)
                self.angle_label.config(text="0.00°")
                self.angle_entry.delete(0, tk.END)
                self.angle_entry.insert(0, "0.00")
    
    def handle_edit_press(self, event):
        """在编辑模式下处理鼠标按下事件"""
        click_x, click_y = event.xdata, event.ydata
        
        # 检查是否点击了某个矩形或其控制点
        for i, (x, y, w, h, theta) in enumerate(self.annotations):
            # 矩形的中心和四个角点
            rect_x = x - w/2
            rect_y = y - h/2
            
            # 检查是否点击了矩形内部
            # 因为可能有旋转，所以转换点到矩形的局部坐标系
            # 这里简化处理，仅检查点击中心点附近区域
            if abs(click_x - x) < w/2 and abs(click_y - y) < h/2:
                self.selected_rect_idx = i
                self.edit_rect_idx = i
                self.drag_start = (click_x, click_y)
                self.resize_handle = None  # 移动整个矩形
                
                # 更新显示
                self.update_annotations_display()
                
                # 更新角度控制
                theta = self.annotations[i][4]
                self.angle_var.set(theta)
                self.angle_label.config(text=f"{theta:.2f}°")
                self.angle_entry.delete(0, tk.END)
                self.angle_entry.insert(0, f"{theta:.2f}")
                
                self.status_var.set(f"Selected annotation #{i+1} for moving")
                return
                
            # 检查八个控制点（四个角点和四条边的中点）
            # 为简化代码，仅计算角点的位置
            corners = [
                (rect_x, rect_y),                  # 左上
                (rect_x + w, rect_y),              # 右上
                (rect_x + w, rect_y + h),          # 右下
                (rect_x, rect_y + h),              # 左下
            ]
            
            # 应用旋转
            rotation_matrix = np.array([
                [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
            ])
            
            rotated_corners = []
            for cx, cy in corners:
                # 将角点移动到以矩形中心为原点的坐标系
                dx = cx - x
                dy = cy - y
                
                # 应用旋转
                rotated_x, rotated_y = np.dot(rotation_matrix, [dx, dy])
                
                # 移回世界坐标系
                rotated_corners.append((rotated_x + x, rotated_y + y))
            
            # 检查是否点击了角点
            for j, (cx, cy) in enumerate(rotated_corners):
                if abs(click_x - cx) < 5 and abs(click_y - cy) < 5:
                    self.selected_rect_idx = i
                    self.edit_rect_idx = i
                    self.drag_start = (click_x, click_y)
                    self.resize_handle = j  # 记录控制点索引
                    
                    # 更新显示
                    self.update_annotations_display()
                    
                    self.status_var.set(f"Resizing annotation #{i+1}")
                    return
    
    def on_mouse_move(self, event):
        """鼠标移动事件处理"""
        if event.inaxes != self.ax:
            return
        
        if self.edit_mode and self.edit_rect_idx >= 0 and self.drag_start is not None:
            # 编辑模式下：移动或调整矩形大小
            self.handle_edit_move(event)
        elif not self.edit_mode and self.drawing:
            # 绘制模式下：更新临时矩形
            self.rect_current = (event.xdata, event.ydata)
            self.draw_temp_rectangle()
    
    def handle_edit_move(self, event):
        """在编辑模式下处理鼠标移动事件"""
        move_x, move_y = event.xdata, event.ydata
        
        if self.edit_rect_idx < 0 or self.edit_rect_idx >= len(self.annotations):
            return
            
        # 获取当前矩形参数
        x, y, w, h, theta = self.annotations[self.edit_rect_idx]
        
        # 计算拖动距离
        dx = move_x - self.drag_start[0]
        dy = move_y - self.drag_start[1]
        
        if self.resize_handle is None:
            # 移动整个矩形
            new_x = x + dx
            new_y = y + dy
            
            # 更新矩形
            self.annotations[self.edit_rect_idx] = (new_x, new_y, w, h, theta)
            
            # 更新拖动起点
            self.drag_start = (move_x, move_y)
        else:
            # 调整大小 - 复杂操作，需要考虑旋转
            # 为简化处理，这里只实现基本的大小调整
            # 根据控制点的索引调整宽度和高度
            
            # 将鼠标点转换到矩形局部坐标系
            rotation_matrix = np.array([
                [np.cos(np.radians(-theta)), -np.sin(np.radians(-theta))],
                [np.sin(np.radians(-theta)), np.cos(np.radians(-theta))]
            ])
            
            # 当前鼠标位置相对于矩形中心的向量
            rel_x = move_x - x
            rel_y = move_y - y
            
            # 旋转到矩形的局部坐标系
            local_x, local_y = np.dot(rotation_matrix, [rel_x, rel_y])
            
            # 根据控制点索引调整矩形大小
            if self.resize_handle == 0:  # 左上角
                new_w = max(5, w - 2 * local_x)
                new_h = max(5, h - 2 * local_y)
                new_x = x + (w - new_w) / 2 * np.cos(np.radians(theta)) - (h - new_h) / 2 * np.sin(np.radians(theta))
                new_y = y + (w - new_w) / 2 * np.sin(np.radians(theta)) + (h - new_h) / 2 * np.cos(np.radians(theta))
            elif self.resize_handle == 1:  # 右上角
                new_w = max(5, w + 2 * local_x)
                new_h = max(5, h - 2 * local_y)
                new_x = x + (w - new_w) / 2 * np.cos(np.radians(theta)) - (h - new_h) / 2 * np.sin(np.radians(theta))
                new_y = y + (w - new_w) / 2 * np.sin(np.radians(theta)) + (h - new_h) / 2 * np.cos(np.radians(theta))
            elif self.resize_handle == 2:  # 右下角
                new_w = max(5, w + 2 * local_x)
                new_h = max(5, h + 2 * local_y)
                new_x = x
                new_y = y
            elif self.resize_handle == 3:  # 左下角
                new_w = max(5, w - 2 * local_x)
                new_h = max(5, h + 2 * local_y)
                new_x = x + (w - new_w) / 2 * np.cos(np.radians(theta)) + (h - new_h) / 2 * np.sin(np.radians(theta))
                new_y = y + (w - new_w) / 2 * np.sin(np.radians(theta)) - (h - new_h) / 2 * np.cos(np.radians(theta))
            
            # 更新矩形
            self.annotations[self.edit_rect_idx] = (new_x, new_y, new_w, new_h, theta)
        
        # 更新显示
        self.update_annotations_display()
    
    def on_mouse_release(self, event):
        """鼠标释放事件处理"""
        if self.edit_mode:
            # 编辑模式下：结束拖动操作
            self.drag_start = None
            self.resize_handle = None
            self.edit_rect_idx = -1
            return
            
        if not self.drawing:
            return
        
        self.drawing = False
        
        # 确保是在图像区域内释放
        if event.inaxes != self.ax:
            # 清除临时矩形
            if self.current_rect is not None:
                self.current_rect.remove()
                self.current_rect = None
            self.canvas.draw()
            return
        
        # 获取终点坐标
        end_point = (event.xdata, event.ydata)
        
        # 计算矩形参数 (x, y, w, h)，其中(x,y)是中心点
        x = (self.rect_start[0] + end_point[0]) / 2
        y = (self.rect_start[1] + end_point[1]) / 2
        w = abs(end_point[0] - self.rect_start[0])
        h = abs(end_point[1] - self.rect_start[1])
        
        # 如果矩形太小则忽略
        if w < 5 or h < 5:
            # 清除临时矩形
            if self.current_rect is not None:
                self.current_rect.remove()
                self.current_rect = None
            self.canvas.draw()
            return
        
        # 获取当前的旋转角度（已在mouse_press中设置）
        theta = self.rotation
        
        # 添加新标注 (x, y, w, h, theta)
        self.annotations.append((x, y, w, h, theta))
        
        # 更新标注框显示
        self.update_annotations_display()
        
        # 清除临时矩形
        if self.current_rect is not None:
            self.current_rect.remove()
            self.current_rect = None
        
        # 设置当前选中的标注为新添加的
        self.selected_rect_idx = len(self.annotations) - 1
        self.annot_listbox.selection_clear(0, tk.END)
        self.annot_listbox.selection_set(self.selected_rect_idx)
        
        # 更新状态
        self.status_var.set(f"Added annotation #{len(self.annotations)}")
    
    # 修改draw_temp_rectangle方法

    def draw_temp_rectangle(self):
        """绘制临时矩形"""
        if not self.rect_start or not self.rect_current:
            return
        
        # 如果已有临时矩形，则删除
        if self.current_rect is not None:
            self.current_rect.remove()
            self.current_rect = None
        
        # 计算矩形参数
        x = min(self.rect_start[0], self.rect_current[0])
        y = min(self.rect_start[1], self.rect_current[1])
        w = abs(self.rect_current[0] - self.rect_start[0])
        h = abs(self.rect_current[1] - self.rect_start[1])
        
        # 创建矩形
        self.current_rect = Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        self.ax.add_patch(self.current_rect)
        
        # 强制更新画布
        self.fig.canvas.draw()  # 完整刷新
        self.fig.canvas.flush_events()  # 确保事件处理完成

        # 在主循环中调度更新，确保界面响应
        self.root.update_idletasks()
    
    def update_annotations_display(self):
        """更新标注框显示"""
        # 清除当前图像，重绘基础图像
        self.ax.clear()
        if self.current_image is not None:
            self.ax.imshow(self.current_image)
            current_file = self.image_files[self.current_index]
            self.ax.set_title(current_file)
        self.ax.axis('off')
        
        # 绘制所有标注框
        for i, (x, y, w, h, theta) in enumerate(self.annotations):
            # 创建矩形，中心点在(x,y)
            rect_x = x - w/2
            rect_y = y - h/2
            
            # 创建矩形
            box = Rectangle((rect_x, rect_y), w, h, 
                          fill=False, 
                          edgecolor='green' if i != self.selected_rect_idx else 'red',
                          linewidth=2)
            
            # 应用旋转变换
            transform = Affine2D().rotate_deg_around(x, y, theta) + self.ax.transData
            box.set_transform(transform)
            
            # 添加到图像
            self.ax.add_patch(box)
            
            # 如果处于编辑模式，添加控制点
            if self.edit_mode and i == self.selected_rect_idx:
                self.draw_control_points(x, y, w, h, theta)
        
        # 更新ListBox
        self.annot_listbox.delete(0, tk.END)
        for i, (x, y, w, h, theta) in enumerate(self.annotations):
            self.annot_listbox.insert(tk.END, f"#{i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, θ={theta:.2f}°")
        
        # 如果有选中的项，确保它被高亮
        if self.selected_rect_idx >= 0:
            self.annot_listbox.selection_clear(0, tk.END)
            self.annot_listbox.selection_set(self.selected_rect_idx)
        
        # 更新画布
        self.canvas.draw()
    
    def draw_control_points(self, x, y, w, h, theta):
        """绘制控制点"""
        # 矩形四个角点
        rect_x = x - w/2
        rect_y = y - h/2
        
        corners = [
            (rect_x, rect_y),              # 左上
            (rect_x + w, rect_y),          # 右上
            (rect_x + w, rect_y + h),      # 右下
            (rect_x, rect_y + h),          # 左下
        ]
        
        # 应用旋转
        rotation_matrix = np.array([
            [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
            [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
        ])
        
        for cx, cy in corners:
            # 将角点移动到以矩形中心为原点的坐标系
            dx = cx - x
            dy = cy - y
            
            # 应用旋转
            rotated_x, rotated_y = np.dot(rotation_matrix, [dx, dy])
            
            # 移回世界坐标系
            final_x = rotated_x + x
            final_y = rotated_y + y
            
            # 绘制控制点
            self.ax.plot(final_x, final_y, 'ro', markersize=5)
    
    def on_key_press(self, event):
        """键盘按键事件处理"""
        if self.current_image is None:
            return
            
        if event.key == 'right' or event.key == 'n':
            self.next_image()
        elif event.key == 'left' or event.key == 'p':
            self.prev_image()
        elif event.key == 'delete':
            self.delete_selected_annotation()
        elif event.key == 'r':  # 逆时针旋转
            if self.selected_rect_idx >= 0:
                new_angle = self.angle_var.get() - 0.1
                if new_angle < -10:
                    new_angle = -10
                self.angle_var.set(new_angle)
                self.update_rotation_value(new_angle)
        elif event.key == 'R':  # 顺时针旋转
            if self.selected_rect_idx >= 0:
                new_angle = self.angle_var.get() + 0.1
                if new_angle > 10:
                    new_angle = 10
                self.angle_var.set(new_angle)
                self.update_rotation_value(new_angle)
        elif event.key == 's':  # 保存标注
            self.save_annotations()
        elif event.key == 'e':  # 快捷切换编辑模式
            self.edit_mode_var.set(not self.edit_mode_var.get())
            self.toggle_edit_mode()
    
    def on_annotation_selected(self, event):
        """标注框被选中的事件处理"""
        if not self.annot_listbox.curselection():
            self.selected_rect_idx = -1
            return
            
        # 获取选中的索引
        self.selected_rect_idx = self.annot_listbox.curselection()[0]
        
        # 更新角度滑块和参数输入框
        if self.selected_rect_idx >= 0 and self.selected_rect_idx < len(self.annotations):
            x, y, w, h, theta = self.annotations[self.selected_rect_idx]
            
            # 更新角度控制
            self.angle_var.set(theta)
            self.angle_label.config(text=f"{theta:.2f}°")
            self.angle_entry.delete(0, tk.END)
            self.angle_entry.insert(0, f"{theta:.2f}")
            
            # 更新矩形参数输入框
            self.x_entry.delete(0, tk.END)
            self.x_entry.insert(0, f"{x:.1f}")
            
            self.y_entry.delete(0, tk.END)
            self.y_entry.insert(0, f"{y:.1f}")
            
            self.w_entry.delete(0, tk.END)
            self.w_entry.insert(0, f"{w:.1f}")
            
            self.h_entry.delete(0, tk.END)
            self.h_entry.insert(0, f"{h:.1f}")
            
            # 状态栏更新
            self.status_var.set(f"Selected annotation #{self.selected_rect_idx+1}")
        
        # 更新显示
        self.update_annotations_display()
    
    def delete_selected_annotation(self):
        """删除选中的标注框"""
        if self.selected_rect_idx < 0 or self.selected_rect_idx >= len(self.annotations):
            return
            
        # 删除标注
        del self.annotations[self.selected_rect_idx]
        
        # 更新显示
        self.selected_rect_idx = -1
        self.update_annotations_display()
        self.status_var.set("Annotation deleted")
    
    def clear_annotations(self, confirm=True):
        """清除所有标注"""
        if confirm and self.annotations:
            if not messagebox.askyesno("Confirm", "Clear all annotations?"):
                return
        
        self.annotations = []
        self.selected_rect_idx = -1
        self.update_annotations_display()
        self.status_var.set("All annotations cleared")
    
    def save_annotations(self):
        """保存当前图像的标注"""
        if self.current_image is None:
            self.status_var.set("No image loaded")
            return
            
        if not self.output_folder:
            self.status_var.set("Output folder not set")
            messagebox.showwarning("Warning", "Please select an output folder first")
            return
            
        if not self.annotations:
            self.status_var.set("No annotations to save")
            return
            
        # 确保输出文件夹存在
        if not self.ensure_output_folders():
            messagebox.showerror("Error", "Failed to create output folders")
            return
            
        try:
            # 获取当前文件名（不含扩展名）
            current_file = self.image_files[self.current_index]
            base_name = os.path.splitext(current_file)[0]
            
            # 保存标注到文本文件
            annot_path = os.path.join(self.output_folder, "annotations", f"{base_name}.txt")
            with open(annot_path, 'w') as f:
                for x, y, w, h, theta in self.annotations:
                    f.write(f"{x} {y} {w} {h} {theta}\n")
            
            # 保存带标注的图像
            img_with_boxes = self.current_image.copy()
            
            # 在图像上绘制所有旋转矩形
            for x, y, w, h, theta in self.annotations:
                # OpenCV的rotatedRect需要中心点、尺寸和角度
                center = (int(x), int(y))
                size = (int(w), int(h))
                
                # 创建旋转矩形
                rect = ((center[0], center[1]), (size[0], size[1]), theta)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # BGR格式的红色
                cv2.drawContours(img_with_boxes, [box], 0, (255, 0, 0), 2)
            
            # 将RGB转回BGR用于保存
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            
            # 保存图像
            img_path = os.path.join(self.output_folder, "images", current_file)
            cv2.imwrite(img_path, img_with_boxes)
            
            # 更新状态
            self.status_var.set(f"Saved {len(self.annotations)} annotations to {annot_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving annotations: {str(e)}")
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.update()  # 在创建应用前先更新一次root窗口
    app = RotatedBoxAnnotator(root)
    root.mainloop()