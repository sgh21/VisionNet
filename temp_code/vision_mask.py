import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from functools import partial

class ImageOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片叠加显示器")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.folder1 = ""
        self.folder2 = ""
        self.image_pairs = []  # 存储匹配的图片对
        self.current_index = 0
        self.alpha = 0.5  # 透明度默认值
        self.current_blended = None
        self.img1 = None
        self.img2 = None
        
        # 创建界面
        self.create_ui()
        
    def create_ui(self):
        # 顶部控制区域
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 文件夹选择区域
        ttk.Button(control_frame, text="选择第一个文件夹", command=lambda: self.select_folder(1)).grid(row=0, column=0, padx=5, pady=5)
        self.lbl_folder1 = ttk.Label(control_frame, text="未选择文件夹")
        self.lbl_folder1.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(control_frame, text="选择第二个文件夹", command=lambda: self.select_folder(2)).grid(row=0, column=2, padx=5, pady=5)
        self.lbl_folder2 = ttk.Label(control_frame, text="未选择文件夹")
        self.lbl_folder2.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # 主内容区域
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左右布局
        main_frame.columnconfigure(1, weight=3)  # 图片显示区域占更多空间
        
        # 左边：图片列表
        list_frame = ttk.Frame(main_frame, padding="10")
        list_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(list_frame, text="匹配的图片:").pack(anchor="w")
        
        # 创建带滚动条的列表框
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 右边：图片显示区域
        display_frame = ttk.Frame(main_frame, padding="10")
        display_frame.grid(row=0, column=1, sticky="nsew")
        
        # 图片信息标签
        self.info_label = ttk.Label(display_frame, text="请选择两个文件夹以查找匹配图片")
        self.info_label.pack(anchor="w")
        
        # 图片显示区域
        self.canvas_frame = ttk.Frame(display_frame, borderwidth=2, relief="sunken")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 透明度控制区域
        control_panel = ttk.Frame(display_frame)
        control_panel.pack(fill=tk.X, pady=5)
        
        # 透明度滑块
        ttk.Label(control_panel, text="透明度:").grid(row=0, column=0, padx=5)
        self.opacity_var = tk.DoubleVar(value=50)
        self.opacity_slider = ttk.Scale(control_panel, from_=0, to=100, 
                                        orient="horizontal", variable=self.opacity_var, 
                                        command=self.update_overlay)
        self.opacity_slider.grid(row=0, column=1, padx=5, sticky="ew")
        
        self.opacity_label = ttk.Label(control_panel, text="50%")
        self.opacity_label.grid(row=0, column=2, padx=5)
        
        # 叠加模式
        ttk.Label(control_panel, text="叠加模式:").grid(row=1, column=0, padx=5, pady=5)
        self.blend_mode_var = tk.StringVar(value="普通")
        blend_modes = ["普通", "加法", "减法", "乘法", "屏幕", "叠加"]
        blend_mode_combo = ttk.Combobox(control_panel, textvariable=self.blend_mode_var, 
                                         values=blend_modes, state="readonly", width=10)
        blend_mode_combo.grid(row=1, column=1, padx=5, sticky="w", pady=5)
        blend_mode_combo.bind("<<ComboboxSelected>>", self.update_overlay)
        
        # 按钮区域
        button_panel = ttk.Frame(control_panel)
        button_panel.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_panel, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_panel, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_panel, text="保存当前叠加图", command=self.save_overlay_image).pack(side=tk.LEFT, padx=5)
        
        # 让控制面板中的滑块可以伸展
        control_panel.columnconfigure(1, weight=1)
        
    def select_folder(self, folder_num):
        folder = filedialog.askdirectory(title=f"选择文件夹 {folder_num}")
        
        if folder:
            if folder_num == 1:
                self.folder1 = folder
                self.lbl_folder1.config(text=folder)
            else:
                self.folder2 = folder
                self.lbl_folder2.config(text=folder)
            
            # 如果两个文件夹都已选择，找出匹配的图片
            if self.folder1 and self.folder2:
                self.find_matching_images()
    def find_matching_images(self):
        """查找两个文件夹中的匹配图片，支持字段匹配"""
        # 获取两个文件夹中的所有图片文件
        files1 = self.get_image_files(self.folder1)
        files2 = self.get_image_files(self.folder2)
        
        # 提取文件名（不含扩展名）
        file_dict1 = {os.path.splitext(os.path.basename(f))[0]: f for f in files1}
        file_dict2 = {os.path.splitext(os.path.basename(f))[0]: f for f in files2}
        
        # 尝试精确匹配
        exact_matches = []
        for name1 in file_dict1:
            if name1 in file_dict2:
                exact_matches.append((name1, file_dict1[name1], file_dict2[name1]))
        
        # 如果找到足够的精确匹配
        if len(exact_matches) >= 3:
            self.image_pairs = exact_matches
            self.update_image_list()
            
            # 显示第一张匹配的图片（如果有）
            if self.image_pairs:
                self.current_index = 0
                self.display_current_image()
            return
        
        # 如果精确匹配不够，尝试字段匹配
        # 步骤1: 将所有文件名按字段分割，准备匹配
        parsed_files1 = []
        parsed_files2 = []
        
        # 解析文件1
        for name, path in file_dict1.items():
            fields = self._parse_filename(name)
            parsed_files1.append((name, path, fields))
        
        # 解析文件2
        for name, path in file_dict2.items():
            fields = self._parse_filename(name)
            parsed_files2.append((name, path, fields))
        
        # 步骤2: 为每个文件1中的文件找最佳匹配
        self.image_pairs = []
        for name1, path1, fields1 in parsed_files1:
            best_match = None
            best_score = 0
            
            for name2, path2, fields2 in parsed_files2:
                # 计算字段匹配得分
                score = self._calculate_field_similarity(fields1, fields2)
                
                if score > best_score:
                    best_score = score
                    best_match = (f"{name1} & {name2}", path1, path2, score)
            
            # 只添加得分高于阈值的匹配
            if best_match and best_match[3] > 0.3:  # 阈值可调整
                self.image_pairs.append(best_match[:3])  # 只保留名称和路径
        
        # 更新列表显示
        self.update_image_list()
        
        # 显示第一张匹配的图片（如果有）
        if self.image_pairs:
            self.current_index = 0
            self.display_current_image()
            messagebox.showinfo("匹配结果", f"找到 {len(self.image_pairs)} 对匹配的图片")
        else:
            messagebox.showinfo("结果", "未找到匹配的图片")

    def _parse_filename(self, filename):
        """解析文件名为字段列表，处理常见前缀和分隔符"""
        # 移除常见前缀
        for prefix in ['gel_', 'mask_', 'img_', 'proc_']:
            if filename.startswith(prefix):
                filename = filename[len(prefix):]
                break
        
        # 按分隔符分割
        fields = []
        # 先按下划线分割
        parts = filename.split('_')
        
        for part in parts:
            # 再按破折号分割
            subparts = part.split('-')
            fields.extend(subparts)
        
        # 提取纯数字字段
        numeric_fields = []
        for field in fields:
            if field.isdigit():
                numeric_fields.append(field)
        
        # 如果有数字字段，优先返回数字字段
        if numeric_fields:
            return numeric_fields
        else:
            return fields

    def _calculate_field_similarity(self, fields1, fields2):
        """计算两组字段之间的相似度"""
        if not fields1 or not fields2:
            return 0.0
        
        # 特殊情况处理：如果两个字段组完全相同
        if fields1 == fields2:
            return 1.0
        
        # 计算共有字段数
        common_fields = set(fields1).intersection(set(fields2))
        num_common = len(common_fields)
        
        # 如果有共同字段，根据共同字段与总字段的比例计算相似度
        if num_common > 0:
            total_unique_fields = len(set(fields1).union(set(fields2)))
            similarity = num_common / total_unique_fields
            
            # 为数字字段匹配给予额外权重
            # 如果都是纯数字字段
            if all(f.isdigit() for f in fields1) and all(f.isdigit() for f in fields2):
                digit_similarity = 0
                
                # 尝试匹配相同位置的数字
                min_len = min(len(fields1), len(fields2))
                matches = 0
                for i in range(min_len):
                    if fields1[i] == fields2[i]:
                        matches += 1
                
                if min_len > 0:
                    digit_similarity = matches / min_len
                    
                # 结合基本相似度和数字相似度
                similarity = (similarity + digit_similarity) / 2
            
            return similarity
        
        # 如果没有共同字段，尝试字符串相似度
        # 把所有字段连接起来比较
        str1 = ''.join(fields1)
        str2 = ''.join(fields2)
        
        # 简单的字符串相似度计算
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 0.0
        
        # 计算最长公共子串长度
        # 这里采用简化版本，实际可以用更复杂的算法
        common_length = 0
        for i in range(len(str1)):
            for j in range(len(str2)):
                k = 0
                while (i+k < len(str1) and j+k < len(str2) and 
                    str1[i+k] == str2[j+k]):
                    k += 1
                common_length = max(common_length, k)
        
        return common_length / max_len
    # def find_matching_images(self):
    #     # 获取两个文件夹中的所有图片文件
    #     files1 = self.get_image_files(self.folder1)
    #     files2 = self.get_image_files(self.folder2)
        
    #     # 提取文件名（不含扩展名）
    #     file_dict1 = {os.path.splitext(os.path.basename(f))[0]: f for f in files1}
    #     file_dict2 = {os.path.splitext(os.path.basename(f))[0]: f for f in files2}
        
    #     # 找出匹配的文件
    #     self.image_pairs = []
    #     for name in file_dict1:
    #         if name in file_dict2:
    #             self.image_pairs.append((name, file_dict1[name], file_dict2[name]))
        
    #     # 更新列表显示
    #     self.update_image_list()
        
    #     # 显示第一张匹配的图片（如果有）
    #     if self.image_pairs:
    #         self.current_index = 0
    #         self.display_current_image()
    #     else:
    #         messagebox.showinfo("结果", "未找到匹配的图片")
    
    def get_image_files(self, folder):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        files = []
        
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def update_image_list(self):
        # 清空列表
        self.image_listbox.delete(0, tk.END)
        
        # 添加匹配的图片名
        for name, _, _ in self.image_pairs:
            self.image_listbox.insert(tk.END, name)
        
        # 更新窗口标题
        self.root.title(f"图片叠加显示器 - 找到 {len(self.image_pairs)} 对匹配图片")
    
    def on_image_select(self, event):
        # 获取所选项的索引
        selection = self.image_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.display_current_image()
    
    def display_current_image(self):
        if not self.image_pairs:
            return
        
        # 获取当前图片对
        name, img1_path, img2_path = self.image_pairs[self.current_index]
        
        # 显示图片名称
        self.info_label.config(text=f"当前图片: {name} ({self.current_index+1}/{len(self.image_pairs)})")
        
        # 读取图片
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        
        # 如果任一图片无法读取，显示错误并返回
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("错误", f"无法读取图片:\n{img1_path}\n或\n{img2_path}")
            return
        
        # 调整图片2的大小以匹配图片1
        if self.img1.shape != self.img2.shape:
            self.img2 = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))
        
        # 更新叠加图像
        self.update_overlay()
    
    def update_overlay(self, *args):
        if not hasattr(self, 'img1') or self.img1 is None or self.img2 is None:
            return
        
        # 获取透明度值
        alpha = self.opacity_var.get() / 100.0
        self.alpha = alpha
        self.opacity_label.config(text=f"{int(alpha*100)}%")
        
        # 根据选择的混合模式执行不同的混合操作
        blend_mode = self.blend_mode_var.get()
        
        if blend_mode == "普通":  # 普通混合
            blended = cv2.addWeighted(self.img1, 1-alpha, self.img2, alpha, 0)
        elif blend_mode == "加法":  # 加法
            blended = cv2.add(cv2.multiply(self.img1, 1-alpha), cv2.multiply(self.img2, alpha))
        elif blend_mode == "减法":  # 减法
            blended = cv2.subtract(self.img1, cv2.multiply(self.img2, alpha))
        elif blend_mode == "乘法":  # 乘法
            # 归一化后相乘
            blended = cv2.multiply(self.img1/255.0, self.img2/255.0)
            blended = (blended * 255).astype(np.uint8)
        elif blend_mode == "屏幕":  # 屏幕模式
            # 屏幕模式：1 - (1-a)*(1-b)
            inv1 = 1.0 - self.img1/255.0
            inv2 = 1.0 - self.img2/255.0
            blended = (1.0 - inv1 * inv2) * 255
            blended = blended.astype(np.uint8)
        elif blend_mode == "叠加":  # 叠加模式
            # 简单实现叠加混合
            blended = np.zeros_like(self.img1)
            for i in range(3):  # 对每个通道
                blended[:,:,i] = np.where(
                    self.img1[:,:,i] < 128,
                    (self.img1[:,:,i] * self.img2[:,:,i]) // 128,
                    255 - ((255 - self.img1[:,:,i]) * (255 - self.img2[:,:,i])) // 128
                )
        
        # 保存当前混合结果用于保存功能
        self.current_blended = blended
        
        # 将OpenCV BGR格式转换为RGB
        rgb_image = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像以显示在Tkinter上
        pil_img = Image.fromarray(rgb_image)
        
        # 调整图片大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 确保画布大小有效
        if canvas_width <= 1 or canvas_height <= 1:
            # 给定一个默认大小
            canvas_width = 800
            canvas_height = 600
        
        # 计算缩放比例
        img_width, img_height = pil_img.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h)
        
        # 如果图片比画布小，不进行缩放
        if scale > 1:
            scale = 1
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        if scale != 1:
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为PhotoImage用于Tkinter
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        # 清除画布并显示新图片
        self.canvas.delete("all")
        
        # 计算图片在画布中的位置（居中）
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_img)
        
        # 更新画布大小
        self.canvas.config(width=canvas_width, height=canvas_height)
    
    def prev_image(self):
        if self.image_pairs:
            self.current_index = (self.current_index - 1) % len(self.image_pairs)
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self.display_current_image()
    
    def next_image(self):
        if self.image_pairs:
            self.current_index = (self.current_index + 1) % len(self.image_pairs)
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self.display_current_image()
    
    def save_overlay_image(self):
        if not hasattr(self, 'current_blended') or self.current_blended is None:
            messagebox.showerror("错误", "没有可保存的叠加图像")
            return
        
        # 获取当前图片名称
        if self.image_pairs:
            name = self.image_pairs[self.current_index][0]
            default_name = f"{name}_overlay.png"
        else:
            default_name = "overlay.png"
        
        # 打开文件保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")],
            initialfile=default_name
        )
        
        if file_path:
            # 保存图像
            try:
                cv2.imwrite(file_path, self.current_blended)
                messagebox.showinfo("成功", f"叠加图像已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存图像: {str(e)}")
                
    def run(self):
        # 设置窗口的响应函数，当窗口大小改变时更新图像
        self.root.bind("<Configure>", lambda e: self.update_overlay() if e.widget == self.root else None)
        
        # 更好的主题支持（如果可用）
        try:
            self.root.tk.call("source", "azure.tcl")
            self.root.tk.call("set_theme", "light")
        except:
            pass
        
        # 启动主循环
        self.root.mainloop()


def main():
    root = tk.Tk()
    app = ImageOverlayApp(root)
    app.run()

if __name__ == "__main__":
    main()