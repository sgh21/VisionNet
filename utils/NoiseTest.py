import tkinter as tk
from tkinter import ttk, filedialog
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import numpy as np
import cv2

class NoiseTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Noise Test Visualization")
        # 设置最小窗口大小
        self.root.minsize(1000, 600)
        # 创建控件
        self.create_widgets()
        
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.noised_image = None
        
    def create_widgets(self):
        # 图片显示框
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(pady=20, expand=True)
        
        self.label_original = ttk.Label(self.frame_images, text="Original Image")
        self.label_original.grid(row=0, column=0)
        # 增大画布尺寸
        self.canvas_original = tk.Canvas(self.frame_images, width=448, height=448)
        self.canvas_original.grid(row=1, column=0, padx=10)
        
        self.label_noised = ttk.Label(self.frame_images, text="Noised Image")
        self.label_noised.grid(row=0, column=1)
        self.canvas_noised = tk.Canvas(self.frame_images, width=448, height=448)
        self.canvas_noised.grid(row=1, column=1, padx=10)
        
        # 控制面板
        self.frame_control = ttk.Frame(self.root)
        self.frame_control.pack(pady=20)
        

        
        ttk.Button(self.frame_control, text="Load Image", 
                  command=self.load_image).pack(pady=5)
        
        # 增大滑动条
        self.noise_scale = ttk.Scale(self.frame_control, from_=0, to=1,
                                   orient="horizontal", length=400,
                                   command=self.update_noise)
        self.noise_scale.set(0.1)
        self.noise_scale.pack(pady=5)
        
        self.noise_label = ttk.Label(self.frame_control, text="Noise Level: 0.1")
        self.noise_label.pack(pady=5)
    
    def add_radial_noise(self, image_tensor, max_noise):
        """添加径向噪声"""
        H, W = image_tensor.shape[-2:]
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        xx, yy = torch.meshgrid(x, y)
        distance = torch.sqrt(xx*xx + yy*yy)
        distance = distance / distance.max()
        noise = torch.randn(3, H, W) * distance[None, :, :] * max_noise
        return torch.clamp(image_tensor + noise, 0, 1)
    
    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        return transforms.ToPILImage()(tensor)
    
    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            # 加载和预处理图像
            image = Image.open(self.image_path)
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ])
            self.original_tensor = transform(image)
            
            # 显示原始图像
            self.original_image = ImageTk.PhotoImage(
                self.tensor_to_pil(self.original_tensor))
            self.canvas_original.create_image(
                224, 224, image=self.original_image)
            
            # 更新噪声图像
            self.update_noise(self.noise_scale.get())
    
    def update_noise(self, value):
        if self.original_tensor is not None:
            noise_level = float(value)
            self.noise_label.config(text=f"Noise Level: {noise_level:.2f}")
            
            # 添加噪声
            noised_tensor = self.add_radial_noise(
                self.original_tensor, noise_level)
            
            # 更新显示
            self.noised_image = ImageTk.PhotoImage(
                self.tensor_to_pil(noised_tensor))
            self.canvas_noised.create_image(
                224, 224, image=self.noised_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseTestApp(root)
    root.mainloop()