import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

def rotate_images(root_path:str):
    """
    查找根目录下的所有png文件，并旋转180度

    Args:
        root_path (str): 根目录路径
    """
    # 递归查找所有png文件
    png_files = glob(os.path.join(root_path, '**/*.png'), recursive=True)
    
    # 使用tqdm创建进度条
    for file_path in tqdm(png_files, desc="Rotating images", unit="file"):
        try:
            # 加载图片
            img = Image.open(file_path)
            
            # 旋转180度
            rotated_img = img.rotate(180)
            
            # 保存回原路径
            rotated_img.save(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def mean_std_statistics(root_path:str):
    """
    先将图片从0-255转为0-1，再计算根目录下所有png图片的均值和方差

    Args:
        root_path (str): 根目录路径
    """
    # 查找所有png文件
    png_files = glob(os.path.join(root_path, '**/*.png'), recursive=True)
    
    # 存储所有图片的均值和标准差
    means = []
    stds = []
    
    # 遍历处理每个文件
    for file_path in tqdm(png_files, desc="Computing statistics"):
        try:
            # 读取图片并转换为numpy数组
            img = np.array(Image.open(file_path)).astype(np.float32)
            
            # 归一化到0-1
            img /= 255.0
            
            # 计算均值和标准差(对每个通道)
            mean = img.mean(axis=(0,1))
            std = img.std(axis=(0,1))
            
            means.append(mean)
            stds.append(std)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            
    # 计算所有图片的总体均值和标准差
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    return mean, std

def show_image(root_path:str, normalize:bool=True, mean:list[float]=None, std:list[float]=None):
    """
    显示根目录下所有图片

    Args:
        root_path (str): 根目录路径
        normalize (bool, optional): 是否归一化到0-1. Defaults to True.
        mean (list[float], optional): 均值. Defaults to None.
        std (list[float], optional): 标准差. Defaults to None.
    """
    # 查找所有png文件
    png_files = glob(os.path.join(root_path, '**/*.png'), recursive=True)
    
    # 遍历处理每个文件
    for file_path in tqdm(png_files, desc="Showing images"):
        try:
            # 读取图片并转换为numpy数组
            img = np.array(Image.open(file_path)).astype(np.float32)
            
            # 归一化到0-1
            if normalize:
                img /= 255.0
                mean = np.array(mean, dtype=np.float32)
                std = np.array(std, dtype=np.float32)
                img = (img - mean) / std
            
                # 反标准化用于显示
                # img = img * std + mean
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            # 显示图片
            plt.imshow(img)
            plt.show()
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def add_radial_noise(image, max_noise=0.1):
    """添加径向噪声，边缘噪声大,中心噪声小
    Args:
        image: [3, H, W] tensor
        max_noise: 最大噪声强度
    """
    H, W = image.shape[-2:]
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    xx, yy = torch.meshgrid(x, y)
    # 计算到中心的距离
    distance = torch.sqrt(xx*xx + yy*yy)
    # 将距离归一化到[0,1]
    distance = distance / distance.max()
    # 生成噪声
    noise = torch.randn(3, H, W) * distance[None, :, :] * max_noise
    noise = noise.to(image.device)
    return torch.clamp(image + noise, 0, 1)

if __name__ == "__main__":
    root_path = "/home/sgh/data/WorkSpace/MultiMAE/dataset/train_data_0208/rgb/"
    # rotate_images(root_path)
    # IMAGE_PLUG_TOUCH_MEAN = [0.24898757, 0.45462582, 0.4540073]
    # IMAGE_PLUG_TOUCH_STD = [0.08211885, 0.04238947, 0.09368436]
    IMAGE_PLUG_MEAN = [0.46378568, 0.36478597, 0.27725574]
    IMAGE_PLUG_STD = [0.26775154, 0.17099987, 0.14608   ]
    # mean_std_statistics(root_path)
    show_image(root_path, True, IMAGE_PLUG_MEAN, IMAGE_PLUG_STD)