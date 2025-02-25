import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import ImageDraw , ImageFont
from scipy.stats import laplace, norm

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

def tensor_to_img(tensor):
    """将归一化的tensor转换为PIL图像"""
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return transforms.ToPILImage()(torch.clamp(denorm(tensor), 0, 1))

def visualize_results_rgb(img1_tensor, img2_tensor, pred, gt, save_path):
    """使用PIL库保存结果
    Args:
        img1_tensor: 加噪声后的图像1 tensor [C,H,W]
        img2_tensor: 加噪声后的图像2 tensor [C,H,W]
        pred: 预测值 [dx, dy, drz]
        gt: 真实值 [dx, dy, drz]
        save_path: 保存路径
    """
    # 转换为PIL图像
    img1 = tensor_to_img(img1_tensor)
    img2 = tensor_to_img(img2_tensor)
    
    # 创建新图像
    width = img1.width * 2 + 10  # 间隔10像素
    height = img1.height + 50    # 底部留50像素显示文本
    result = Image.new('RGB', (width, height), 'white')
    
    # 粘贴图像
    result.paste(img1, (0, 0))
    result.paste(img2, (img1.width + 10, 0))
    
    # 添加文本
    draw = ImageDraw.Draw(result)
    text = f'Pred: dx={pred[0]:.2f}, dy={pred[1]:.2f}, drz={pred[2]:.2f}\n' + \
           f'GT: dx={gt[0]:.2f}, dy={gt[1]:.2f}, drz={gt[2]:.2f}'
    
    # 计算文本位置使其居中
    text_width = draw.textlength(text.split('\n')[0])  # 估算文本宽度
    x = (width - text_width) // 2
    draw.text((x, img1.height + 10), text, fill='black')
    
    # 保存结果
    result.save(save_path)

def visualize_results_rgb_touch(rgb_img1_tensor, rgb_img2_tensor, 
                              touch_img1_tensor, touch_img2_tensor, 
                              pred, gt, save_path):
    """使用PIL库保存RGB和触觉图像结果
    Args:
        rgb_img1_tensor: RGB图像1 tensor [C,H,W]
        rgb_img2_tensor: RGB图像2 tensor [C,H,W]
        touch_img1_tensor: 触觉图像1 tensor [C,H,W]
        touch_img2_tensor: 触觉图像2 tensor [C,H,W]
        pred: 预测值 [dx, dy, drz]
        gt: 真实值 [dx, dy, drz]
        save_path: 保存路径
    """
    # 转换为PIL图像
    rgb_img1 = tensor_to_img(rgb_img1_tensor)
    rgb_img2 = tensor_to_img(rgb_img2_tensor)
    touch_img1 = tensor_to_img(touch_img1_tensor)
    touch_img2 = tensor_to_img(touch_img2_tensor)
    
    # 创建新图像
    width = rgb_img1.width * 2 + 10   # 间隔10像素
    height = rgb_img1.height * 2 + 60  # 底部留60像素显示文本
    result = Image.new('RGB', (width, height), 'white')
    
    # 粘贴图像
    result.paste(rgb_img1, (0, 0))  # 左上
    result.paste(rgb_img2, (rgb_img1.width + 10, 0))  # 右上
    result.paste(touch_img1, (0, rgb_img1.height + 10))  # 左下
    result.paste(touch_img2, (touch_img1.width + 10, rgb_img1.height + 10))  # 右下
    
    # 添加文本
    draw = ImageDraw.Draw(result)
    text = f'Pred: dx={pred[0]:.2f}, dy={pred[1]:.2f}, drz={pred[2]:.2f}\n' + \
           f'GT: dx={gt[0]:.2f}, dy={gt[1]:.2f}, drz={gt[2]:.2f}'
    
    # 计算文本位置使其居中
    text_width = draw.textlength(text.split('\n')[0])
    x = (width - text_width) // 2
    y = height - 50  # 底部文本位置
    draw.text((x, y), text, fill='black')
    
    # 保存结果
    result.save(save_path)
    
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
    return image + noise

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
from scipy.optimize import minimize

# 计算均方误差（MSE）
def mse(params, data, dist_type='gaussian'):
    """计算拟合误差
    Args:
        params: 分布参数 [loc/mean, scale/std]
        data: 原始数据
        dist_type: 分布类型 'gaussian' 或 'laplace'
    """
    # 计算直方图
    hist_data, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算理论分布在bin_centers处的PDF
    if dist_type == 'gaussian':
        mean, std = params
        pdf = norm.pdf(bin_centers, mean, std)
    elif dist_type == 'laplace':
        loc, scale = params
        pdf = laplace.pdf(bin_centers, loc, scale)
    
    # 返回均方误差
    return np.sum((pdf - hist_data) ** 2)

# 计算最大似然估计（MLE）
def mle(params, data, dist_type='gaussian'):
    if dist_type == 'gaussian':
        mean, std = params
        # 计算正态分布的对数似然函数
        log_likelihood = np.sum(np.log(norm.pdf(data, mean, std)))
    elif dist_type == 'laplace':
        loc, scale = params
        # 计算拉普拉斯分布的对数似然函数
        log_likelihood = np.sum(np.log(laplace.pdf(data, loc, scale)))
    
    # 负的对数似然，因为我们希望最大化对数似然
    return -log_likelihood

# 对数据进行拟合，并选择优化的指标类型
def fit_data(data, dist_type='gaussian', optimize_type='mse'):
    """拟合数据分布
    Args:
        data: 原始数据
        dist_type: 分布类型
        optimize_type: 优化方法 'mse' 或 'mle'
    """
    # 初始参数估计
    if dist_type == 'gaussian':
        initial_params = [np.mean(data), np.std(data)]
    elif dist_type == 'laplace':
        initial_params = [np.median(data), np.mean(np.abs(data - np.median(data)))]
    
    # 参数优化
    if optimize_type == 'mse':
        result = minimize(
            mse, 
            initial_params, 
            args=(data, dist_type),
            bounds=[(None, None), (1e-10, None)],  # 防止scale为0
            method='Nelder-Mead'  # 使用更稳定的优化方法
        )
    elif optimize_type == 'mle':
        result = minimize(
            mle, 
            initial_params, 
            args=(data, dist_type),
            bounds=[(None, None), (1e-10, None)],
            method='Nelder-Mead'
        )
    
    if not result.success:
        print(f"Warning: Optimization failed: {result.message}")
        return initial_params
        
    return result.x

def data_statistics(errors, error_cut=0.9973, dtype='laplace', optimize_type='mse'):
    """计算统计量，返回99.73%区间的截断误差"""
    
    # 使用拟合数据
    optimized_params = fit_data(errors, dist_type=dtype, optimize_type=optimize_type)

    if dtype == 'laplace':
        loc, scale = optimized_params
        k = -np.log(1 - error_cut)
        return loc + k * scale
    elif dtype == 'gaussian':
        mean, std = optimized_params
        return mean + 3 * std

def plot_error_distribution(errors_x, errors_y, errors_rz, save_path, dtype='gaussian', optimize_type='mse'):
    """绘制误差分布直方图和分布拟合曲线"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制拉普拉斯分布
    def plot_with_laplace(ax, data, title, xlabel):
        # 使用拟合方法优化参数
        optimized_params = fit_data(data, dist_type='laplace', optimize_type=optimize_type)
        loc, scale = optimized_params
        
        # 绘制直方图
        ax.hist(data, bins=30, density=True, alpha=0.7, label='Histogram')
        
        # 生成拉普拉斯分布曲线
        x = np.linspace(min(data), max(data), 100)
        laplace_pdf = 1 / (2 * scale) * np.exp(-np.abs(x - loc) / scale)
        ax.plot(x, laplace_pdf, 'r-', label=f'loc={loc:.2f}\nscale={scale:.2f}')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()

    # 绘制正态分布
    def plot_with_gaussian(ax, data, title, xlabel):
        # 使用拟合方法优化参数
        optimized_params = fit_data(data, dist_type='gaussian', optimize_type=optimize_type)
        mean, std = optimized_params
        
        # 绘制直方图
        ax.hist(data, bins=30, density=True, alpha=0.7, label='Histogram')
        
        # 生成正态分布曲线
        x = np.linspace(min(data), max(data), 100)
        gaussian_pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        ax.plot(x, gaussian_pdf, 'r-', label=f'μ={mean:.2f}\nσ={std:.2f}')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()

    # 根据选择绘制拉普拉斯或正态分布
    if dtype == 'laplace':
        plot_with_laplace(ax1, errors_x, 'X Error Distribution', 'Error (mm)')
        plot_with_laplace(ax2, errors_y, 'Y Error Distribution', 'Error (mm)')
        plot_with_laplace(ax3, errors_rz, 'Rz Error Distribution', 'Error (deg)')
    elif dtype == 'gaussian':
        plot_with_gaussian(ax1, errors_x, 'X Error Distribution', 'Error (mm)')
        plot_with_gaussian(ax2, errors_y, 'Y Error Distribution', 'Error (mm)')
        plot_with_gaussian(ax3, errors_rz, 'Rz Error Distribution', 'Error (deg)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, laplace

# def data_statistics(errors, error_cut=0.9973, dtype='laplace'):
#     """计算统计量，返回99.73%区间的截断误差"""
    
#     if dtype == 'laplace':
#         # 使用scipy的laplace.fit()进行拟合
#         loc, scale = laplace.fit(errors)  # 拉普拉斯分布的参数
        
#         # 计算误差的截断区间，使用k = -log(1 - error_cut)来计算
#         k = - np.log(1 - error_cut)
        
#         # 计算99.73%截断误差
#         return loc + k * scale
#     elif dtype == 'gaussian':
#         # 使用scipy的norm.fit()进行拟合
#         mean, std = norm.fit(errors)  # 正态分布的参数
        
#         # 计算99.73%截断误差（3σ）
#         return mean + 3 * std

# def plot_error_distribution(errors_x, errors_y, errors_rz, save_path, dtype='gaussian'):
#     """绘制误差分布直方图和分布拟合曲线"""
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     # 绘制拉普拉斯分布
#     def plot_with_laplace(ax, data, title, xlabel):
#         # 使用scipy的laplace.fit()进行拟合
#         loc, scale = laplace.fit(data)
        
#         # 绘制直方图
#         ax.hist(data, bins=30, density=True, alpha=0.7, label='Histogram')
        
#         # 生成拉普拉斯分布曲线
#         x = np.linspace(min(data), max(data), 100)
#         laplace_pdf = 1 / (2 * scale) * np.exp(-np.abs(x - loc) / scale)
#         ax.plot(x, laplace_pdf, 'r-', label=f'loc={loc:.2f}\nscale={scale:.2f}')
        
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel('Density')
#         ax.legend()

#     # 绘制正态分布
#     def plot_with_gaussian(ax, data, title, xlabel):
#         # 使用scipy的norm.fit()进行拟合
#         mean, std = norm.fit(data)
        
#         std = 0.8 * std  # 缩小标准差
#         # 绘制直方图
#         ax.hist(data, bins=30, density=True, alpha=0.7, label='Histogram')
        
#         # 生成正态分布曲线
#         x = np.linspace(min(data), max(data), 100)
#         gaussian_pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
#         ax.plot(x, gaussian_pdf, 'r-', label=f'μ={mean:.2f}\nσ={std:.2f}')
        
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel('Density')
#         ax.legend()

#     # 根据选择绘制拉普拉斯或正态分布
#     if dtype == 'laplace':
#         plot_with_laplace(ax1, errors_x, 'X Error Distribution', 'Error (mm)')
#         plot_with_laplace(ax2, errors_y, 'Y Error Distribution', 'Error (mm)')
#         plot_with_laplace(ax3, errors_rz, 'Rz Error Distribution', 'Error (deg)')
#     elif dtype == 'gaussian':
#         plot_with_gaussian(ax1, errors_x, 'X Error Distribution', 'Error (mm)')
#         plot_with_gaussian(ax2, errors_y, 'Y Error Distribution', 'Error (mm)')
#         plot_with_gaussian(ax3, errors_rz, 'Rz Error Distribution', 'Error (deg)')
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()



if __name__ == "__main__":
    root_path = "/home/sgh/data/WorkSpace/MultiMAE/dataset/train_data_0208/rgb/"
    # rotate_images(root_path)
    # IMAGE_PLUG_TOUCH_MEAN = [0.24898757, 0.45462582, 0.4540073]
    # IMAGE_PLUG_TOUCH_STD = [0.08211885, 0.04238947, 0.09368436]
    IMAGE_PLUG_MEAN = [0.46378568, 0.36478597, 0.27725574]
    IMAGE_PLUG_STD = [0.26775154, 0.17099987, 0.14608   ]
    # mean_std_statistics(root_path)
    show_image(root_path, True, IMAGE_PLUG_MEAN, IMAGE_PLUG_STD)