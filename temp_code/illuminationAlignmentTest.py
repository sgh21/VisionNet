import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from scipy import ndimage
from tkinter import filedialog
import torch
import sys
import os

# 添加项目根目录到路径，以便导入utils模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TransUtils import PatchBasedIlluminationAlignment, ssim_np2torch

def select_image():
    """使用文件对话框选择图片"""
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path

def calculate_image_stats(img):
    """计算图像的RGB和灰度通道的均值和方差"""
    # BGR通道统计
    b, g, r = cv2.split(img)
    
    # 计算每个通道的均值和方差
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    r_var = np.var(r)
    g_var = np.var(g)
    b_var = np.var(b)
    
    # 计算灰度图的均值和方差
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = np.mean(gray)
    gray_var = np.var(gray)
    
    return {
        'mean': {'R': r_mean, 'G': g_mean, 'B': b_mean, 'Gray': gray_mean},
        'variance': {'R': r_var, 'G': g_var, 'B': b_var, 'Gray': gray_var}
    }

def np_to_torch(img_np, device='cuda', normalize=True):
    """
    将NumPy图像转为PyTorch张量
    
    参数:
        img_np: NumPy格式的图像，形状为(H, W, C)
        device: 设备，默认为'cuda'
        normalize: 是否将像素值归一化到[0,1]范围
        
    返回:
        torch_tensor: PyTorch张量，形状为(1, C, H, W)
    """
    # 检查是否有可用的CUDA设备
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA不可用，将使用CPU")
    
    # 将BGR格式转为RGB格式
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # 调整轴顺序并添加批次维度
    img_chw = np.transpose(img_rgb, (2, 0, 1))  # (C, H, W)
    img_bchw = np.expand_dims(img_chw, axis=0)  # (1, C, H, W)
    
    # 转为PyTorch张量
    tensor = torch.from_numpy(img_bchw).float()
    
    # 如果需要，归一化到[0,1]范围
    if normalize:
        tensor = tensor / 255.0
    
    return tensor.to(device)

def torch_to_np(tensor, denormalize=True):
    """
    将PyTorch张量转回NumPy格式
    
    参数:
        tensor: PyTorch张量，形状为(B, C, H, W)
        denormalize: 是否将像素值反归一化到[0,255]范围
        
    返回:
        img_np: NumPy图像，形状为(H, W, C)，BGR格式
    """
    # 将张量移动到CPU并转为NumPy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 取批次中的第一个图像
    tensor = tensor[0]  # (C, H, W)
    
    # 如果需要，反归一化到[0,255]范围
    if denormalize:
        tensor = tensor * 255.0
    
    # 裁剪到有效范围
    tensor = torch.clamp(tensor, 0, 255)
    
    # 转为NumPy并调整轴顺序
    img_np = tensor.detach().numpy().astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
    
    # 将RGB格式转回BGR格式
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def illumination_alignment_pytorch_wrapper(img, template, window_size=112, kernel_size=None, keep_variance=False):
    """
    使用PyTorch实现的光照对齐函数的包装器
    
    参数:
        img: NumPy格式的输入图像 (H, W, C)
        template: NumPy格式的模板图像 (H, W, C)
        window_size: 作用窗口大小
        kernel_size: 计算均值和方差时的卷积核大小，默认为None
        keep_variance: 是否保持方差，默认为False
        
    返回:
        aligned_img: NumPy格式的光照调整后的图像 (H, W, C)
    """
    # 检查有没有CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 将图像转为PyTorch张量
    img_tensor = np_to_torch(img, device, normalize=True)
    template_tensor = np_to_torch(template, device, normalize=True)
    
    # 创建对齐模块并移动到相应设备
    alignment_module = PatchBasedIlluminationAlignment(window_size=window_size, kernel_size=kernel_size, keep_variance=keep_variance).to(device)
    
    # 应用光照对齐
    with torch.no_grad():  # 不需要梯度
        aligned_tensor = alignment_module(img_tensor, template_tensor)
    
    # 将结果转回NumPy格式
    aligned_img = torch_to_np(aligned_tensor)
    
    return aligned_img

def ssim(im1, im2, win_size=11, sigma=1.5, k1=0.01, k2=0.03, data_range=None, 
             gaussian_weights=True, use_sample_covariance=True, full=False, wo_light=False):
    """
    计算两张图像之间的结构相似性指数(SSIM)，与skimage.metrics.structural_similarity匹配
    
    参数:
        im1, im2: 输入图像数组
        win_size: 窗口大小，必须是奇数
        sigma: 高斯权重的标准差
        k1, k2: SSIM计算中的常数
        data_range: 数据范围(最大值-最小值)，None时自动从数据类型估计
        gaussian_weights: 是否使用高斯加权窗口(True)还是均匀窗口(False)
        use_sample_covariance: 如果为True，使用N-1归一化协方差，否则使用N
        full: 如果为True，返回完整的SSIM图像和均值，否则只返回均值
    
    返回:
        如果full=False，返回SSIM得分(标量)
        如果full=True，返回(SSIM得分, SSIM图像)
    """
    # 检查输入
    if im1.shape != im2.shape:
        raise ValueError("输入图像必须具有相同的尺寸")
    
    # 如果没有提供data_range，从图像类型估计
    if data_range is None:
        if np.issubdtype(im1.dtype, np.floating):
            raise ValueError("对于浮点图像，必须明确指定data_range")
        dmin, dmax = 0, 255  # 假设8位图像
        data_range = dmax - dmin
    
    # 转换为浮点类型
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    
    # 确保窗口大小是奇数
    if win_size % 2 == 0:
        win_size = win_size + 1
    
    # 常数
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    # 创建过滤器
    NP = win_size ** im1.ndim
    
    if gaussian_weights:
        # 使用skimage中相同的高斯核参数
        truncate = 3.5
        radius = int(truncate * sigma + 0.5)
        x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
        kernel = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
        kernel = kernel / kernel.sum()
        
        # 计算均值和方差使用高斯卷积
        filter_func = lambda x, **kwargs: ndimage.convolve(x, kernel, mode='reflect')
    else:
        # 使用均匀窗口
        filter_func = lambda x, **kwargs: ndimage.uniform_filter(x, size=win_size, mode='reflect')
    
    # 样本协方差标准化
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # 样本协方差
    else:
        cov_norm = 1.0  # 总体协方差
    
    # 计算均值
    ux = filter_func(im1)
    uy = filter_func(im2)
    
    # 计算方差和协方差
    uxx = filter_func(im1 * im1)
    uyy = filter_func(im2 * im2)
    uxy = filter_func(im1 * im2)
    
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    
    # SSIM计算方式匹配skimage实现
    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux**2 + uy**2 + C1
    B2 = vx + vy + C2
    if wo_light:
        S = A2 / B2
    else:
        D = B1 * B2
        S = (A1 * A2) / D
    
    # 忽略边缘效应
    pad = (win_size - 1) // 2
    
    # 使用裁剪方式匹配skimage
    if pad > 0:
        S_cropped = S[pad:-pad, pad:-pad]
    else:
        S_cropped = S
    
    # 计算平均SSIM
    mssim = np.mean(S_cropped)
    
    if full:
        return mssim, S
    else:
        return mssim
def calculate_mae(img1, img2):
    """
    计算两个图像的平均绝对误差(MAE)
    
    参数:
        img1, img2: 输入图像，应为相同大小的NumPy数组
    
    返回:
        total_mae: 所有通道的平均MAE
        channel_mae: 各通道单独的MAE字典
    """
    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("输入图像必须具有相同的尺寸")
    
    # 转换为浮点型以避免溢出
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 计算各通道的MAE
    if len(img1.shape) == 3 and img1.shape[2] == 3:  # 彩色图像
        # 分别计算B、G、R通道的MAE
        b1, g1, r1 = cv2.split(img1)
        b2, g2, r2 = cv2.split(img2)
        
        mae_b = np.mean(np.abs(b1 - b2))
        mae_g = np.mean(np.abs(g1 - g2))
        mae_r = np.mean(np.abs(r1 - r2))
        
        # 计算灰度图的MAE
        gray1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mae_gray = np.mean(np.abs(gray1 - gray2))
        
        # 所有像素点的总体MAE
        total_mae = np.mean(np.abs(img1 - img2))
        
        return total_mae, {'B': mae_b, 'G': mae_g, 'R': mae_r, 'Gray': mae_gray}
    else:  # 灰度图或其他格式
        mae = np.mean(np.abs(img1 - img2))
        return mae, {'Gray': mae}
def calculate_psnr(mse, max_value=255.0):
    """
    根据MSE计算峰值信噪比(PSNR)
    
    参数:
        mse: 均方误差
        max_value: 像素最大值
    
    返回:
        psnr: 峰值信噪比，单位为dB
    """
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_value / np.sqrt(mse))

# def compare_images(img1, img2, align=True):
#     """比较两张图片并返回比较结果，同时计算考虑光照和不考虑光照的SSIM"""
#     # 确保两张图片大小相同
#     if img1.shape != img2.shape:
#         img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
#     import time
#     start_time = time.time()
#     # 使用PyTorch版本的光照对齐函数
#     if align:
#         img1_aligned = illumination_alignment_pytorch_wrapper(img1, img2, window_size=2, kernel_size=4, keep_variance=True)
#     else:
#         img1_aligned = img1
#     end_time = time.time()
#     print(f"PyTorch版本光照对齐执行时间: {end_time - start_time:.4f} 秒")
    
#     # 计算图像统计信息
#     img1_stats = calculate_image_stats(img1_aligned)
#     img2_stats = calculate_image_stats(img2)
    
#     # 转为灰度图进行结构相似性比较
#     gray1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     # 设置共同的SSIM参数
#     win_size = 11
#     sigma = 1.5
#     k1 = 0.01
#     k2 = 0.03
#     use_sample_covariance = True  # 匹配skimage默认值
#     gaussian_weights = True      # 匹配skimage默认值
    
#     # 计算考虑光照的SSIM (默认)
#     score_with_light, diff_with_light = ssim(
#         gray1, gray2, 
#         win_size=win_size,
#         sigma=sigma, 
#         k1=k1, k2=k2,
#         gaussian_weights=gaussian_weights,
#         use_sample_covariance=use_sample_covariance,
#         data_range=255,
#         full=True,
#         wo_light=False  # 包含亮度项
#     )
#     img1_rgb = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
#     img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#     # 计算不考虑光照的SSIM (仅结构和对比度)
#     score_without_light, diff_without_light = ssim_np2torch(
#         img1_rgb, img2_rgb,
#         win_size=win_size,
#         sigma=sigma,
#         k1=k1, k2=k2,
#         gaussian_weights=gaussian_weights,
#         use_sample_covariance=use_sample_covariance,
#         data_range=255,
#         full=True,
#         wo_light=True  # 包含亮度项
#     )
    
#     # 使用不考虑光照的差异图进行后续处理(通常能更好地展示结构差异)
#     diff = diff_without_light
    
#     # 将差异图标准化为8位无符号整型
#     diff = (diff * 255).astype("uint8")
    
#     # 阈值处理，找出差异明显的区域
#     thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     # 查找轮廓
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 在原图上绘制差异区域
#     img_diff = img1_aligned.copy()
#     cv2.drawContours(img_diff, contours, -1, (0, 0, 255), 2)
    
#     # 生成绝对差异图
#     abs_diff = cv2.absdiff(img1_aligned, img2)
    
#     # 生成热力图
#     heat_map = cv2.applyColorMap(cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    
#     return {
#         "img1": img1_aligned,  # 返回对齐后的图像
#         "img2": img2,
#         "abs_diff": abs_diff,
#         "heat_map": heat_map,
#         "img_diff": img_diff,
#         "ssim_score_with_light": score_with_light,
#         "ssim_score_without_light": score_without_light,
#         "contours_count": len(contours),
#         "img1_stats": img1_stats,
#         "img2_stats": img2_stats
#     }

# def visualize_comparison(comparison_result):
#     """可视化比较结果，同时显示考虑光照和不考虑光照的SSIM值"""
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
#     # 获取统计信息
#     img1_stats = comparison_result["img1_stats"]
#     img2_stats = comparison_result["img2_stats"]
    
#     # 显示原图1 (对齐后)
#     axes[0, 0].imshow(cv2.cvtColor(comparison_result["img1"], cv2.COLOR_BGR2RGB))
#     axes[0, 0].set_title("Image 1 (Aligned)")
#     stats_text1 = (f"Mean RGB: ({img1_stats['mean']['R']:.1f}, {img1_stats['mean']['G']:.1f}, {img1_stats['mean']['B']:.1f}), Gray: {img1_stats['mean']['Gray']:.1f}\n"
#                    f"Var RGB: ({img1_stats['variance']['R']:.1f}, {img1_stats['variance']['G']:.1f}, {img1_stats['variance']['B']:.1f}), Gray: {img1_stats['variance']['Gray']:.1f}")
#     axes[0, 0].text(0.5, -0.1, stats_text1, transform=axes[0, 0].transAxes, ha='center', fontsize=9)
#     axes[0, 0].axis("off")
    
#     # 显示原图2
#     axes[0, 1].imshow(cv2.cvtColor(comparison_result["img2"], cv2.COLOR_BGR2RGB))
#     axes[0, 1].set_title("Image 2 (Target)")
#     stats_text2 = (f"Mean RGB: ({img2_stats['mean']['R']:.1f}, {img2_stats['mean']['G']:.1f}, {img2_stats['mean']['B']:.1f}), Gray: {img2_stats['mean']['Gray']:.1f}\n"
#                    f"Var RGB: ({img2_stats['variance']['R']:.1f}, {img2_stats['variance']['G']:.1f}, {img2_stats['variance']['B']:.1f}), Gray: {img2_stats['variance']['Gray']:.1f}")
#     axes[0, 1].text(0.5, -0.1, stats_text2, transform=axes[0, 1].transAxes, ha='center', fontsize=9)
#     axes[0, 1].axis("off")
    
#     # 显示绝对差异图
#     axes[0, 2].imshow(cv2.cvtColor(comparison_result["abs_diff"], cv2.COLOR_BGR2RGB))
#     axes[0, 2].set_title("Absolute Difference")
#     axes[0, 2].axis("off")
    
#     # 显示热力图
#     axes[1, 0].imshow(cv2.cvtColor(comparison_result["heat_map"], cv2.COLOR_BGR2RGB))
#     axes[1, 0].set_title("Difference Heat Map")
#     axes[1, 0].axis("off")
    
#     # 显示轮廓标记的差异图
#     axes[1, 1].imshow(cv2.cvtColor(comparison_result["img_diff"], cv2.COLOR_BGR2RGB))
#     axes[1, 1].set_title(f"Marked Differences ({comparison_result['contours_count']} regions)")
#     axes[1, 1].axis("off")
    
#     # 显示SSIM分数（同时显示考虑光照和不考虑光照的分数）
#     ssim_text = (
#         f"SSIM Score (with luminance): {comparison_result['ssim_score_with_light']:.4f}\n"
#         f"SSIM Score (w/o luminance): {comparison_result['ssim_score_without_light']:.4f}"
#     )
#     axes[1, 2].text(0.5, 0.5, ssim_text, 
#                   horizontalalignment='center', verticalalignment='center', fontsize=14)
#     axes[1, 2].axis("off")
    
#     # 添加RGB和灰度值的统计图表
#     # 显示图像1的RGB和灰度通道均值柱状图
#     channel_names = ['R', 'G', 'B', 'Gray']
#     img1_means = [img1_stats['mean']['R'], img1_stats['mean']['G'], img1_stats['mean']['B'], img1_stats['mean']['Gray']]
#     img2_means = [img2_stats['mean']['R'], img2_stats['mean']['G'], img2_stats['mean']['B'], img2_stats['mean']['Gray']]
    
#     x = np.arange(len(channel_names))
#     width = 0.35
    
#     axes[2, 0].bar(x - width/2, img1_means, width, label='Image 1')
#     axes[2, 0].bar(x + width/2, img2_means, width, label='Image 2')
    
#     axes[2, 0].set_title('Channel Means Comparison')
#     axes[2, 0].set_xticks(x)
#     axes[2, 0].set_xticklabels(channel_names)
#     axes[2, 0].legend()
    
#     # 显示图像的RGB和灰度通道方差柱状图
#     img1_vars = [img1_stats['variance']['R'], img1_stats['variance']['G'], img1_stats['variance']['B'], img1_stats['variance']['Gray']]
#     img2_vars = [img2_stats['variance']['R'], img2_stats['variance']['G'], img2_stats['variance']['B'], img2_stats['variance']['Gray']]
    
#     axes[2, 1].bar(x - width/2, img1_vars, width, label='Image 1')
#     axes[2, 1].bar(x + width/2, img2_vars, width, label='Image 2')
    
#     axes[2, 1].set_title('Channel Variances Comparison')
#     axes[2, 1].set_xticks(x)
#     axes[2, 1].set_xticklabels(channel_names)
#     axes[2, 1].legend()
    
#     # 将均值和方差的差异显示为饼图
#     mean_diffs = [abs(img1_means[i] - img2_means[i]) for i in range(len(channel_names))]
#     var_diffs = [abs(img1_vars[i] - img2_vars[i]) for i in range(len(channel_names))]
    
#     axes[2, 2].pie(mean_diffs, labels=channel_names, autopct='%1.1f%%',
#                    startangle=90)
#     axes[2, 2].set_title('Channel Mean Differences')
    
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.3)
#     plt.show()

def compare_images(img1, img2, align=True):
    """比较两张图片并返回比较结果，同时计算考虑光照和不考虑光照的SSIM以及MAE"""
    # 确保两张图片大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    import time
    start_time = time.time()
    # 使用PyTorch版本的光照对齐函数
    if align:
        img1_aligned = illumination_alignment_pytorch_wrapper(img1, img2, window_size=4, kernel_size=8, keep_variance=True)
    else:
        img1_aligned = img1
    end_time = time.time()
    print(f"PyTorch版本光照对齐执行时间: {end_time - start_time:.4f} 秒")
    
    # 计算图像统计信息
    img1_stats = calculate_image_stats(img1_aligned)
    img2_stats = calculate_image_stats(img2)
    
    # 计算MAE
    total_mae, channel_mae = calculate_mae(img1_aligned, img2)
    
    # 转为灰度图进行结构相似性比较
    gray1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 设置共同的SSIM参数
    win_size = 11
    sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    use_sample_covariance = True  # 匹配skimage默认值
    gaussian_weights = True      # 匹配skimage默认值
    
    # 计算考虑光照的SSIM (默认)
    score_with_light, diff_with_light = ssim(
        gray1, gray2, 
        win_size=win_size,
        sigma=sigma, 
        k1=k1, k2=k2,
        gaussian_weights=gaussian_weights,
        use_sample_covariance=use_sample_covariance,
        data_range=255,
        full=True,
        wo_light=False  # 包含亮度项
    )
    img1_rgb = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 计算不考虑光照的SSIM (仅结构和对比度)
    score_without_light, diff_without_light = ssim_np2torch(
        img1_rgb, img2_rgb,
        win_size=win_size,
        sigma=sigma,
        k1=k1, k2=k2,
        gaussian_weights=gaussian_weights,
        use_sample_covariance=use_sample_covariance,
        data_range=255,
        full=True,
        wo_light=True  # 包含亮度项
    )
    
    # 使用不考虑光照的差异图进行后续处理(通常能更好地展示结构差异)
    diff = diff_without_light
    
    # 将差异图标准化为8位无符号整型
    diff = (diff * 255).astype("uint8")
    
    # 阈值处理，找出差异明显的区域
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制差异区域
    img_diff = img1_aligned.copy()
    cv2.drawContours(img_diff, contours, -1, (0, 0, 255), 2)
    
    # 生成绝对差异图
    abs_diff = cv2.absdiff(img1_aligned, img2)
    
    # 生成热力图
    heat_map = cv2.applyColorMap(cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    
    # 生成MAE的逐像素差异图 - 直接使用绝对差异
    mae_map = np.abs(img1_aligned.astype(np.float32) - img2.astype(np.float32))
    # 将MAE图缩放到0-255范围用于可视化
    mae_map_normalized = np.mean(mae_map, axis=2)  # 取平均值作为灰度图
    mae_map_normalized = (mae_map_normalized / mae_map_normalized.max() * 255).astype(np.uint8)
    mae_heat_map = cv2.applyColorMap(mae_map_normalized, cv2.COLORMAP_JET)
    
    return {
        "img1": img1_aligned,  # 返回对齐后的图像
        "img2": img2,
        "abs_diff": abs_diff,
        "heat_map": heat_map,
        "img_diff": img_diff,
        "mae_heat_map": mae_heat_map,
        "ssim_score_with_light": score_with_light,
        "ssim_score_without_light": score_without_light,
        "mae": total_mae,
        "channel_mae": channel_mae,
        "contours_count": len(contours),
        "img1_stats": img1_stats,
        "img2_stats": img2_stats
    }
def visualize_comparison(comparison_result):
    """可视化比较结果，同时显示考虑光照和不考虑光照的SSIM值以及MAE"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # 获取统计信息
    img1_stats = comparison_result["img1_stats"]
    img2_stats = comparison_result["img2_stats"]
    
    # 显示原图1 (对齐后)
    axes[0, 0].imshow(cv2.cvtColor(comparison_result["img1"], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Image 1 (Aligned)")
    stats_text1 = (f"Mean RGB: ({img1_stats['mean']['R']:.1f}, {img1_stats['mean']['G']:.1f}, {img1_stats['mean']['B']:.1f}), Gray: {img1_stats['mean']['Gray']:.1f}\n"
                   f"Var RGB: ({img1_stats['variance']['R']:.1f}, {img1_stats['variance']['G']:.1f}, {img1_stats['variance']['B']:.1f}), Gray: {img1_stats['variance']['Gray']:.1f}")
    axes[0, 0].text(0.5, -0.1, stats_text1, transform=axes[0, 0].transAxes, ha='center', fontsize=9)
    axes[0, 0].axis("off")
    
    # 显示原图2
    axes[0, 1].imshow(cv2.cvtColor(comparison_result["img2"], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Image 2 (Target)")
    stats_text2 = (f"Mean RGB: ({img2_stats['mean']['R']:.1f}, {img2_stats['mean']['G']:.1f}, {img2_stats['mean']['B']:.1f}), Gray: {img2_stats['mean']['Gray']:.1f}\n"
                   f"Var RGB: ({img2_stats['variance']['R']:.1f}, {img2_stats['variance']['G']:.1f}, {img2_stats['variance']['B']:.1f}), Gray: {img2_stats['variance']['Gray']:.1f}")
    axes[0, 1].text(0.5, -0.1, stats_text2, transform=axes[0, 1].transAxes, ha='center', fontsize=9)
    axes[0, 1].axis("off")
    
    # 显示绝对差异图
    axes[0, 2].imshow(cv2.cvtColor(comparison_result["abs_diff"], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Absolute Difference")
    axes[0, 2].axis("off")
    
    # 显示热力图
    axes[1, 0].imshow(cv2.cvtColor(comparison_result["heat_map"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Difference Heat Map")
    axes[1, 0].axis("off")
    
    # 显示MAE热力图
    axes[1, 1].imshow(cv2.cvtColor(comparison_result["mae_heat_map"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("MAE Heat Map")
    axes[1, 1].axis("off")
    
    # 显示SSIM和MAE分数
    ssim_text = (
        f"SSIM Score (with luminance): {comparison_result['ssim_score_with_light']:.4f}\n"
        f"SSIM Score (w/o luminance): {comparison_result['ssim_score_without_light']:.4f}\n"
        f"MAE: {comparison_result['mae']:.2f}"
    )
    axes[1, 2].text(0.5, 0.5, ssim_text, 
                  horizontalalignment='center', verticalalignment='center', fontsize=14)
    axes[1, 2].axis("off")
    
    # 添加RGB和灰度值的统计图表
    # 显示图像1的RGB和灰度通道均值柱状图
    channel_names = ['R', 'G', 'B', 'Gray']
    img1_means = [img1_stats['mean']['R'], img1_stats['mean']['G'], img1_stats['mean']['B'], img1_stats['mean']['Gray']]
    img2_means = [img2_stats['mean']['R'], img2_stats['mean']['G'], img2_stats['mean']['B'], img2_stats['mean']['Gray']]
    
    x = np.arange(len(channel_names))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, img1_means, width, label='Image 1')
    axes[2, 0].bar(x + width/2, img2_means, width, label='Image 2')
    
    axes[2, 0].set_title('Channel Means Comparison')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(channel_names)
    axes[2, 0].legend()
    
    # 显示MAE通道对比柱状图
    mae_values = [comparison_result['channel_mae']['R'], 
                 comparison_result['channel_mae']['G'], 
                 comparison_result['channel_mae']['B'], 
                 comparison_result['channel_mae']['Gray']]
    
    axes[2, 1].bar(x, mae_values, color=['red', 'green', 'blue', 'gray'])
    axes[2, 1].set_title('Channel MAE Values')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(channel_names)
    
    # 将均值和方差的差异显示为饼图
    mean_diffs = [abs(img1_means[i] - img2_means[i]) for i in range(len(channel_names))]
    
    axes[2, 2].pie(mean_diffs, labels=channel_names, autopct='%1.1f%%',
                   startangle=90)
    axes[2, 2].set_title('Channel Mean Differences')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
def main(img1, img2, align=True):
    # 对比图像
    comparison_result = compare_images(img1, img2, align=align)
    
    # 打印统计信息
    img1_stats = comparison_result["img1_stats"]
    img2_stats = comparison_result["img2_stats"]
    
    print("\n图像 1 统计信息 (对齐后):")
    print(f"均值 - R: {img1_stats['mean']['R']:.2f}, G: {img1_stats['mean']['G']:.2f}, B: {img1_stats['mean']['B']:.2f}, 灰度: {img1_stats['mean']['Gray']:.2f}")
    print(f"方差 - R: {img1_stats['variance']['R']:.2f}, G: {img1_stats['variance']['G']:.2f}, B: {img1_stats['variance']['B']:.2f}, 灰度: {img1_stats['variance']['Gray']:.2f}")
    
    print("\n图像 2 统计信息:")
    print(f"均值 - R: {img2_stats['mean']['R']:.2f}, G: {img2_stats['mean']['G']:.2f}, B: {img2_stats['mean']['B']:.2f}, 灰度: {img2_stats['mean']['Gray']:.2f}")
    print(f"方差 - R: {img2_stats['variance']['R']:.2f}, G: {img2_stats['variance']['G']:.2f}, B: {img2_stats['variance']['B']:.2f}, 灰度: {img2_stats['variance']['Gray']:.2f}")
    
    print("\nSSIM 相似度评分:")
    print(f"考虑光照的SSIM: {comparison_result['ssim_score_with_light']:.4f}")
    print(f"不考虑光照的SSIM: {comparison_result['ssim_score_without_light']:.4f}")
    print(f"检测到 {comparison_result['contours_count']} 个不同区域")
    # 在main函数中添加
    # print("\nMAE (平均绝对误差):")
    # print(f"总体MAE: {comparison_result['mae']:.2f}")
    # print(f"通道MAE - R: {comparison_result['channel_mae']['R']:.2f}, G: {comparison_result['channel_mae']['G']:.2f}, B: {comparison_result['channel_mae']['B']:.2f}, 灰度: {comparison_result['channel_mae']['Gray']:.2f}")
    visualize_comparison(comparison_result)

if __name__ == "__main__":
    # 显示CUDA是否可用
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"使用设备: {'CUDA - ' + device_name if cuda_available else device_name}")
    
    print("请选择第一张图像...")
    img1_path = select_image()
    if not img1_path:
        print("未选择图像，退出程序")
        exit()
    
    print("请选择第二张图像...")
    img2_path = select_image()
    if not img2_path:
        print("未选择图像，退出程序")
        exit()
    
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("读取图像失败，请检查文件路径")
        exit()
    
    print(f"比较图像:\n - {img1_path}\n - {img2_path}")
    main(img1=img1, img2=img2, align=True)
    main(img1=img1, img2=img2, align=False)