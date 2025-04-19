import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute
import multiprocessing
import time
import pickle
from functools import partial
import sys
# 导入自定义数据集
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custome_datasets import TransMAEDataset

def label_normalize(label, weight=[10, 5, 20]):
    """标签归一化函数"""
    weight = torch.tensor(weight).to(label.device)
    label = label * weight
    return label

def label_denormalize(label, weight=[10, 5, 20]):
    """标签反归一化函数"""
    weight = torch.tensor(weight).to(label.device)
    label = label / weight
    return label

def create_transform_matrix(vector, intrinsic, img_size, scale=1.0):
    """
    从物理坐标系的平移和旋转参数生成图像变换矩阵
    
    Args:
        vector (torch.Tensor): 形状为[B, 3]的张量，包含[tx, ty, theta]
            - tx, ty: 平移量，单位为mm
            - theta: 旋转角度，单位为deg
        intrinsic (list): 包含[fx, fy, flag]
            - fx, fy: 内参矩阵的焦距参数，单位为mm/pixel
            - flag: 旋转是否反向
        img_size (list): [H, W] 图像尺寸
            
    Returns:
        torch.Tensor: 形状为[B, 3]的变换矩阵参数 [theta, tx, ty]
    """
    B = vector.shape[0]
    device = vector.device
    
    # 提取参数
    tx_mm = vector[:, 0]  # x方向平移(mm)
    ty_mm = vector[:, 1]  # y方向平移(mm)
    theta_deg = vector[:, 2]  # 旋转角度(deg)
    
    # 提取内参
    fx, fy, flag = intrinsic
    
    # 提取图片尺寸
    H, W = img_size

    # 将角度转换为弧度
    theta_rad = theta_deg * (torch.pi / 180.0) * flag
    
    # 将物理平移量(mm)转换为归一化图像坐标
    tx_norm = tx_mm / (fx * W/2) # 图像x方向的归一化平移量
    ty_norm = ty_mm / (fy * H/2) # 图像y方向的归一化平移量
    
    # 构建完整的变换矩阵参数 [theta, tx, ty]
    transform_matrix = torch.stack([theta_rad, tx_norm, ty_norm], dim=1)
    
    return transform_matrix.to(device=device)

def apply_transform(img, params, CXCY=None):
    """
    使用3参数[theta,tx,ty]应用仿射变换到输入图像
    
    Args:
        img (Tensor): 输入数据，[B, C, H, W]
        params (Tensor): 变换参数，[B, 3] (theta, tx, ty)
        CXCY (list): 旋转中心坐标 [cx, cy] (-1, 1)范围
    Returns:
        Tensor: 变换后的图像，[B, C, H, W]
    """
    B, C, H, W = img.shape
    device = img.device
    
    # 提取参数
    theta = params[:, 0]  # 旋转角度
    tx = params[:, 1]  # x方向平移
    ty = params[:, 2]  # y方向平移
    
    if CXCY is not None:
        # 转化为tensor
        cx = torch.full((B, 1, 1), CXCY[0], device=device)
        cy = torch.full((B, 1, 1), CXCY[1], device=device)
    else:
        # 使用图像中心作为旋转中心
        cx = torch.zeros(B, 1, 1, device=device)
        cy = torch.zeros(B, 1, 1, device=device)

    # 构建旋转矩阵
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # 旋转矩阵元素
    a = cos_theta
    b = -sin_theta
    c = sin_theta
    d = cos_theta
    
    # 创建归一化网格坐标
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    # 扩展网格坐标到批次维度
    grid_x = grid_x.unsqueeze(0).expand(B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, H, W)
    
    # 计算行列式(稳定性检查)
    det = a * d - b * c
    eps = 1e-6
    safe_det = torch.where(torch.abs(det) < eps, 
                       torch.ones_like(det) * eps * torch.sign(det), 
                       det)
    
    # 计算逆变换矩阵(用于逆向映射)
    inv_a = d / safe_det
    inv_b = -b / safe_det
    inv_c = -c / safe_det
    inv_d = a / safe_det
    
    # 将参数调整为[B,1,1]形状，方便广播
    inv_a = inv_a.view(B, 1, 1)
    inv_b = inv_b.view(B, 1, 1)
    inv_c = inv_c.view(B, 1, 1)
    inv_d = inv_d.view(B, 1, 1)
    tx = tx.view(B, 1, 1)
    ty = ty.view(B, 1, 1)
    
    # 逆向映射坐标计算
    x_after_trans = grid_x - tx
    y_after_trans = grid_y - ty
    
    # 将坐标相对于旋转中心
    x_centered = x_after_trans - cx
    y_centered = y_after_trans - cy
    
    # 应用旋转的逆变换
    x_unrotated = inv_a * x_centered + inv_b * y_centered
    y_unrotated = inv_c * x_centered + inv_d * y_centered
    
    # 加回旋转中心
    x_in = x_unrotated + cx
    y_in = y_unrotated + cy
    
    # 组合成采样网格
    grid = torch.stack([x_in, y_in], dim=-1)
    
    # 使用grid_sample实现双线性插值
    return F.grid_sample(
        img, 
        grid, 
        mode='bilinear',      
        padding_mode='zeros', 
        align_corners=True    
    )

def calculate_transform_diff_loss(img1, img2, params, sigma=[0.5, 0.5], CXCY=None):
    """
    计算两个图像之间的MSE损失，权重图会随图像变换而变换
    
    Args:
        img1 (Tensor): 输入图像1，形状为[B, C, H, W]
        img2 (Tensor): 输入图像2，形状为[B, C, H, W]
        params (Tensor): 变换参数，[B, 3] (theta, tx, ty)
        sigma (List(float)): 高斯权重的标准差
        CXCY (list): 旋转中心坐标 [cx, cy]
                
    Returns:
        Tensor: 加权MSE损失
    """
    B, C, H, W = img1.shape
    device = img1.device
    
    # 创建基础权重图
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    sigma_x, sigma_y = sigma
    # 计算到图像中心的距离
    dist_x = x_grid.pow(2) / (2 * sigma_x**2)
    dist_y = y_grid.pow(2) / (2 * sigma_y**2)
    
    # 使用高斯函数生成权重图
    base_weights = torch.exp(-(dist_x + dist_y))
    
    # 归一化权重，使权重总和为像素数量
    base_weights = base_weights * (H * W) / base_weights.sum()
    
    # 保存为单通道图像
    base_weight_map = base_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 将基础权重图扩展到批次大小
    batch_weight_map = base_weight_map.expand(B, 1, H, W)
    
    # 应用变换到第二张图像
    img2_trans = apply_transform(img2, params, CXCY)
    
    # 应用变换到权重图
    transformed_weight_map = apply_transform(batch_weight_map, params, CXCY)
    
    # 扩展到匹配通道数
    weights = transformed_weight_map.expand(B, C, H, W)
    
    # 计算MSE损失并应用权重
    squared_diff = F.mse_loss(img1, img2_trans, reduction='none')
    loss = (squared_diff * weights).sum() / (B * C * H * W)
    
    return loss, img2_trans

def evaluate_intrinsic(intrinsic_params, images1, images2, labels1, labels2, rot_center, img_size, device):
    """
    评估给定内参的损失
    
    Args:
        intrinsic_params: [fx, fy, flag]内参参数和旋转方向标志
        images1, images2: 图像对
        labels1, labels2: 对应的标签
        rot_center: 旋转中心坐标 [cx, cy]
        img_size: 图像尺寸 [H, W]
        device: 计算设备
        
    Returns:
        float: 损失值
    """
    fx, fy, flag = intrinsic_params
    intrinsic = [fx, fy, flag]
    
    total_loss = 0
    batch_size = 32  # 分批处理以避免内存不足
    
    with torch.no_grad():
        for i in range(0, len(images1), batch_size):
            end_idx = min(i + batch_size, len(images1))
            batch_img1 = images1[i:end_idx].to(device)
            batch_img2 = images2[i:end_idx].to(device)
            batch_label1 = labels1[i:end_idx].to(device)
            batch_label2 = labels2[i:end_idx].to(device)
            
            # 计算变换参数
            delta_label = batch_label2 - batch_label1
            trans_params = create_transform_matrix(delta_label, intrinsic, img_size)
            
            # 计算变换损失
            batch_loss, _ = calculate_transform_diff_loss(
                batch_img1, batch_img2, trans_params, 
                sigma=[0.6, 0.3], CXCY=rot_center
            )
            
            total_loss += batch_loss.item() * (end_idx - i)
    
    avg_loss = total_loss / len(images1)
    return avg_loss

def evaluate_single_point(args):
    """评估单个内参点，用于并行处理"""
    fx, fy, flag, images1, images2, labels1, labels2, rot_center, img_size, device_idx = args
    
    # 分配到不同GPU或退回到CPU
    if torch.cuda.is_available() and device_idx >= 0 and device_idx < torch.cuda.device_count():
        device = torch.device(f'cuda:{device_idx}')
    else:
        device = torch.device('cpu')
    
    # 将数据移动到正确的设备上
    images1_device = images1.to(device)
    images2_device = images2.to(device)
    labels1_device = labels1.to(device)
    labels2_device = labels2.to(device)
    
    # 计算损失
    loss = evaluate_intrinsic(
        [fx, fy, flag], 
        images1_device, images2_device, 
        labels1_device, labels2_device, 
        rot_center, img_size, device
    )
    
    return (fx, fy, flag, loss)

def grid_search_intrinsic(images1, images2, labels1, labels2, fx_range, fy_range, flag_values, 
                         rot_center, img_size, n_jobs=None):
    """
    网格搜索内参
    
    Args:
        images1, images2: 图像对张量
        labels1, labels2: 对应的标签张量
        fx_range: (min_fx, max_fx, steps) - fx搜索范围和步数
        fy_range: (min_fy, max_fy, steps) - fy搜索范围和步数
        flag_values: 旋转方向标志的可能值列表，通常为[-1, 1]
        rot_center: 旋转中心坐标 [cx, cy]
        img_size: 图像尺寸 [H, W]
        n_jobs: 并行处理的作业数，None表示使用所有可用核心
        
    Returns:
        最优内参 [fx, fy, flag] 和对应的损失值
    """
    # 准备网格点
    min_fx, max_fx, fx_steps = fx_range
    min_fy, max_fy, fy_steps = fy_range
    
    fx_values = np.linspace(min_fx, max_fx, fx_steps)
    fy_values = np.linspace(min_fy, max_fy, fy_steps)
    
    print(f"搜索网格大小: {len(fx_values)}x{len(fy_values)}x{len(flag_values)} = {len(fx_values)*len(fy_values)*len(flag_values)} 点")
    
    # 准备多进程计算
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # 如果有多个GPU，分配工作到不同GPU
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 准备参数列表
    args_list = []
    current_device = 0
    
    for fx in fx_values:
        for fy in fy_values:
            for flag in flag_values:
                device_to_use = current_device % max(1, gpu_count)
                args_list.append((fx, fy, flag, images1, images2, labels1, labels2, rot_center, img_size, device_to_use))
                current_device += 1
    
    # 使用进程池并行计算
    t_start = time.time()
    results = []
    
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for result in tqdm(pool.imap(evaluate_single_point, args_list), total=len(args_list)):
            results.append(result)
    
    t_end = time.time()
    print(f"网格搜索耗时: {t_end - t_start:.2f} 秒")
    
    # 找出最优结果
    best_result = min(results, key=lambda x: x[3])
    best_fx, best_fy, best_flag, min_loss = best_result
    
    print(f"最优内参: fx={best_fx:.6f}, fy={best_fy:.6f}, flag={best_flag}, 损失={min_loss:.6f}")
    
    # 创建3D误差图可视化
    visualize_search_results(results, fx_values, fy_values, flag_values, best_result)
    
    return [best_fx, best_fy, best_flag], min_loss

def visualize_search_results(results, fx_values, fy_values, flag_values, best_result):
    """可视化搜索结果"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 提取结果
    fx_results = [r[0] for r in results]
    fy_results = [r[1] for r in results]
    flag_results = [r[2] for r in results]
    loss_results = [r[3] for r in results]
    
    # 创建分别针对每个flag值的图
    unique_flags = set(flag_results)
    
    for flag in unique_flags:
        # 筛选特定flag值的结果
        idx = [i for i, f in enumerate(flag_results) if f == flag]
        fx_flag = [fx_results[i] for i in idx]
        fy_flag = [fy_results[i] for i in idx]
        loss_flag = [loss_results[i] for i in idx]
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
        scatter = ax.scatter(fx_flag, fy_flag, loss_flag, c=loss_flag, cmap='viridis', 
                            alpha=0.6, s=50, marker='o')
        
        # 标记最优点
        if flag == best_result[2]:  # 如果当前flag是最优flag
            ax.scatter([best_result[0]], [best_result[1]], [best_result[3]], 
                      color='red', s=200, marker='*', label='最优点')
        
        # 设置标题和标签
        ax.set_title(f'内参搜索结果 (flag={flag})')
        ax.set_xlabel('fx')
        ax.set_ylabel('fy')
        ax.set_zlabel('Loss')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Loss值')
        
        # 设置视角
        ax.view_init(elev=30, azim=-60)
        
        # 显示图例
        ax.legend()
        
        # 保存图像
        plt.savefig(f'intrinsic_search_flag_{flag}.png', dpi=300)
        plt.close()
    
    # 创建2D热力图
    for flag in unique_flags:
        # 创建网格用于热力图
        loss_grid = np.ones((len(fy_values), len(fx_values))) * np.inf
        
        # 填充网格
        for fx, fy, f, loss in zip(fx_results, fy_results, flag_results, loss_results):
            if f == flag:
                # 找到最近的网格点
                fx_idx = np.abs(fx_values - fx).argmin()
                fy_idx = np.abs(fy_values - fy).argmin()
                loss_grid[fy_idx, fx_idx] = loss
        
        # 创建2D热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(loss_grid, cmap='viridis', aspect='auto',
                   extent=[fx_values[0], fx_values[-1], fy_values[-1], fy_values[0]])
        
        if flag == best_result[2]:  # 如果当前flag是最优flag
            plt.plot(best_result[0], best_result[1], 'r*', markersize=15, label='最优点')
        
        plt.colorbar(label='Loss值')
        plt.title(f'内参搜索热力图 (flag={flag})')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'intrinsic_heatmap_flag_{flag}.png', dpi=300)
        plt.close()

def optimize_intrinsic_fine(images1, images2, labels1, labels2, initial_intrinsic,
                           rot_center, img_size, device, bounds=None):
    """
    使用Nelder-Mead算法优化内参
    
    Args:
        images1, images2: 图像对
        labels1, labels2: 对应的标签
        initial_intrinsic: 初始内参 [fx, fy, flag]
        rot_center: 旋转中心坐标 [cx, cy]
        img_size: 图像尺寸 [H, W]
        device: 计算设备
        bounds: 参数范围，形式为[(min_fx, max_fx), (min_fy, max_fy)]
        
    Returns:
        最优内参 [fx, fy, flag] 和对应的损失值
    """
    # 由于flag是离散值，我们固定它并只优化fx和fy
    fx_init, fy_init, flag = initial_intrinsic
    
    # 定义要优化的函数
    def objective(params):
        fx, fy = params
        return evaluate_intrinsic([fx, fy, flag], images1, images2, labels1, labels2, rot_center, img_size, device)
    
    print(f"开始精细优化内参，初始值: fx={fx_init:.6f}, fy={fy_init:.6f}, flag={flag}")
    
    # 设置优化边界，如果未提供则使用默认值
    if bounds is None:
        bounds = [(fx_init * 0.8, fx_init * 1.2), (fy_init * 0.8, fy_init * 1.2)]
    
    # 使用Nelder-Mead方法优化
    result = minimize(objective, [fx_init, fy_init], method='Nelder-Mead', 
                     bounds=bounds, options={'maxiter': 100, 'disp': True, 'adaptive': True})
    
    if not result.success:
        print("Nelder-Mead优化不成功，尝试使用Powell方法...")
        result = minimize(objective, [fx_init, fy_init], method='Powell',
                         bounds=bounds, options={'maxiter': 100, 'disp': True})
    
    optimal_fx, optimal_fy = result.x
    optimal_loss = result.fun
    
    print(f"精细优化结果: fx={optimal_fx:.6f}, fy={optimal_fy:.6f}, flag={flag}, 损失={optimal_loss:.6f}")
    
    return [optimal_fx, optimal_fy, flag], optimal_loss

def load_and_prepare_dataset(config, transform):
    """加载和准备数据集"""
    dataset = TransMAEDataset(config, is_train=True, transform=transform, use_fix_template=config.use_fix_template)
    
    # 转换为所需格式的张量
    images1 = []
    images2 = []
    labels1 = []
    labels2 = []
    
    # 使用DataLoader加载更高效
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    for batch in tqdm(loader, desc="加载数据集"):
        img1, img2, high_res_img1, high_res_img2, label1, label2 = batch
        
        # 使用high_res图像以获得更好的精度
        images1.append(high_res_img1)
        images2.append(high_res_img2)
        labels1.append(label1)
        labels2.append(label2)
    
    # 连接所有批次
    images1 = torch.cat(images1)
    images2 = torch.cat(images2)
    labels1 = torch.cat(labels1)
    labels2 = torch.cat(labels2)
    
    print(f"已加载数据集: {len(images1)} 对图像")
    
    return images1, images2, labels1, labels2

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='优化相机内参')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--high_res_size', type=int, default=560, help='高分辨率图像大小')
    parser.add_argument('--input_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--use_fix_template', action='store_true', help='使用固定模板图像')
    parser.add_argument('--fx_min', type=float, default=0.018, help='fx最小值')
    parser.add_argument('--fx_max', type=float, default=0.022, help='fx最大值')
    parser.add_argument('--fx_steps', type=int, default=8, help='fx搜索步数')
    parser.add_argument('--fy_min', type=float, default=0.018, help='fy最小值')
    parser.add_argument('--fy_max', type=float, default=0.022, help='fy最大值')
    parser.add_argument('--fy_steps', type=int, default=8, help='fy搜索步数')
    parser.add_argument('--fine_tune', action='store_true', help='是否进行精细优化')
    parser.add_argument('--rot_center_x', type=float, default=-0.0102543, help='旋转中心x坐标')
    parser.add_argument('--rot_center_y', type=float, default=-0.0334525, help='旋转中心y坐标')
    parser.add_argument('--n_jobs', type=int, default=None, help='并行处理的作业数')
    parser.add_argument('--cache', action='store_true', help='是否缓存计算结果')
    parser.add_argument('--pair_downsample', type=float, default=0.1, help='图像对下采样比例')
    args = parser.parse_args()
    
    # 设置配置
    class Config:
        def __init__(self, args):
            self.data_path = args.data_path
            self.high_res_size = args.high_res_size
            self.use_fix_template = args.use_fix_template
            self.pair_downsample = args.pair_downsample
    
    config = Config(args)
    
    # 设置图像变换
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("加载数据集...")
    images1, images2, labels1, labels2 = load_and_prepare_dataset(config, transform)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置旋转中心
    rot_center = [args.rot_center_x, args.rot_center_y]
    
    # 设置图像大小
    img_size = [args.high_res_size, args.high_res_size]
    
    # 检查是否有缓存
    cache_file = "intrinsic_search_cache.pkl"
    if args.cache and os.path.exists(cache_file):
        print(f"加载缓存数据: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        optimal_intrinsic = cache_data['optimal_intrinsic']
        min_loss = cache_data['min_loss']
    else:
        # 执行网格搜索
        print("执行网格搜索...")
        fx_range = (args.fx_min, args.fx_max, args.fx_steps)
        fy_range = (args.fy_min, args.fy_max, args.fy_steps)
        flag_value = [-1.0]  # 正负旋转方向
        
        optimal_intrinsic, min_loss = grid_search_intrinsic(
            images1, images2, labels1, labels2,
            fx_range, fy_range, flag_value,
            rot_center, img_size, args.n_jobs
        )
        
        # 保存缓存
        if args.cache:
            cache_data = {
                'optimal_intrinsic': optimal_intrinsic,
                'min_loss': min_loss
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
    
    # 精细优化
    if args.fine_tune:
        print("执行精细优化...")
        # 创建边界
        fx_init, fy_init, flag = optimal_intrinsic
        fx_init = 0.0206
        fy_init = 0.0207
        flag = -1.0
        bounds = [(fx_init * 0.95, fx_init * 1.05), (fy_init * 0.95, fy_init * 1.05)]
        
        optimal_intrinsic, min_loss = optimize_intrinsic_fine(
            images1, images2, labels1, labels2,
            optimal_intrinsic, rot_center, img_size,
            device, bounds
        )
    
    # 输出最终结果
    fx, fy, flag = optimal_intrinsic
    print("\n========== 最终结果 ==========")
    print(f"最优内参: fx={fx:.6f}, fy={fy:.6f}, flag={flag}")
    print(f"最小损失值: {min_loss:.6f}")
    print("==============================\n")
    
    # 保存结果
    result_file = "optimal_intrinsic_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"数据集: {args.data_path}\n")
        f.write(f"高分辨率图像大小: {args.high_res_size}x{args.high_res_size}\n")
        f.write(f"旋转中心: ({args.rot_center_x}, {args.rot_center_y})\n")
        f.write(f"最优内参: fx={fx:.6f}, fy={fy:.6f}, flag={flag}\n")
        f.write(f"最小损失值: {min_loss:.6f}\n")
        f.write(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"结果已保存到: {result_file}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()