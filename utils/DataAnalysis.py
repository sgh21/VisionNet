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