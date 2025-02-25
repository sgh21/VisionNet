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
    # 使用matplotlib内置样式
    plt.style.use('classic')  # 或使用其他内置样式如 'bmh', 'ggplot', 'default'
    
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    def plot_distribution(ax, data, title, xlabel):
        # 计算数据统计量
        mean = np.mean(data)
        std = np.std(data)
        
        # 使用Freedman-Diaconis规则计算bin宽度
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1/3))
        n_bins = int((max(data) - min(data)) / bin_width)
        n_bins = min(max(n_bins, 50), 100)  # 限制bins范围
        
        # 绘制直方图
        # 改进直方图样式
        ax.hist(data, bins=n_bins, density=True, alpha=0.7, 
                color='lightblue', edgecolor='darkblue', linewidth=0.8,
                label='Histogram')
        
        # 设置x轴范围
        x_min = mean - 4*std
        x_max = mean + 4*std
        x = np.linspace(x_min, x_max, 200)
        
        # 拟合参数
        optimized_params = fit_data(data, dist_type=dtype, optimize_type=optimize_type)
        
        if dtype == 'laplace':
            loc, scale = optimized_params
            pdf = laplace.pdf(x, loc, scale)
            label = f'Laplace\nloc={loc:.3f}\nscale={scale:.3f}'
        else:
            mean, std = optimized_params
            pdf = norm.pdf(x, mean, std)
            label = f'Gaussian\nμ={mean:.3f}\nσ={std:.3f}'
            
        # 改进拟合曲线样式
        ax.plot(x, pdf, '-', color='red', linewidth=2, label=label)
        
        # 设置更多图形属性
        ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(x_min, x_max)

    # 绘制三个维度的分布
    plot_distribution(ax1, errors_x, 'X Error Distribution', 'Error (mm)')
    plot_distribution(ax2, errors_y, 'Y Error Distribution', 'Error (mm)')
    plot_distribution(ax3, errors_rz, 'Rz Error Distribution', 'Error (deg)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# def plot_error_distribution(errors_x, errors_y, errors_rz, save_path, dtype='gaussian', optimize_type='mse'):
#     """绘制误差分布直方图和分布拟合曲线"""
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     # 绘制拉普拉斯分布
#     def plot_with_laplace(ax, data, title, xlabel):
#         # 使用拟合方法优化参数
#         optimized_params = fit_data(data, dist_type='laplace', optimize_type=optimize_type)
#         loc, scale = optimized_params
        
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
#         # 使用拟合方法优化参数
#         optimized_params = fit_data(data, dist_type='gaussian', optimize_type=optimize_type)
#         mean, std = optimized_params
        
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