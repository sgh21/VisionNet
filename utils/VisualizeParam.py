import torch
import torch.nn as nn


def visualize_images(images, titles=None, figsize=None, cmap=None, rows=None, cols=None, save_path=None, display=True, block=True):
    """
    通用图像可视化函数，可展示任意数量的图像并自动排布

    Args:
        images (list): 图像列表，每个元素是一个numpy数组
        titles (list): 标题列表，与images列表一一对应。若为None则不显示标题
        figsize (tuple): 图形大小 (width, height)，若为None则自动计算
        cmap (str/list): 颜色映射，可以是单一字符串应用到所有图像，或者列表为每个图像单独指定
        rows (int): 自定义行数，若为None则自动计算
        cols (int): 自定义列数，若为None则自动计算
        save_path (str): 保存路径，若为None则不保存
        display (bool): 是否尝试在屏幕上显示图像，默认为True
        block (bool): 显示图像时是否阻塞程序执行，默认为True

    Returns:
        matplotlib.figure.Figure: 生成的图像对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # 处理空输入
    if not images:
        raise ValueError("必须提供至少一张图片")
    
    # 获取图片数量
    n_images = len(images)
    
    # 处理标题
    if titles is None:
        titles = [f"图像 {i+1}" for i in range(n_images)]
    elif len(titles) != n_images:
        raise ValueError("标题数量必须与图像数量相同")
    
    # 自动计算行列数
    if rows is None and cols is None:
        # 自动确定行列数：尽量接近正方形布局
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # 自动计算图形大小
    if figsize is None:
        # 根据图像数量和行列数计算合适的图形大小
        base_size = 3  # 基础单元大小
        figsize = (cols * base_size, rows * base_size)
    
    # 创建图形和子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    # 处理颜色映射
    if cmap is None:
        # 默认使用灰度图显示单通道图像，彩色图显示多通道图像
        cmaps = [None] * n_images
    elif isinstance(cmap, str):
        # 对所有图像使用相同的颜色映射
        cmaps = [cmap] * n_images
    else:
        # 对每个图像使用指定的颜色映射
        cmaps = cmap
        if len(cmaps) != n_images:
            raise ValueError("颜色映射列表长度必须等于图像数量")
    
    # 绘制图像
    for i in range(n_images):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        # 获取当前图像
        img = images[i]
        
        # 自动判断使用的颜色映射
        if cmaps[i] is None and (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
            # 单通道图像默认使用灰度
            curr_cmap = 'gray'
        else:
            curr_cmap = cmaps[i]
        
        # 显示图像
        ax.imshow(img, cmap=curr_cmap)
        ax.set_title(titles[i])
        
        # 关闭刻度
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 隐藏多余的子图
    for i in range(n_images, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    # 尝试显示图像
    if display:
        try:
            if block:
                plt.show()  # 阻塞显示
            else:
                plt.show(block=False)  # 非阻塞显示
                plt.pause(0.1)  # 确保图像渲染
        except Exception as e:
            print(f"无法显示图像: {e}, 请检查matplotlib配置或环境")
            print("提示: 如果在无GUI环境下运行，可以将display参数设置为False")
    
    return fig

def visualize_conv_filters(model: nn.Module, layer_name: str = 'conv1', max_filters: int = 8):
    import matplotlib.pyplot as plt
    import numpy as np
    
    layer = getattr(model, layer_name, None)
    if layer is None:
        print(f"❌ 未找到名为 {layer_name} 的层.")
        return
    if not isinstance(layer, nn.Conv2d):
        print(f"❌ {layer_name} 不是一个 Conv2d 层.")
        return
    
    with torch.no_grad():
        weights = layer.weight.cpu().numpy()
    
    out_channels, in_channels, kernel_h, kernel_w = weights.shape
    num_filters_to_show = min(out_channels, max_filters)
    
    fig, axes = plt.subplots(1, num_filters_to_show, figsize=(3 * num_filters_to_show, 3))
    
    for i in range(num_filters_to_show):
        ax = axes[i] if num_filters_to_show > 1 else axes
        # filter_ = weights[i, 0]  # 只取第1个通道
        filter_ = np.mean(weights[i], axis=0)  # 多通道做平均
        fmin, fmax = filter_.min(), filter_.max()
        filter_ = (filter_ - fmin) / (fmax - fmin + 1e-5)
        ax.imshow(filter_, cmap='gray')
        ax.set_title(f'{layer_name} #{i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_parameter_histogram(model: nn.Module, param_name: str):
    import matplotlib.pyplot as plt
    import numpy as np
    
    named_params = dict(model.named_parameters())
    if param_name not in named_params:
        print(f"❌ 模型中无此参数: {param_name}")
        return
    
    param_tensor = named_params[param_name].data.cpu().numpy().ravel()
    plt.figure(figsize=(6, 4))
    plt.hist(param_tensor, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of [{param_name}]")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def print_parameter_statistics(model: nn.Module):
    for name, param in model.named_parameters():
        data = param.data
        print(f"===> Param: {name}")
        print(f"     shape: {tuple(data.shape)}")
        print(f"     mean: {data.mean().item():.6f}, std: {data.std().item():.6f}, "
              f"min: {data.min().item():.6f}, max: {data.max().item():.6f}\n")