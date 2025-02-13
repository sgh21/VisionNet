import torch
import torch.nn as nn

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