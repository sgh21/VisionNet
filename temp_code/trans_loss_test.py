import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import time
import argparse
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torchvision.transforms as transforms
from utils.TransUtils import TerraceMapGenerator, ssim_pytorch, PatchBasedIlluminationAlignment
from scipy.optimize import minimize

class TransformationLossAnalyzer:
    """
    Analyze MSE loss of image translation transformations, and generate terrain maps and univariate curves
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', loss_type='mse', ssim_weight=1.0):
        """
        Initialize analyzer
        
        Args:
            device: Computing device, GPU or CPU
            loss_type: Loss type, 'mse', 'ssim', or 'combined'
            ssim_weight: Weight for SSIM loss when using 'combined' loss type
        """
        self.device = device
        self.loss_type = loss_type.lower()
        self.ssim_weight = ssim_weight
        
        # Validate loss type
        valid_loss_types = ['mse', 'ssim', 'combined']
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}. 有效选项: {valid_loss_types}")
        
        from config import EXPANSION_SIZE
        # Initialize terrace map generator
        self.terrace_map_generator = TerraceMapGenerator(
            intensity_scaling=[0.0, 0.6, 0.8, 1.0],
            edge_enhancement=2.0,
            expansion_size=EXPANSION_SIZE,
        )
        
        # self.illumination_alignment = PatchBasedIlluminationAlignment(
        #     window_size=4,
        #     kernel_size=8,
        #     keep_variance=True,
        # )
        print(f"使用设备: {self.device}")
        print(f"损失函数类型: {self.loss_type}")
        if self.loss_type == 'combined':
            print(f"SSIM权重: {self.ssim_weight}")

    def load_images(self, template_path, input_path, size=None, use_gradients=False):
        """
        加载模板图像和输入图像
        
        Args:
            template_path: 模板图像路径
            input_path: 输入图像路径
            size: 可选，调整图像大小
            use_gradients: 是否返回梯度图而非原始图像
            
        Returns:
            template_tensor, input_tensor: 张量格式的模板和输入图像
        """
        # 加载图像
        template_img = Image.open(template_path).convert('RGB')
        input_img = Image.open(input_path).convert('RGB')
        
        # 调整大小（如需要）
        if size is not None:
            template_img = template_img.resize(size)
            input_img = input_img.resize(size)
        
        # 转换为张量
        to_tensor = transforms.ToTensor()
        template_tensor = to_tensor(template_img).to(self.device)
        input_tensor = to_tensor(input_img).to(self.device)
        
        # 添加批处理维度
        if template_tensor.dim() == 3:
            template_tensor = template_tensor.unsqueeze(0)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # 保存原始图像用于可视化
        if use_gradients:
            self.original_template = template_tensor.clone()
            self.original_input = input_tensor.clone()
            
            # 计算梯度幅值
            template_tensor = self.compute_gradient_magnitude(template_tensor)
            input_tensor = self.compute_gradient_magnitude(input_tensor)
                
        return template_tensor, input_tensor
    
    # def load_images(self, template_path, input_path, size=None):
    #     """
    #     Load template and input images
        
    #     Args:
    #         template_path: Template image path
    #         input_path: Input image path
    #         size: Optional, resize images
            
    #     Returns:
    #         template_tensor, input_tensor: Tensor format of template and input images
    #     """
    #     # Load images
    #     template_img = Image.open(template_path).convert('RGB')
    #     input_img = Image.open(input_path).convert('RGB')
        
    #     # Resize if needed
    #     if size is not None:
    #         template_img = template_img.resize(size)
    #         input_img = input_img.resize(size)
        
    #     # Convert to tensors
    #     to_tensor = transforms.ToTensor()
    #     template_tensor = to_tensor(template_img).to(self.device)
    #     input_tensor = to_tensor(input_img).to(self.device)
        
    #     # Add batch dimension
    #     if template_tensor.dim() == 3:
    #         template_tensor = template_tensor.unsqueeze(0)
    #     if input_tensor.dim() == 3:
    #         input_tensor = input_tensor.unsqueeze(0)
            
    #     return template_tensor, input_tensor
    
    def load_touch_masks(self, template_mask_path, input_mask_path, size=None):
        """
        Load touch mask images
        
        Args:
            template_mask_path: Template mask path
            input_mask_path: Input mask path
            size: Optional, resize masks
            
        Returns:
            template_mask_tensor, input_mask_tensor: Tensor format of masks
        """
        if template_mask_path is None or input_mask_path is None:
            return None, None
            
        # Load masks
        template_mask_img = Image.open(template_mask_path).convert('L')
        input_mask_img = Image.open(input_mask_path).convert('L')
        
        # Resize if needed
        if size is not None:
            template_mask_img = template_mask_img.resize(size)
            input_mask_img = input_mask_img.resize(size)
            
        return template_mask_img, input_mask_img

    def create_terrace_weight_maps(self, template_mask, input_mask, serial='default'):
        """
        Create weight maps using TerraceMapGenerator
        
        Args:
            template_mask: Template mask tensor
            input_mask: Input mask tensor
            serial: Serial number for TerraceMapGenerator
            
        Returns:
            template_weight, input_weight: Weight maps
        """
        
        # Generate terrace weight maps
        template_weight = self.terrace_map_generator(template_mask, serial=serial)
        input_weight = self.terrace_map_generator(input_mask, serial=serial)
        
        # Ensure they are tensors and on the correct device
        if not isinstance(template_weight, torch.Tensor):
            to_tensor = transforms.ToTensor()
            template_weight = to_tensor(template_weight).to(self.device)
            input_weight = to_tensor(input_weight).to(self.device)
        else:
            template_weight = template_weight.to(self.device)
            input_weight = input_weight.to(self.device)
        
        # Add batch dimension
        if template_weight.dim() == 3:
            template_weight = template_weight.unsqueeze(0)
        if input_weight.dim() == 3:
            input_weight = input_weight.unsqueeze(0)
            
        return template_weight, input_weight
    def compute_gradient_magnitude(self, image_tensor):
        """
        计算图像的梯度幅值
        
        Args:
            image_tensor: 输入图像张量 [B, C, H, W]
            
        Returns:
            梯度幅值图 [B, 1, H, W]
        """
        # 定义Sobel滤波器
        device = image_tensor.device
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        B, C, H, W = image_tensor.shape
        grad_magnitude = torch.zeros((B, 1, H, W), device=device)
        
        # 对彩色图像，计算各通道梯度后取平均
        for c in range(C):
            # 提取单通道
            channel = image_tensor[:, c:c+1, :, :]
            
            # 使用padding以避免边缘效应
            pad = torch.nn.functional.pad(channel, (1, 1, 1, 1), mode='replicate')
            
            # 计算x和y方向的梯度
            grad_x = torch.nn.functional.conv2d(pad, sobel_x)
            grad_y = torch.nn.functional.conv2d(pad, sobel_y)
            
            # 计算梯度幅值并累加
            grad_magnitude += torch.sqrt(grad_x**2 + grad_y**2)
        
        # 对所有通道求平均
        grad_magnitude = grad_magnitude / C
        
        return grad_magnitude
    def transform_image(self, image, tx, ty):
        """
        Apply translation transformation to image
        
        Args:
            image: Input image tensor [B, C, H, W]
            tx: x-direction translation in pixels
            ty: y-direction translation in pixels
            
        Returns:
            Transformed image
        """
        B, C, H, W = image.shape
        device = image.device
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # Expand grid coordinates to batch dimension
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)
        
        # Calculate translation (normalized to [-1, 1] range)
        norm_tx = 2 * tx / W  # normalized x-direction translation
        norm_ty = 2 * ty / H  # normalized y-direction translation
        
        # Apply translation
        grid_x = grid_x - norm_tx
        grid_y = grid_y - norm_ty
        
        # Combine into sampling grid
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Use grid_sample to implement bilinear interpolation
        return torch.nn.functional.grid_sample(
            image, 
            grid, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
    
    def transform_mask(self, mask, tx, ty):
        """
        Apply translation transformation to mask (using nearest interpolation to maintain binary nature)
        
        Args:
            mask: Input mask tensor [B, 1, H, W]
            tx: x-direction translation in pixels
            ty: y-direction translation in pixels
            
        Returns:
            Transformed mask
        """
        B, C, H, W = mask.shape
        device = mask.device
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # Expand grid coordinates to batch dimension
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)
        
        # Calculate translation (normalized to [-1, 1] range)
        norm_tx = 2 * tx / W  # normalized x-direction translation
        norm_ty = 2 * ty / H  # normalized y-direction translation
        
        # Apply translation
        grid_x = grid_x - norm_tx
        grid_y = grid_y - norm_ty
        
        # Combine into sampling grid
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Use grid_sample to implement nearest interpolation (to maintain binary nature)
        return torch.nn.functional.grid_sample(
            mask, 
            grid, 
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )
    
    def calculate_weighted_mse_loss(self, template, transformed_input, template_weight=None, transformed_input_weight=None):
        """
        Calculate weighted MSE loss
        
        Args:
            template: Template image
            transformed_input: Transformed input image
            template_weight: Template weight map
            transformed_input_weight: Transformed input weight map
            
        Returns:
            Weighted MSE loss
        """
        # Calculate MSE loss
        mse_loss = nn.MSELoss(reduction='none')(template, transformed_input)
        
        # If there are weight maps, apply weights
        if template_weight is not None and transformed_input_weight is not None:
            # Combine weights (take union of the two weight maps)
            combined_weight = torch.max(template_weight, transformed_input_weight)
            
            # Normalize weights so their sum equals pixel count
            B, C, H, W = template.shape
            pixel_count = H * W
            weight_sum = combined_weight.sum(dim=(2, 3), keepdim=True)
            normalized_weight = combined_weight * (pixel_count / (weight_sum + 1e-8))
            
            # Expand weights to match channel count
            if normalized_weight.shape[1] == 1 and C > 1:
                normalized_weight = normalized_weight.expand(-1, C, -1, -1)
            
            # Apply weights
            weighted_mse = (mse_loss * normalized_weight).sum() / (B * C * H * W)
            return weighted_mse.item()
        else:
            # If no weight maps, return regular MSE
            return mse_loss.mean().item()

    def calculate_weighted_ssim_loss(self, template, transformed_input, template_weight=None, transformed_input_weight=None):
        """
        Calculate weighted SSIM loss (1-SSIM to convert to loss)
        使用权重图对SSIM损失进行加权，与MSE权重计算类似
        
        Args:
            template: Template image
            transformed_input: Transformed input image
            template_weight: Template weight map
            transformed_input_weight: Transformed input weight map
            
        Returns:
            Weighted SSIM loss (1-SSIM)
        """
        # 确定图像是否归一化到[0,1]范围
        is_normalized = template.max() <= 1.1
        data_range = 1.0 if is_normalized else 255.0
        
        # 如果没有权重图，执行常规的SSIM计算
        if template_weight is None or transformed_input_weight is None:
            # 计算SSIM (返回0到1之间的值，1表示完全相同)
            ssim_value = ssim_pytorch(template, transformed_input, 
                                    data_range=data_range, 
                                    full=False).item()
            
            # 转换为损失 (1-SSIM，其中0表示完全相同)
            ssim_loss = 1.0 - ssim_value
            return ssim_loss
        else:
            # 使用full=True，获取每个位置的SSIM值而不是平均值
            _, ssim_map = ssim_pytorch(template, transformed_input, 
                                data_range=data_range, 
                                full=True)
            
            # ssim_map的形状为[B, 1, H, W]
            # 将SSIM结果转换为损失图 (1-SSIM)
            ssim_loss_map = 1.0 - ssim_map
            
            # 与MSE损失类似，结合权重图
            # 合并权重（取两个权重图的并集）
            combined_weight = torch.max(template_weight, transformed_input_weight)
            
            # 归一化权重，使其和等于像素数
            B, _, H, W = template.shape
            pixel_count = H * W
            weight_sum = combined_weight.sum(dim=(2, 3), keepdim=True)
            normalized_weight = combined_weight * (pixel_count / (weight_sum + 1e-8))
            
            # 应用权重到SSIM损失图
            weighted_ssim_loss = (ssim_loss_map * normalized_weight).sum() / (B * H * W)
            
            return weighted_ssim_loss.item()
            
    def calculate_combined_loss(self, template, transformed_input, template_weight=None, transformed_input_weight=None):
        """
        Calculate combined loss (weighted sum of MSE and SSIM losses)
        
        Args:
            template: Template image
            transformed_input: Transformed input image
            template_weight: Template weight map
            transformed_input_weight: Transformed input weight map
            
        Returns:
            Combined loss
        """
        mse_loss = self.calculate_weighted_mse_loss(
            template, transformed_input, template_weight, transformed_input_weight
        )
        
        ssim_loss = self.calculate_weighted_ssim_loss(
            template, transformed_input, template_weight, transformed_input_weight
        )
        
        # Combine losses with appropriate weighting
        # MSE typically ranges from 0 to 1 for normalized images, SSIM loss also ranges from 0 to 1
        combined_loss = mse_loss + self.ssim_weight * ssim_loss
        
        return combined_loss
    
    def calculate_loss(self, template, transformed_input, template_weight=None, transformed_input_weight=None):
        """
        Calculate loss based on selected loss type
        
        Args:
            template: Template image
            transformed_input: Transformed input image
            template_weight: Template weight map
            transformed_input_weight: Transformed input weight map
            
        Returns:
            Loss value
        """
        # transformed_input = self.illumination_alignment(transformed_input, template)
        if self.loss_type == 'mse':
            return self.calculate_weighted_mse_loss(
                template, transformed_input, template_weight, transformed_input_weight
            )
        elif self.loss_type == 'ssim':
            return self.calculate_weighted_ssim_loss(
                template, transformed_input, template_weight, transformed_input_weight
            )
        elif self.loss_type == 'combined':
            return self.calculate_combined_loss(
                template, transformed_input, template_weight, transformed_input_weight
            )
        else:
            # Should never happen due to validation in __init__
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
    def analyze_transformation_loss(self, 
                              template_img, 
                              input_img, 
                              x_range=(-50, 50), 
                              y_range=(-50, 50), 
                              step=1.0,
                              template_weight=None,
                              input_weight=None):
        """
        Analyze translation transformation loss
        
        Args:
            template_img: Template image tensor
            input_img: Input image tensor
            x_range: x-direction translation range
            y_range: y-direction translation range
            step: Translation step size (can be float for sub-pixel precision)
            template_weight: Template weight map
            input_weight: Input weight map
            
        Returns:
            x_values, y_values, loss_grid: Data for plotting
        """
        # Prepare data structures with floating point step size support
        x_values = np.arange(x_range[0], x_range[1] + step/2, step).tolist()  # 加上step/2确保包含end point
        y_values = np.arange(y_range[0], y_range[1] + step/2, step).tolist()
        loss_grid = np.zeros((len(y_values), len(x_values)))
        
        # Timing
        start_time = time.time()
        
        # Calculate loss for each translation position
        for i, ty in enumerate(tqdm(y_values, desc="Computing Y-direction loss")):
            for j, tx in enumerate(x_values):
                # Apply translation to input image
                transformed_input = self.transform_image(input_img, tx, ty)
                
                # Apply translation to input weight map (if it exists)
                transformed_input_weight = None
                if input_weight is not None:
                    transformed_input_weight = self.transform_mask(input_weight, tx, ty)
                
                # Calculate loss
                loss = self.calculate_loss(
                    template_img, 
                    transformed_input,
                    template_weight=template_weight,
                    transformed_input_weight=transformed_input_weight
                )
                
                # Store loss value
                loss_grid[i, j] = loss
        
        # Calculate total time
        total_time = time.time() - start_time
        total_iterations = len(x_values) * len(y_values)
        print(f"Total computation time: {total_time:.2f} seconds, Average per iteration: {total_time/total_iterations*1000:.2f} ms")
        
        # Find position of minimum loss
        min_idx = np.argmin(loss_grid.flatten())
        min_i, min_j = np.unravel_index(min_idx, loss_grid.shape)
        min_tx, min_ty = x_values[min_j], y_values[min_i]
        min_loss = loss_grid[min_i, min_j]
        
        print(f"Minimum loss point: tx={min_tx:.2f}, ty={min_ty:.2f}, loss={min_loss:.6f}")
        
        return x_values, y_values, loss_grid, (min_tx, min_ty, min_loss)
    
    def objective_function(self, params, template_img, input_img, template_weight=None, input_weight=None):
        """
        Objective function for optimization
        
        Args:
            params: [tx, ty] translation parameters
            template_img: Template image tensor
            input_img: Input image tensor
            template_weight: Template weight map
            input_weight: Input weight map
            
        Returns:
            loss: Loss value
        """
        tx, ty = params
        
        # Apply translation to input image
        transformed_input = self.transform_image(input_img, tx, ty)
        
        # Apply translation to input weight map (if it exists)
        transformed_input_weight = None
        if input_weight is not None:
            transformed_input_weight = self.transform_mask(input_weight, tx, ty)
        
        # Calculate loss
        loss = self.calculate_loss(
            template_img, 
            transformed_input,
            template_weight=template_weight,
            transformed_input_weight=transformed_input_weight
        )
        
        return loss
    def optimize_transformation(self, 
                          template_img, 
                          input_img, 
                          x_range=(-50, 50), 
                          y_range=(-50, 50),
                          template_weight=None,
                          input_weight=None,
                          method='L-BFGS-B',
                          grid_search_points=100):
        """
        Optimize translation parameters using gradient-based and grid search methods
        
        Args:
            template_img: Template image tensor
            input_img: Input image tensor
            x_range: x-direction translation range
            y_range: y-direction translation range
            template_weight: Template weight map
            input_weight: Input weight map
            method: Optimization method for scipy.optimize.minimize
            grid_search_points: Number of points for initial grid search
            
        Returns:
            result: Optimization result
            grid_result: Grid search result (x_values, y_values, loss_grid, min_point)
        """
        print("Starting combined optimization (Grid Search + Gradient Descent)")
        
        # Step 1: Perform a coarse grid search to find good initial point
        x_step = (x_range[1] - x_range[0]) / grid_search_points
        y_step = (y_range[1] - y_range[0]) / grid_search_points
        
        # 浮点步长不需要取整
        # if x_step < 1:
        #     x_step = 1
        # if y_step < 1:
        #     y_step = 1
        # 
        # x_step = int(max(1, x_step))
        # y_step = int(max(1, y_step))
        
        print(f"Grid search with step size: x={x_step:.4f}, y={y_step:.4f}")
        
        grid_result = self.analyze_transformation_loss(
            template_img, 
            input_img,
            x_range=x_range,
            y_range=y_range,
            step=max(x_step, y_step),  # 使用较大的步长保证效率
            template_weight=template_weight,
            input_weight=input_weight
        )
        
        x_values, y_values, loss_grid, (min_tx, min_ty, min_loss) = grid_result
        
        # Step 2: Use the grid search result as initial point for fine optimization
        print(f"Starting gradient-based optimization from initial point: tx={min_tx:.4f}, ty={min_ty:.4f}")
        
        # Define bounds
        bounds = [(x_range[0], x_range[1]), (y_range[0], y_range[1])]
        
        # Run optimization
        start_time = time.time()
        result = minimize(
            self.objective_function,
            [min_tx, min_ty],
            args=(template_img, input_img, template_weight, input_weight),
            method=method,
            bounds=bounds,
            options={'disp': True}
        )
        
        opt_time = time.time() - start_time
        print(f"Gradient-based optimization completed in {opt_time:.2f} seconds")
        print(f"Optimal translation: tx={result.x[0]:.4f}, ty={result.x[1]:.4f}, loss={result.fun:.6f}")
        
        # Return both grid search and optimization results
        return result, grid_result
    
    def visualize_results(self, 
                         x_values, 
                         y_values, 
                         loss_grid, 
                         min_point,
                         template_img, 
                         input_img,
                         opt_result=None,
                         save_dir=None,
                         prefix=""):
        """
        Visualize results, including terrain map and univariate curves
        
        Args:
            x_values: x-direction translation value list
            y_values: y-direction translation value list
            loss_grid: Loss grid
            min_point: Minimum loss point (tx, ty, loss)
            template_img: Template image
            input_img: Input image
            opt_result: Optional, result from gradient-based optimization
            save_dir: Save directory
            prefix: Filename prefix
        """
        # Create save directory
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Extract minimum point information
        min_tx, min_ty, min_loss = min_point
        
        # Extract optimization result if available
        opt_tx, opt_ty, opt_loss = None, None, None
        if opt_result is not None:
            opt_tx, opt_ty = opt_result.x
            opt_loss = opt_result.fun
        
        # 1. Draw 3D terrain map
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare grid data
        X, Y = np.meshgrid(x_values, y_values)
        
        # Plot surface with color mapping
        surf = ax.plot_surface(X, Y, loss_grid, cmap=cm.viridis,
                              linewidth=0, antialiased=True)
        
        # Add marker at the minimum point from grid search
        ax.scatter([min_tx], [min_ty], [min_loss], color='red', s=50, marker='*', 
                  label=f'Grid Min: ({min_tx:.1f}, {min_ty:.1f})')
        
        # Add marker at the minimum point from optimization if available
        if opt_tx is not None:
            # Find the closest z value for the optimized point
            opt_z = self.objective_function([opt_tx, opt_ty], template_img, input_img, 
                                           getattr(self, 'template_weight', None), 
                                           getattr(self, 'input_weight', None))
            ax.scatter([opt_tx], [opt_ty], [opt_z], color='green', s=50, marker='^', 
                      label=f'Opt Min: ({opt_tx:.1f}, {opt_ty:.1f})')
        
        # Set axis labels and title
        ax.set_xlabel('X Translation (pixels)')
        ax.set_ylabel('Y Translation (pixels)')
        ax.set_zlabel(f'{self.loss_type.upper()} Loss')
        ax.set_title(f'Translation Transformation {self.loss_type.upper()} Loss Terrain Map')
        
        # Add legend
        ax.legend()
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}3d_terrain.png"), dpi=300)
        
        plt.close()
        
        # 2. Draw 2D heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(loss_grid, cmap='viridis', origin='lower', 
                  extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
        
        # Add color bar and title
        plt.colorbar(label=f'{self.loss_type.upper()} Loss')
        plt.title(f'Translation Transformation {self.loss_type.upper()} Loss Heatmap')
        plt.xlabel('X Translation (pixels)')
        plt.ylabel('Y Translation (pixels)')
        
        # Mark the minimum point from grid search
        plt.scatter(min_tx, min_ty, color='red', marker='*', s=100, 
                   label=f'Grid Min: ({min_tx:.1f}, {min_ty:.1f})')
        plt.annotate(f'Grid Min: ({min_tx:.1f}, {min_ty:.1f})\nLoss: {min_loss:.6f}', 
                    (min_tx, min_ty), xytext=(min_tx+5, min_ty+5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        # Mark the minimum point from optimization if available
        if opt_tx is not None:
            plt.scatter(opt_tx, opt_ty, color='green', marker='^', s=100,
                       label=f'Opt Min: ({opt_tx:.1f}, {opt_ty:.1f})')
            plt.annotate(f'Opt Min: ({opt_tx:.1f}, {opt_ty:.1f})\nLoss: {opt_loss:.6f}', 
                        (opt_tx, opt_ty), xytext=(opt_tx+5, opt_ty-25),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
        # Add legend
        plt.legend()
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}2d_heatmap.png"), dpi=300)
        
        plt.close()
        
        # 3. Draw univariate curve - X direction
        min_idx_y = np.where(np.array(y_values) == min_ty)[0][0]
        x_slice = loss_grid[min_idx_y, :]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, x_slice)
        plt.scatter(min_tx, min_loss, color='red', marker='*', s=100, 
                   label=f'Grid Min: ({min_tx:.1f}, {min_ty:.1f})')
        
        # Mark the optimization point on the x slice if available
        if opt_tx is not None and opt_ty is not None:
            # Find the closest y value in the grid
            closest_y_idx = np.argmin(np.abs(np.array(y_values) - opt_ty))
            closest_y = y_values[closest_y_idx]
            
            # Get the loss at the optimization x coordinate but at the closest grid y
            opt_x_slice_loss = self.objective_function([opt_tx, closest_y], template_img, input_img,
                                                      getattr(self, 'template_weight', None),
                                                      getattr(self, 'input_weight', None))
            
            plt.scatter(opt_tx, opt_x_slice_loss, color='green', marker='^', s=100,
                       label=f'Opt Min: ({opt_tx:.1f}, {opt_ty:.1f})')
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f'X Direction Translation {self.loss_type.upper()} Loss Curve (Y fixed at {min_ty})')
        plt.xlabel('X Translation (pixels)')
        plt.ylabel(f'{self.loss_type.upper()} Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}x_slice.png"), dpi=300)
        
        plt.close()
        
        # 4. Draw univariate curve - Y direction
        min_idx_x = np.where(np.array(x_values) == min_tx)[0][0]
        y_slice = loss_grid[:, min_idx_x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_values, y_slice)
        plt.scatter(min_ty, min_loss, color='red', marker='*', s=100,
                   label=f'Grid Min: ({min_tx:.1f}, {min_ty:.1f})')
        
        # Mark the optimization point on the y slice if available
        if opt_tx is not None and opt_ty is not None:
            # Find the closest x value in the grid
            closest_x_idx = np.argmin(np.abs(np.array(x_values) - opt_tx))
            closest_x = x_values[closest_x_idx]
            
            # Get the loss at the optimization y coordinate but at the closest grid x
            opt_y_slice_loss = self.objective_function([closest_x, opt_ty], template_img, input_img,
                                                      getattr(self, 'template_weight', None),
                                                      getattr(self, 'input_weight', None))
            
            plt.scatter(opt_ty, opt_y_slice_loss, color='green', marker='^', s=100,
                       label=f'Opt Min: ({opt_tx:.1f}, {opt_ty:.1f})')
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f'Y Direction Translation {self.loss_type.upper()} Loss Curve (X fixed at {min_tx})')
        plt.xlabel('Y Translation (pixels)')
        plt.ylabel(f'{self.loss_type.upper()} Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}y_slice.png"), dpi=300)
        
        plt.close()
        
        # 5. Visualize best transformation result
        # Get the best parameters (from optimization if available, otherwise from grid search)
        best_tx, best_ty = (opt_tx, opt_ty) if opt_tx is not None else (min_tx, min_ty)
        best_loss = opt_loss if opt_loss is not None else min_loss
        
        # Apply best transformation to input image
        best_transformed_input = self.transform_image(input_img, best_tx, best_ty)
            
        # Convert to numpy arrays for visualization
        template_np = template_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        input_np = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        transformed_np = best_transformed_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Draw original images and best transformation result
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(template_np)
        plt.title('Template Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(input_np)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(transformed_np)
        plt.title(f'Best Transformation Result (tx={best_tx:.2f}, ty={best_ty:.2f})')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}transformation_result.png"), dpi=300)
        
        plt.close()
        
        # 6. Draw difference map
        diff = np.abs(template_np - transformed_np)
        diff_normalized = diff / diff.max() if diff.max() > 0 else diff
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(template_np)
        plt.title('Template Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(transformed_np)
        plt.title('Best Transformation Result')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(diff_normalized)
        plt.title(f'Difference Map (Loss: {best_loss:.6f})')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save image
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}difference_map.png"), dpi=300)
        
        plt.close()
        
        # 7. If there are weight maps, visualize them and create vision-touch overlay
        if hasattr(self, 'template_weight') and hasattr(self, 'input_weight'):
            # Convert to numpy arrays for visualization
            template_weight_np = self.template_weight.squeeze().cpu().numpy()
            input_weight_np = self.input_weight.squeeze().cpu().numpy()
            
            # Apply best transformation to input weight
            transformed_weight = self.transform_mask(self.input_weight, best_tx, best_ty)
            transformed_weight_np = transformed_weight.squeeze().cpu().numpy()
            
            # Visualize weight maps
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(template_weight_np, cmap='viridis')
            plt.title('Template Weight Map')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(input_weight_np, cmap='viridis')
            plt.title('Input Weight Map')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(transformed_weight_np, cmap='viridis')
            plt.title('Transformed Input Weight Map')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save weight maps
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{prefix}weight_maps.png"), dpi=300)
            
            plt.close()
            
            # 8. Create vision-touch overlay visualizations
            plt.figure(figsize=(15, 10))
            
            # Create transparent versions of weight maps for overlay (red hue)
            # For template image and weight
            plt.subplot(231)
            plt.imshow(template_np)
            plt.title('Template Vision Image')
            plt.axis('off')
            
            plt.subplot(232)
            plt.imshow(template_weight_np, cmap='inferno')
            plt.title('Template Touch Weight Map')
            plt.axis('off')
            
            plt.subplot(233)
            # Create overlay - template
            plt.imshow(template_np)
            plt.imshow(template_weight_np, cmap='inferno', alpha=0.6)
            plt.title('Template Vision-Touch Overlay')
            plt.axis('off')
            
            # For transformed input image and weight
            plt.subplot(234)
            plt.imshow(transformed_np)
            plt.title('Transformed Vision Image')
            plt.axis('off')
            
            plt.subplot(235)
            plt.imshow(transformed_weight_np, cmap='inferno')
            plt.title('Transformed Touch Weight Map')
            plt.axis('off')
            
            plt.subplot(236)
            # Create overlay - transformed input
            plt.imshow(transformed_np)
            plt.imshow(transformed_weight_np, cmap='inferno', alpha=0.6)
            plt.title('Transformed Vision-Touch Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save vision-touch overlay visualization
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{prefix}vision_touch_overlay.png"), dpi=300)
            
            plt.close()
            
            # 9. Create a direct comparison of the aligned vision-touch overlays
            plt.figure(figsize=(15, 7))
            
            plt.subplot(121)
            # Template overlay
            plt.imshow(template_np)
            plt.imshow(template_weight_np, cmap='inferno', alpha=0.6)
            plt.title('Template Vision-Touch Overlay')
            plt.axis('off')
            
            plt.subplot(122)
            # Transformed input overlay
            plt.imshow(transformed_np)
            plt.imshow(transformed_weight_np, cmap='inferno', alpha=0.6)
            plt.title(f'Transformed Vision-Touch Overlay (tx={best_tx:.2f}, ty={best_ty:.2f})')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save direct comparison
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{prefix}overlay_comparison.png"), dpi=300)
            
            plt.close()
            
            # 10. Create a side-by-side comparison with difference metrics
            plt.figure(figsize=(15, 12))
            
            # Template image
            plt.subplot(331)
            plt.imshow(template_np)
            plt.title('Template Vision Image')
            plt.axis('off')
            
            # Transformed image
            plt.subplot(332)
            plt.imshow(transformed_np)
            plt.title('Transformed Vision Image')
            plt.axis('off')
            
            # Vision difference
            plt.subplot(333)
            plt.imshow(diff_normalized)
            plt.title(f'Vision Difference (Loss: {best_loss:.6f})')
            plt.axis('off')
            
            # Template touch
            plt.subplot(334)
            plt.imshow(template_weight_np, cmap='inferno')
            plt.title('Template Touch Map')
            plt.axis('off')
            
            # Transformed touch
            plt.subplot(335)
            plt.imshow(transformed_weight_np, cmap='inferno')
            plt.title('Transformed Touch Map')
            plt.axis('off')
            
            # Touch difference
            touch_diff = np.abs(template_weight_np - transformed_weight_np)
            touch_diff_norm = touch_diff / touch_diff.max() if touch_diff.max() > 0 else touch_diff
            plt.subplot(336)
            plt.imshow(touch_diff_norm, cmap='inferno')
            plt.title('Touch Difference')
            plt.axis('off')
            
            # Template overlay
            plt.subplot(337)
            plt.imshow(template_np)
            plt.imshow(template_weight_np, cmap='inferno', alpha=0.6)
            plt.title('Template Vision-Touch Overlay')
            plt.axis('off')
            
            # Transformed overlay
            plt.subplot(338)
            plt.imshow(transformed_np)
            plt.imshow(transformed_weight_np, cmap='inferno', alpha=0.6)
            plt.title('Transformed Vision-Touch Overlay')
            plt.axis('off')
            
            # Combined difference (vision + touch)
            combined_diff = (diff.mean(axis=2) + touch_diff) / 2
            combined_diff_norm = combined_diff / combined_diff.max() if combined_diff.max() > 0 else combined_diff
            plt.subplot(339)
            plt.imshow(combined_diff_norm, cmap='inferno')
            plt.title('Combined Vision-Touch Difference')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save comprehensive comparison
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{prefix}comprehensive_comparison.png"), dpi=300)
            
            plt.close()

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Analyze image translation transformation loss')
    
    # Add command line arguments
    # Add command line arguments
    # parser.add_argument('--template', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_3530P_0.png', help='Template image path')
    # parser.add_argument('--input', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/val/val/rgb_images/image_3530P_54.png', help='Input image path')
    # parser.add_argument('--template-mask', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/touch_images_mask_process/gel_image_3530P_0.png', help='Template mask path')
    # parser.add_argument('--input-mask', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/val/val/touch_masks/gel_image_3530P_54.png', help='Input mask path')
    # parser.add_argument('--template-mask', type=str, default=None, help='Template mask path')
    # parser.add_argument('--input-mask', type=str, default=None, help='Input mask path')
    
    parser.add_argument('--template', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_3524P_0.png', help='Template image path')
    parser.add_argument('--input', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/rgb_images/image_3524P_40.png', help='Input image path')
    parser.add_argument('--template-mask', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/touch_images_mask_process/gel_image_3524P_0.png', help='Template mask path')
    parser.add_argument('--input-mask', type=str, default='/home/sgh/data/WorkSpace/VisionNet/dataset/visionnet_train_0411/vision_touch/train/touch_images_mask_process/gel_image_3524P_40.png', help='Input mask path')
    parser.add_argument('--serial', type=str, default='3524P', help='Product serial number for TerraceMapGenerator')
    parser.add_argument('--x-range', type=str, default='-112,112', help='X-direction translation range, format "min,max"')
    parser.add_argument('--y-range', type=str, default='-112,112', help='Y-direction translation range, format "min,max"')

    parser.add_argument('--step', type=int, default=0.4, help='Translation step size')
    parser.add_argument('--size', type=str, default='224, 224', help='Resize images, format "width,height"')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--prefix', type=str, default='', help='Output filename prefix')
    parser.add_argument('--optimize', action='store_true', help='Use gradient-based optimization after grid search')
    parser.add_argument('--grid-points', type=int, default=100, help='Number of grid points for each dimension')
    parser.add_argument('--opt-method', type=str, default='L-BFGS-B', help='Optimization method for scipy.optimize.minimize')
    parser.add_argument('--loss-type', type=str, default='mse', choices=['mse', 'ssim', 'combined'], 
                      help='Loss function type: mse, ssim, or combined')
    parser.add_argument('--ssim-weight', type=float, default=1.0, 
                      help='Weight for SSIM loss when using combined loss type')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse ranges
    x_min, x_max = map(int, args.x_range.split(','))
    y_min, y_max = map(int, args.y_range.split(','))
    
    # Parse size
    size = None
    if args.size:
        width, height = map(int, args.size.split(','))
        size = (width, height)
    
    # Create analyzer with specified loss type
    analyzer = TransformationLossAnalyzer(
        loss_type=args.loss_type,
        ssim_weight=args.ssim_weight
    )
    
    # Load images
    template_img, input_img = analyzer.load_images(args.template, args.input, size, use_gradients=True)
    
    # Process masks and weight maps
    template_weight, input_weight = None, None
    
    # If masks are provided, use TerraceMapGenerator to generate weight maps
    if args.template_mask and args.input_mask:
        print("Loading mask files and generating weight maps using TerraceMapGenerator")
        template_mask, input_mask = analyzer.load_touch_masks(args.template_mask, args.input_mask, size)
        template_weight, input_weight = analyzer.create_terrace_weight_maps(
            template_mask, input_mask, serial=args.serial
        )
        
        # Save weight maps for later visualization
        analyzer.template_weight = template_weight
        analyzer.input_weight = input_weight
    
    # Run analysis
    if args.optimize:
        # Use combined optimization (grid search + gradient-based)
        print(f"Running combined optimization with {args.grid_points} grid points per dimension")
        
        # Adjust step size based on grid points
        x_step = max(1, (x_max - x_min) // args.grid_points)
        y_step = max(1, (y_max - y_min) // args.grid_points)
        
        print(f"Grid search step size: x={x_step}, y={y_step}")
        
        opt_result, grid_result = analyzer.optimize_transformation(
            template_img, 
            input_img,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            template_weight=template_weight,
            input_weight=input_weight,
            method=args.opt_method,
            grid_search_points=args.grid_points
        )
        
        # Extract grid search results
        x_values, y_values, loss_grid, min_point = grid_result
    else:
        # Just do grid search
        print(f"Analyzing translation transformation, X range: [{x_min}, {x_max}], Y range: [{y_min}, {y_max}], Step: {args.step}")
        x_values, y_values, loss_grid, min_point = analyzer.analyze_transformation_loss(
            template_img, 
            input_img,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            step=args.step,
            template_weight=template_weight,
            input_weight=input_weight
        )
        opt_result = None
    
    # Create output directory with loss type in it
    output_dir = os.path.join(args.output_dir, f"{args.loss_type}_loss")
    if args.loss_type == 'combined':
        output_dir = f"{output_dir}_w{args.ssim_weight}"
    
    # Visualize and save results
    print(f"Visualizing results and saving to: {output_dir}")
    analyzer.visualize_results(
        x_values,
        y_values,
        loss_grid,
        min_point,
        template_img,
        input_img,
        opt_result=opt_result if args.optimize else None,
        save_dir=output_dir,
        prefix=args.prefix
    )
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()