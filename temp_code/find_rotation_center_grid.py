import os
import cv2
import numpy as np
import glob
from scipy.optimize import minimize
import matplotlib
# 设置matplotlib使用Agg后端，这是一个非交互式后端，不需要Qt支持
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import argparse
import time
from functools import partial
import pickle
from find_rotation_center import (
    load_images_from_directory, 
    find_reference_image_index, 
    rotate_image_around_center,
    visualize_results
)

def evaluate_rotation_center_mse(params, reference_img, ref_angle, angles, images, roi_size=560):
    """评估给定旋转中心的对齐误差，使用MSE作为度量"""
    cx, cy = params
    
    # 获取图像尺寸
    h, w = reference_img.shape[:2]
    
    # 使用图像中心作为ROI中心
    center_x, center_y = w // 2, h // 2
    
    # 确保ROI不会超出图像边界
    roi_half = min(roi_size // 2, center_x, center_y, w - center_x, h - center_y)
    
    # 提取参考图像中心区域
    roi_x_start = center_x - roi_half
    roi_x_end = center_x + roi_half
    roi_y_start = center_y - roi_half
    roi_y_end = center_y + roi_half
    
    roi_ref = reference_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # 将参考ROI转换为灰度图像（如果是彩色的）
    if len(roi_ref.shape) > 2:  # 彩色图像
        gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
    else:
        gray_ref = roi_ref
    
    # 计算与参考图像的总误差
    total_error = 0
    
    for i, (angle, img) in enumerate(zip(angles, images)):
        # 如果当前图像就是参考图像，则跳过
        if angle == ref_angle:
            continue
        
        # 计算相对于参考角度的旋转角度
        relative_angle = angle - ref_angle
        
        # 计算需要应用的反向旋转角度
        reverse_angle = -relative_angle
        
        # 绕指定中心旋转图像
        rotated = rotate_image_around_center(img, reverse_angle, center=(cx, cy), visualize=False)
        
        # 提取相同位置的ROI
        roi_rotated = rotated[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 计算均方误差
        if roi_ref.shape == roi_rotated.shape:
            # 将旋转后的ROI转换为灰度图像（如果是彩色的）
            if len(roi_rotated.shape) > 2:  # 彩色图像
                gray_rotated = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
            else:
                gray_rotated = roi_rotated
            
            # 确保尺寸匹配
            if gray_ref.shape != gray_rotated.shape:
                # 如果大小不匹配，调整大小
                gray_rotated = cv2.resize(gray_rotated, (gray_ref.shape[1], gray_ref.shape[0]))
            
            # 计算MSE
            # 使用浮点数类型计算以提高精度
            diff = gray_ref.astype(np.float32) - gray_rotated.astype(np.float32)
            mse = np.mean(diff * diff)
            
            # 将MSE添加到总误差
            total_error += mse
        else:
            # 如果ROI超出边界，给予高惩罚值
            total_error += 1e6
    
    return total_error

def evaluate_single_point(args):
    """评估单个旋转中心点，用于并行处理"""
    cx, cy, reference_img, ref_angle, angles, images, roi_size = args
    error = evaluate_rotation_center_mse((cx, cy), reference_img, ref_angle, angles, images, roi_size)
    return (cx, cy, error)

def dynamic_grid_search(images, angles, cx_range, cy_range, max_step=10.0, min_step=0.01, 
                         roi_size=560, n_jobs=None, zoom_factor=0.25):
    """
    使用动态步长网格搜索寻找最优旋转中心
    
    Args:
        images: 图像列表
        angles: 角度列表
        cx_range: (min_cx, max_cx) - x坐标初始搜索范围
        cy_range: (min_cy, max_cy) - y坐标初始搜索范围
        max_step: 最大步长
        min_step: 最小步长
        roi_size: 评估使用的ROI大小
        n_jobs: 并行处理的作业数，None表示使用所有可用核心
        zoom_factor: 每次迭代缩小搜索范围的比例因子
        
    Returns:
        optimal_cx, optimal_cy: 最终最优旋转中心坐标
        error_maps: 不同步长下的误差图列表
        search_history: 搜索过程中的区域和最优点历史
        min_error: 最小误差值
        ref_idx: 参考图像索引
        ref_angle: 参考图像角度
    """
    # 找出角度最接近0的图像作为参考
    ref_idx = find_reference_image_index(angles)
    reference_img = images[ref_idx]
    ref_angle = angles[ref_idx]
    
    print(f"使用角度为 {-ref_angle}° 的图像作为参考 (索引 {ref_idx})")
    
    # 准备多进程计算
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # 存储搜索历史和误差图
    search_history = []
    error_maps = []
    
    # 当前步长和搜索范围
    current_step = max_step
    current_cx_range = cx_range
    current_cy_range = cy_range
    
    # 最优点和误差
    overall_best_cx, overall_best_cy = None, None
    overall_min_error = float('inf')
    
    iteration = 1
    last_step = float('inf')
    # 逐步缩小步长直到达到最小步长
    while current_step >= min_step and (last_step > current_step):
        print(f"\n迭代 {iteration}, 步长: {current_step}")
        print(f"当前搜索范围: X[{current_cx_range[0]:.4f}-{current_cx_range[1]:.4f}] " 
              f"Y[{current_cy_range[0]:.4f}-{current_cy_range[1]:.4f}]")
        last_step = current_step
        # 创建当前步长下的搜索网格
        min_cx, max_cx = current_cx_range
        min_cy, max_cy = current_cy_range
        
        # 确保网格点数不会过多（如果步长较小）
        max_grid_points = 1000  # 限制最大网格点数
        
        # 计算预期的网格点数
        expected_points = ((max_cx - min_cx) / current_step + 1) * ((max_cy - min_cy) / current_step + 1)
        
        # 如果预期点数过多，调整步长
        if expected_points > max_grid_points:
            adjusted_step = np.sqrt((max_cx - min_cx) * (max_cy - min_cy) / max_grid_points)
            adjusted_step = max(adjusted_step, min_step)
            print(f"网格点数过多({expected_points:.0f})，调整步长为: {adjusted_step:.4f}")
            current_step = adjusted_step
        
        cx_values = np.arange(min_cx, max_cx + current_step/2, current_step)
        cy_values = np.arange(min_cy, max_cy + current_step/2, current_step)
        
        print(f"搜索网格大小: {len(cx_values)}x{len(cy_values)} = {len(cx_values)*len(cy_values)} 点")
        
        # 准备参数列表
        args_list = []
        for cx in cx_values:
            for cy in cy_values:
                args_list.append((cx, cy, reference_img, ref_angle, angles, images, roi_size))
        
        # 使用进程池并行计算
        t_start = time.time()
        results = []
        
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for result in tqdm(pool.imap(evaluate_single_point, args_list), 
                               total=len(args_list), 
                               desc=f"步长 {current_step:.4f}"):
                results.append(result)
        
        t_end = time.time()
        print(f"本轮搜索耗时: {t_end - t_start:.2f} 秒")
        
        # 找出最优结果
        best_result = min(results, key=lambda x: x[2])
        best_cx, best_cy, min_error = best_result
        
        print(f"本轮最优点: ({best_cx:.4f}, {best_cy:.4f}), 误差: {min_error:.4f}")
        
        # 更新全局最优解
        if min_error < overall_min_error:
            overall_best_cx, overall_best_cy = best_cx, best_cy
            overall_min_error = min_error
        
        # 创建误差地图
        error_map = np.ones((len(cy_values), len(cx_values))) * np.inf
        for res_cx, res_cy, error in results:
            cx_idx = np.abs(cx_values - res_cx).argmin()
            cy_idx = np.abs(cy_values - res_cy).argmin()
            error_map[cy_idx, cx_idx] = error
        
        # 记录搜索历史
        search_history.append({
            'step': current_step,
            'cx_range': current_cx_range,
            'cy_range': current_cy_range,
            'best_cx': best_cx,
            'best_cy': best_cy,
            'min_error': min_error,
            'cx_values': cx_values,
            'cy_values': cy_values
        })
        
        # 记录误差图
        error_maps.append({
            'step': current_step,
            'error_map': error_map,
            'cx_values': cx_values,
            'cy_values': cy_values
        })
        
        # 更新下一轮搜索范围，将范围缩小到最优点周围
        # 确保新范围不超过当前步长的一定倍数，从而在最优点周围进行精细搜索
        range_width = current_step / zoom_factor
        
        new_min_cx = max(min_cx, best_cx - range_width)
        new_max_cx = min(max_cx, best_cx + range_width)
        new_min_cy = max(min_cy, best_cy - range_width)
        new_max_cy = min(max_cy, best_cy + range_width)
        
        # 更新搜索范围
        current_cx_range = (new_min_cx, new_max_cx)
        current_cy_range = (new_min_cy, new_max_cy)
        
        # 将步长减半（或其他合适的减小方式）
        current_step /= 2
        
        # 确保步长不小于最小步长
        current_step = max(current_step, min_step)
        
        iteration += 1
    
    return overall_best_cx, overall_best_cy, error_maps, search_history, overall_min_error, ref_idx, ref_angle
def visualize_search_process(error_maps, search_history, optimal_cx, optimal_cy, save_dir=None):
    """Visualize the dynamic step size search process, including 3D terrain maps and summary results"""
    from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
    import matplotlib.pyplot as plt  # Import plt within the function to ensure it's available
    
    # For each step size create a 2D heatmap
    for i, error_map_data in enumerate(error_maps):
        step = error_map_data['step']
        error_map = error_map_data['error_map']
        cx_values = error_map_data['cx_values']
        cy_values = error_map_data['cy_values']
        
        # Optimal point for current step
        best_cx = search_history[i]['best_cx']
        best_cy = search_history[i]['best_cy']
        best_error = search_history[i]['min_error']
        
        # Create 2D heatmap
        plt.figure(figsize=(10, 8))
        
        # Draw error heatmap
        plt.imshow(error_map, cmap='viridis', aspect='auto',
                   extent=[cx_values[0], cx_values[-1], cy_values[-1], cy_values[0]])
        
        # Mark current step optimal point
        plt.plot(best_cx, best_cy, 'ro', markersize=10, label=f'Current Best ({best_cx:.4f}, {best_cy:.4f})')
        
        # Mark global optimal point (if not the final round)
        if i < len(error_maps) - 1 or (best_cx != optimal_cx or best_cy != optimal_cy):
            plt.plot(optimal_cx, optimal_cy, 'r*', markersize=12, label=f'Global Best ({optimal_cx:.4f}, {optimal_cy:.4f})')
        
        # If not the final round, show next round search range
        if i < len(search_history) - 1:
            next_min_cx, next_max_cx = search_history[i+1]['cx_range']
            next_min_cy, next_max_cy = search_history[i+1]['cy_range']
            
            # Draw rectangle for next round search range
            rect = plt.Rectangle((next_min_cx, next_min_cy), 
                               next_max_cx - next_min_cx, 
                               next_max_cy - next_min_cy,
                               linewidth=2, edgecolor='white', facecolor='none',
                               label='Next Search Range')
            plt.gca().add_patch(rect)
        
        # Set titles and labels
        plt.title(f'Error Map for Step Size {step:.4f}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar(label='Error Value')
        
        plt.legend()
        
        # Grid lines
        plt.grid(True, alpha=0.3)
        
        # Save 2D image
        if save_dir:
            plt.tight_layout()
            error_map_path = os.path.join(save_dir, f'error_map_2d_step_{step:.4f}.png')
            plt.savefig(error_map_path, dpi=300)
        
        plt.close()
        
        # Create separate 3D terrain map
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        X, Y = np.meshgrid(cx_values, cy_values)
        
        # Process error values for better visualization
        error_map_viz = np.copy(error_map)
        threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)  # Use 95th percentile as threshold
        error_map_viz[error_map_viz > threshold] = threshold
        
        # Draw 3D surface
        surf = ax.plot_surface(X, Y, error_map_viz, cmap='viridis', 
                              edgecolor='none', alpha=0.8,
                              rstride=1, cstride=1, linewidth=0)
        
        # Add contour projection
        cset = ax.contour(X, Y, error_map_viz, zdir='z', offset=np.min(error_map_viz),
                         cmap='viridis', levels=10)
        
        # Mark optimal point
        # Get Z value for best point
        best_cx_idx = np.abs(cx_values - best_cx).argmin()
        best_cy_idx = np.abs(cy_values - best_cy).argmin()
        best_z = error_map[best_cy_idx, best_cx_idx]
        
        # Ensure best point Z value is not infinity
        if not np.isfinite(best_z):
            best_z = np.min(error_map_viz[np.isfinite(error_map_viz)])
            
        # Mark current best point on 3D plot
        ax.scatter([best_cx], [best_cy], [best_z], color='red', s=100, marker='o',
                  label=f'Current Best ({best_cx:.4f}, {best_cy:.4f}, {best_error:.4f})')
        
        # Mark vertical line for better position visibility
        ax.plot([best_cx, best_cx], [best_cy, best_cy], 
               [np.min(error_map_viz), best_z], 'r--', alpha=0.5)
        
        # Set view angle
        ax.view_init(elev=30, azim=-60)
        
        # Title and labels
        ax.set_title(f'3D Error Terrain for Step Size {step:.4f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Error Value')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Error Value')
        
        # Save 3D figure
        if save_dir:
            plt.tight_layout()
            error_3d_path = os.path.join(save_dir, f'error_map_3d_step_{step:.4f}.png')
            plt.savefig(error_3d_path, dpi=300)
        
        plt.close()
    
    # Create summary 3D visualization of all points - non-interactive version (for saving)
    print("Creating 3D visualization of search trajectory...")
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect all search points and best points
    all_points = []
    best_points = []
    for i, (error_map_data, history_data) in enumerate(zip(error_maps, search_history)):
        step = error_map_data['step']
        error_map = error_map_data['error_map']
        cx_values = error_map_data['cx_values']
        cy_values = error_map_data['cy_values']
        
        # Get best point for current step size
        best_cx = history_data['best_cx']
        best_cy = history_data['best_cy']
        best_error = history_data['min_error']
        
        # Record each round's best point
        best_points.append((best_cx, best_cy, best_error))
        
        # Build grid for 3D surface
        X, Y = np.meshgrid(cx_values, cy_values)
        
        # Process error values for better visualization
        error_map_viz = np.copy(error_map)
        threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)
        error_map_viz[error_map_viz > threshold] = threshold
        
        # Use transparency to show different iterations
        alpha = 0.15 + 0.7 * (i / len(error_maps))
        
        # Adjust color to distinguish different step sizes
        cmap = plt.cm.viridis
        color = cmap(i / len(error_maps))
        
        # Draw 3D surface
        surf = ax.plot_surface(X, Y, error_map_viz, 
                              color=color,
                              edgecolor='none', 
                              alpha=alpha,
                              rstride=1, cstride=1, 
                              linewidth=0,
                              label=f'Step {step:.4f}')
        
        # Record all points and their error values for this round
        for x_idx, cx in enumerate(cx_values):
            for y_idx, cy in enumerate(cy_values):
                z = error_map[y_idx, x_idx]
                if np.isfinite(z):
                    all_points.append((cx, cy, z, step))
    
    # Use different colors to mark best points from each round
    best_xs, best_ys, best_zs = zip(*best_points)
    
    # Connect best points to form optimization trajectory
    ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=3, label='Optimization Path')
    
    # Mark best points
    for i, (x, y, z) in enumerate(best_points):
        step = error_maps[i]['step']
        ax.scatter([x], [y], [z], color='red', s=100-i*5, 
                  marker='o', label=f'Step {step:.4f} Best' if i==0 else "")
    
    # Mark global best point
    ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
               color='gold', s=150, marker='*', label='Global Best')
    
    # Set view angle
    ax.view_init(elev=30, azim=-60)
    
    # Set title and labels
    ax.set_title('Global Optimization Trajectory of Dynamic Grid Search')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Error Value')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    # Save 3D figure
    if save_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'global_search_trajectory_3d.png'), dpi=300)
    
    plt.close()
    
    # === Try to create interactive 3D visualization (with different backends) ===
    print("\nCreating interactive 3D visualization...(close window to continue)")
    print("Tip: You can use the mouse to rotate and zoom to view the search trajectory")
    
    # Get current backend and try to switch to an interactive one
    import matplotlib
    current_backend = matplotlib.get_backend()
    
    # Try multiple possible interactive backends
    interactive_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'macosx']
    backend_success = False
    
    for backend in interactive_backends:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            backend_success = True
            print(f"Using interactive backend: {backend}")
            break
        except Exception as e:
            print(f"Cannot use backend {backend}: {e}")
    
    if not backend_success:
        print("Warning: Cannot switch to interactive backend, using current backend:", current_backend)
        import matplotlib.pyplot as plt
    
    try:
        # Create new interactive figure window
        fig = plt.figure(figsize=(14, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Set axis label colors to white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        # Set title color to white
        ax.set_title('Dynamic Grid Search Optimization Trajectory (Close window to continue)', color='white', fontsize=14)
        
        # Initial range
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        min_z = min(p[2] for p in all_points if np.isfinite(p[2]))
        max_z = max(p[2] for p in all_points if np.isfinite(p[2]))
        
        # Set axis range, slightly larger for better viewing
        ax.set_xlim(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1)
        ax.set_ylim(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1)
        ax.set_zlim(min_z - (max_z - min_z) * 0.1, max_z + (max_z - min_z) * 0.1)
        
        # Draw error surfaces for each round
        for i, error_map_data in enumerate(error_maps):
            step = error_map_data['step']
            error_map = error_map_data['error_map']
            cx_values = error_map_data['cx_values']
            cy_values = error_map_data['cy_values']
            
            # Build grid
            X, Y = np.meshgrid(cx_values, cy_values)
            
            # Process error values for better visualization
            error_map_viz = np.copy(error_map)
            threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)
            error_map_viz[error_map_viz > threshold] = threshold
            
            # Use gradient transparency for different iterations
            alpha = 0.1 + 0.6 * (i / len(error_maps))
            
            # Use color gradient for different step sizes
            norm = plt.Normalize(0, len(error_maps)-1)
            cmap = plt.cm.viridis
            color = cmap(norm(i))
            
            # Draw semi-transparent surface
            surf = ax.plot_surface(X, Y, error_map_viz, 
                                  color=color,
                                  edgecolor='none', 
                                  alpha=alpha,
                                  rstride=1, cstride=1, 
                                  linewidth=0)
            
            # Add step size text annotation at top left
            ax.text(min_x, max_y, max_z - i*(max_z-min_z)/20, 
                    f"Step {step:.4f}", 
                    color=cmap(norm(i)), fontsize=10)
        
        # Draw optimization path - connect best points from each round
        ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=4, label='Optimization Path')
        
        # Mark best points from each round
        for i, (x, y, z) in enumerate(best_points):
            step = error_maps[i]['step']
            # Point size decreases as step size decreases
            point_size = 200 - i * 150 / len(best_points)
            ax.scatter([x], [y], [z], color='red', s=point_size, marker='o', alpha=0.7)
            # Add text annotation showing step size
            ax.text(x, y, z, f"  Step:{step:.4f}", color='white', fontsize=8)
        
        # Mark global best point
        ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
                   color='gold', s=200, marker='*', alpha=1.0)
        
        # Add global best point text annotation
        ax.text(optimal_cx, optimal_cy, best_points[-1][2], 
                f"  Global Best\n  ({optimal_cx:.4f}, {optimal_cy:.4f})", 
                color='yellow', fontsize=12)
        
        # Set initial view angle
        ax.view_init(elev=30, azim=-60)
        
        # Add instruction text
        plt.figtext(0.5, 0.01, 
                   "Drag to rotate view,    Scroll to zoom,    Right-click to pan", 
                   ha="center", color='white', fontsize=12)
        
        # Show interactive figure
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Interactive visualization creation failed: {e}")
        print("Will continue processing...")
    
    # Try to restore original backend
    try:
        matplotlib.use(current_backend, force=True)
    except:
        # If failed, default back to Agg backend
        matplotlib.use('Agg', force=True)
    
    # Create 3D point cloud visualization (visualize all search points as a point cloud)
    try:
        import matplotlib.pyplot as plt
        
        # Collect all valid search points
        all_valid_points = [(p[0], p[1], p[2]) for p in all_points if np.isfinite(p[2])]
        if all_valid_points:
            xs, ys, zs = zip(*all_valid_points)
            steps = [p[3] for p in all_points if np.isfinite(p[2])]
            
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create step size color mapping from small to large
            norm = plt.Normalize(min(steps), max(steps))
            cmap = plt.cm.plasma
            colors = [cmap(norm(step)) for step in steps]
            
            # Scatter plot for all points
            scatter = ax.scatter(xs, ys, zs, c=colors, marker='.', alpha=0.5, s=10)
            
            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Step Size')
            
            # Mark optimization path and best points
            ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=3, label='Optimization Path')
            for i, (x, y, z) in enumerate(best_points):
                ax.scatter([x], [y], [z], color='red', s=100-i*5, marker='o')
            
            # Mark global best point
            ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
                      color='gold', s=150, marker='*', label='Global Best')
            
            # Title and labels
            ax.set_title('3D Point Cloud View of All Search Points')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Error Value')
            ax.legend(loc='upper right')
            
            # Save point cloud figure
            if save_dir:
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'search_points_cloud_3d.png'), dpi=300)
            
            plt.close()
    except Exception as e:
        print(f"Point cloud visualization creation failed: {e}")
# def visualize_search_process(error_maps, search_history, optimal_cx, optimal_cy, save_dir=None):
#     """可视化动态步长搜索过程，包括3D地形图和汇总结果"""
#     from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
#     import matplotlib.pyplot as plt  # 在函数内导入plt，确保其可用
#     import matplotlib.font_manager as fm  # 导入字体管理器
    
#     # 设置中文字体支持
#     try:
#         # 尝试使用系统中的中文字体
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
#         plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
#         # 使用自定义字体文件(如果上面的方法不成功，您可以尝试使用此方法)
#         # 在Linux系统上常见的中文字体路径
#         font_paths = [
#             '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
#             '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid字体
#             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto字体
#             '/usr/share/fonts/truetype/arphic/uming.ttc',  # 文鼎明体
#             '/usr/share/fonts/truetype/arphic/ukai.ttc'   # 文鼎楷体
#         ]
        
#         font_found = False
#         for font_path in font_paths:
#             if os.path.exists(font_path):
#                 custom_font = fm.FontProperties(fname=font_path)
#                 font_found = True
#                 break
        
#         if not font_found:
#             print("警告: 未找到合适的中文字体文件，可能会导致中文显示为乱码")
#             custom_font = None
            
#     except Exception as e:
#         print(f"设置中文字体时出错: {e}")
#         print("将尝试继续，但中文可能显示为乱码")
#         custom_font = None
    
#     # 为每个步长创建2D热力图（保留原有功能）
#     for i, error_map_data in enumerate(error_maps):
#         step = error_map_data['step']
#         error_map = error_map_data['error_map']
#         cx_values = error_map_data['cx_values']
#         cy_values = error_map_data['cy_values']
        
#         # 当前步长的最优点
#         best_cx = search_history[i]['best_cx']
#         best_cy = search_history[i]['best_cy']
#         best_error = search_history[i]['min_error']
        
#         # 创建2D热力图
#         plt.figure(figsize=(10, 8))
        
#         # 绘制误差热力图
#         plt.imshow(error_map, cmap='viridis', aspect='auto',
#                    extent=[cx_values[0], cx_values[-1], cy_values[-1], cy_values[0]])
        
#         # 标记当前步长最优点
#         plt.plot(best_cx, best_cy, 'ro', markersize=10, label=f'当前最优 ({best_cx:.4f}, {best_cy:.4f})')
        
#         # 标记全局最优点（如果不是最后一轮）
#         if i < len(error_maps) - 1 or (best_cx != optimal_cx or best_cy != optimal_cy):
#             plt.plot(optimal_cx, optimal_cy, 'r*', markersize=12, label=f'全局最优 ({optimal_cx:.4f}, {optimal_cy:.4f})')
        
#         # 如果不是最后一轮，显示下一轮搜索范围
#         if i < len(search_history) - 1:
#             next_min_cx, next_max_cx = search_history[i+1]['cx_range']
#             next_min_cy, next_max_cy = search_history[i+1]['cy_range']
            
#             # 绘制下一轮搜索范围的矩形
#             rect = plt.Rectangle((next_min_cx, next_min_cy), 
#                                next_max_cx - next_min_cx, 
#                                next_max_cy - next_min_cy,
#                                linewidth=2, edgecolor='white', facecolor='none',
#                                label='下一轮搜索范围')
#             plt.gca().add_patch(rect)
        
#         # 标题和标签
#         if custom_font:
#             plt.title(f'步长 {step:.4f} 的误差地图', fontproperties=custom_font)
#             plt.xlabel('X坐标', fontproperties=custom_font)
#             plt.ylabel('Y坐标', fontproperties=custom_font)
#             cbar = plt.colorbar(label='误差值')
#             cbar.set_label('误差值', fontproperties=custom_font)
#         else:
#             plt.title(f'步长 {step:.4f} 的误差地图')
#             plt.xlabel('X坐标')
#             plt.ylabel('Y坐标')
#             plt.colorbar(label='误差值')
        
#         plt.legend(prop=custom_font)
        
#         # 网格线
#         plt.grid(True, alpha=0.3)
        
#         # 保存2D图
#         if save_dir:
#             plt.tight_layout()
#             error_map_path = os.path.join(save_dir, f'error_map_2d_step_{step:.4f}.png')
#             plt.savefig(error_map_path, dpi=300)
        
#         plt.close()
        
#         # 创建单独的3D地形图
#         fig = plt.figure(figsize=(12, 10))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # 创建网格
#         X, Y = np.meshgrid(cx_values, cy_values)
        
#         # 对误差值进行一些处理，使得可视化效果更好
#         error_map_viz = np.copy(error_map)
#         threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)  # 使用95百分位作为阈值
#         error_map_viz[error_map_viz > threshold] = threshold
        
#         # 绘制3D表面
#         surf = ax.plot_surface(X, Y, error_map_viz, cmap='viridis', 
#                               edgecolor='none', alpha=0.8,
#                               rstride=1, cstride=1, linewidth=0)
        
#         # 添加等高线投影
#         cset = ax.contour(X, Y, error_map_viz, zdir='z', offset=np.min(error_map_viz),
#                          cmap='viridis', levels=10)
        
#         # 标记最优点
#         # 获取最优点对应的Z值
#         best_cx_idx = np.abs(cx_values - best_cx).argmin()
#         best_cy_idx = np.abs(cy_values - best_cy).argmin()
#         best_z = error_map[best_cy_idx, best_cx_idx]
        
#         # 确保最优点的Z值不是无穷大
#         if not np.isfinite(best_z):
#             best_z = np.min(error_map_viz[np.isfinite(error_map_viz)])
            
#         # 在3D图上标记当前最优点
#         ax.scatter([best_cx], [best_cy], [best_z], color='red', s=100, marker='o',
#                   label=f'当前最优 ({best_cx:.4f}, {best_cy:.4f}, {best_error:.4f})')
        
#         # 标记垂直线以便更好地看到位置
#         ax.plot([best_cx, best_cx], [best_cy, best_cy], 
#                [np.min(error_map_viz), best_z], 'r--', alpha=0.5)
        
#         # 设置视角
#         ax.view_init(elev=30, azim=-60)
        
#         # 标题和标签
#         if custom_font:
#             ax.set_title(f'步长 {step:.4f} 的误差3D地形图', fontproperties=custom_font)
#             ax.set_xlabel('X坐标', fontproperties=custom_font)
#             ax.set_ylabel('Y坐标', fontproperties=custom_font)
#             ax.set_zlabel('误差值', fontproperties=custom_font)
#             cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#             cbar.set_label('误差值', fontproperties=custom_font)
#         else:
#             ax.set_title(f'步长 {step:.4f} 的误差3D地形图')
#             ax.set_xlabel('X坐标')
#             ax.set_ylabel('Y坐标')
#             ax.set_zlabel('误差值')
#             fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='误差值')
        
#         # 保存3D图
#         if save_dir:
#             plt.tight_layout()
#             error_3d_path = os.path.join(save_dir, f'error_map_3d_step_{step:.4f}.png')
#             plt.savefig(error_3d_path, dpi=300)
        
#         plt.close()
    
#     # 创建所有点的汇总3D可视化 - 非交互式版本(保存用)
#     print("创建搜索轨迹的3D可视化...")
#     fig = plt.figure(figsize=(14, 12))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 收集所有搜索点和最优点
#     all_points = []
#     best_points = []
#     for i, (error_map_data, history_data) in enumerate(zip(error_maps, search_history)):
#         step = error_map_data['step']
#         error_map = error_map_data['error_map']
#         cx_values = error_map_data['cx_values']
#         cy_values = error_map_data['cy_values']
        
#         # 获取当前步长的最优点
#         best_cx = history_data['best_cx']
#         best_cy = history_data['best_cy']
#         best_error = history_data['min_error']
        
#         # 标记每轮的最优点
#         best_points.append((best_cx, best_cy, best_error))
        
#         # 构建网格以绘制3D表面
#         X, Y = np.meshgrid(cx_values, cy_values)
        
#         # 处理误差值以便更好的可视化
#         error_map_viz = np.copy(error_map)
#         threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)
#         error_map_viz[error_map_viz > threshold] = threshold
        
#         # 使用透明度表示不同迭代轮次
#         alpha = 0.15 + 0.7 * (i / len(error_maps))
        
#         # 调整颜色来区分不同步长
#         cmap = plt.cm.viridis
#         color = cmap(i / len(error_maps))
        
#         # 绘制3D表面
#         surf = ax.plot_surface(X, Y, error_map_viz, 
#                               color=color,
#                               edgecolor='none', 
#                               alpha=alpha,
#                               rstride=1, cstride=1, 
#                               linewidth=0,
#                               label=f'步长 {step:.4f}')
        
#         # 记录此轮的所有点及其误差值
#         for x_idx, cx in enumerate(cx_values):
#             for y_idx, cy in enumerate(cy_values):
#                 z = error_map[y_idx, x_idx]
#                 if np.isfinite(z):
#                     all_points.append((cx, cy, z, step))
    
#     # 使用不同颜色标记每轮的最优点
#     best_xs, best_ys, best_zs = zip(*best_points)
    
#     # 连接最优点形成优化轨迹
#     ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=3, label='优化轨迹')
    
#     # 标记最优点
#     for i, (x, y, z) in enumerate(best_points):
#         step = error_maps[i]['step']
#         ax.scatter([x], [y], [z], color='red', s=100-i*5, 
#                   marker='o', label=f'步长 {step:.4f} 最优点' if i==0 else "")
    
#     # 标记全局最优点
#     ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
#                color='gold', s=150, marker='*', label='全局最优点')
    
#     # 设置视角
#     ax.view_init(elev=30, azim=-60)
    
#     # 设置标题和标签
#     if custom_font:
#         ax.set_title('动态网格搜索的全局优化轨迹', fontproperties=custom_font)
#         ax.set_xlabel('X坐标', fontproperties=custom_font)
#         ax.set_ylabel('Y坐标', fontproperties=custom_font)
#         ax.set_zlabel('误差值', fontproperties=custom_font)
#     else:
#         ax.set_title('动态网格搜索的全局优化轨迹')
#         ax.set_xlabel('X坐标')
#         ax.set_ylabel('Y坐标')
#         ax.set_zlabel('误差值')
    
#     # 添加图例
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = []
#     unique_handles = []
#     for handle, label in zip(handles, labels):
#         if label not in unique_labels:
#             unique_labels.append(label)
#             unique_handles.append(handle)
#     if custom_font:
#         ax.legend(unique_handles, unique_labels, loc='upper right', prop=custom_font)
#     else:
#         ax.legend(unique_handles, unique_labels, loc='upper right')
    
#     # 保存3D图
#     if save_dir:
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, 'global_search_trajectory_3d.png'), dpi=300)
    
#     plt.close()
    
#     # === 尝试创建交互式3D可视化（使用不同后端尝试） ===
#     print("\n创建交互式3D可视化...(关闭窗口继续)")
#     print("提示: 您可以使用鼠标旋转、缩放图形以查看搜索轨迹")
    
#     # 获取当前后端并尝试切换到交互式后端
#     import matplotlib
#     current_backend = matplotlib.get_backend()
    
#     # 尝试多种可能的交互式后端
#     interactive_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'macosx']
#     backend_success = False
    
#     for backend in interactive_backends:
#         try:
#             matplotlib.use(backend, force=True)
#             import matplotlib.pyplot as plt
#             backend_success = True
#             print(f"使用交互式后端: {backend}")
#             break
#         except Exception as e:
#             print(f"无法使用后端 {backend}: {e}")
    
#     if not backend_success:
#         print("警告: 无法切换到交互式后端，将使用当前后端:", current_backend)
#         import matplotlib.pyplot as plt
    
#     try:
#         # 创建新的交互式图形窗口
#         fig = plt.figure(figsize=(14, 12), facecolor='black')
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_facecolor('black')
        
#         # 设置轴标签颜色为白色
#         ax.xaxis.label.set_color('white')
#         ax.yaxis.label.set_color('white')
#         ax.zaxis.label.set_color('white')
#         ax.tick_params(axis='x', colors='white')
#         ax.tick_params(axis='y', colors='white')
#         ax.tick_params(axis='z', colors='white')
        
#         # 设置标题颜色为白色
#         if custom_font:
#             ax.set_title('动态网格搜索优化轨迹 (关闭窗口继续)', color='white', fontsize=14, fontproperties=custom_font)
#         else:
#             ax.set_title('动态网格搜索优化轨迹 (关闭窗口继续)', color='white', fontsize=14)
        
#         # 初始范围
#         min_x = min(p[0] for p in all_points)
#         max_x = max(p[0] for p in all_points)
#         min_y = min(p[1] for p in all_points)
#         max_y = max(p[1] for p in all_points)
#         min_z = min(p[2] for p in all_points if np.isfinite(p[2]))
#         max_z = max(p[2] for p in all_points if np.isfinite(p[2]))
        
#         # 设置轴范围，稍微扩大一点以便查看
#         ax.set_xlim(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1)
#         ax.set_ylim(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1)
#         ax.set_zlim(min_z - (max_z - min_z) * 0.1, max_z + (max_z - min_z) * 0.1)
        
#         # 绘制每一轮的误差曲面
#         for i, error_map_data in enumerate(error_maps):
#             step = error_map_data['step']
#             error_map = error_map_data['error_map']
#             cx_values = error_map_data['cx_values']
#             cy_values = error_map_data['cy_values']
            
#             # 构建网格
#             X, Y = np.meshgrid(cx_values, cy_values)
            
#             # 处理误差值以便更好的可视化
#             error_map_viz = np.copy(error_map)
#             threshold = np.percentile(error_map_viz[np.isfinite(error_map_viz)], 95)
#             error_map_viz[error_map_viz > threshold] = threshold
            
#             # 使用渐变透明度表示不同迭代轮次
#             alpha = 0.1 + 0.6 * (i / len(error_maps))
            
#             # 使用颜色渐变表示不同步长
#             norm = plt.Normalize(0, len(error_maps)-1)
#             cmap = plt.cm.viridis
#             color = cmap(norm(i))
            
#             # 绘制半透明曲面
#             surf = ax.plot_surface(X, Y, error_map_viz, 
#                                   color=color,
#                                   edgecolor='none', 
#                                   alpha=alpha,
#                                   rstride=1, cstride=1, 
#                                   linewidth=0)
            
#             # 添加步长文本标注在左上角
#             if custom_font:
#                 ax.text(min_x, max_y, max_z - i*(max_z-min_z)/20, 
#                         f"步长 {step:.4f}", 
#                         color=cmap(norm(i)), fontsize=10, fontproperties=custom_font)
#             else:
#                 ax.text(min_x, max_y, max_z - i*(max_z-min_z)/20, 
#                         f"步长 {step:.4f}", 
#                         color=cmap(norm(i)), fontsize=10)
        
#         # 绘制优化轨迹 - 连接每轮的最优点
#         ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=4, label='优化轨迹')
        
#         # 标记每轮的最优点
#         for i, (x, y, z) in enumerate(best_points):
#             step = error_maps[i]['step']
#             # 点的大小随步长减小而减小
#             point_size = 200 - i * 150 / len(best_points)
#             ax.scatter([x], [y], [z], color='red', s=point_size, marker='o', alpha=0.7)
#             # 添加文本标注显示步长
#             if custom_font:
#                 ax.text(x, y, z, f"  步长:{step:.4f}", color='white', fontsize=8, fontproperties=custom_font)
#             else:
#                 ax.text(x, y, z, f"  步长:{step:.4f}", color='white', fontsize=8)
        
#         # 标记全局最优点
#         ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
#                    color='gold', s=200, marker='*', alpha=1.0)
        
#         # 添加全局最优点文本标注
#         if custom_font:
#             ax.text(optimal_cx, optimal_cy, best_points[-1][2], 
#                     f"  全局最优点\n  ({optimal_cx:.4f}, {optimal_cy:.4f})", 
#                     color='yellow', fontsize=12, fontproperties=custom_font)
#         else:
#             ax.text(optimal_cx, optimal_cy, best_points[-1][2], 
#                     f"  全局最优点\n  ({optimal_cx:.4f}, {optimal_cy:.4f})", 
#                     color='yellow', fontsize=12)
        
#         # 设置初始视角
#         ax.view_init(elev=30, azim=-60)
        
#         # 添加说明文本
#         if custom_font:
#             plt.figtext(0.5, 0.01, 
#                        "使用鼠标拖动旋转视图,    滚轮缩放,    右键平移", 
#                        ha="center", color='white', fontsize=12, fontproperties=custom_font)
#         else:
#             plt.figtext(0.5, 0.01, 
#                        "使用鼠标拖动旋转视图,    滚轮缩放,    右键平移", 
#                        ha="center", color='white', fontsize=12)
        
#         # 显示交互式图形
#         plt.tight_layout()
#         plt.show()
        
#     except Exception as e:
#         print(f"交互式可视化创建失败: {e}")
#         print("将继续处理...")
    
#     # 尝试恢复原始后端
#     try:
#         matplotlib.use(current_backend, force=True)
#     except:
#         # 如果失败，默认回到Agg后端
#         matplotlib.use('Agg', force=True)
    
#     # 创建3D点云可视化（将所有搜索的点以点云形式可视化）
#     try:
#         import matplotlib.pyplot as plt
        
#         # 收集所有有效的搜索点
#         all_valid_points = [(p[0], p[1], p[2]) for p in all_points if np.isfinite(p[2])]
#         if all_valid_points:
#             xs, ys, zs = zip(*all_valid_points)
#             steps = [p[3] for p in all_points if np.isfinite(p[2])]
            
#             fig = plt.figure(figsize=(14, 12))
#             ax = fig.add_subplot(111, projection='3d')
            
#             # 创建从小到大的步长颜色映射
#             norm = plt.Normalize(min(steps), max(steps))
#             cmap = plt.cm.plasma
#             colors = [cmap(norm(step)) for step in steps]
            
#             # 散点图绘制所有点
#             scatter = ax.scatter(xs, ys, zs, c=colors, marker='.', alpha=0.5, s=10)
            
#             # 添加颜色条
#             cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
#             if custom_font:
#                 cbar.set_label('步长', fontproperties=custom_font)
#             else:
#                 cbar.set_label('步长')
            
#             # 标记优化轨迹和最优点
#             ax.plot(best_xs, best_ys, best_zs, 'r-', linewidth=3, label='优化轨迹')
#             for i, (x, y, z) in enumerate(best_points):
#                 ax.scatter([x], [y], [z], color='red', s=100-i*5, marker='o')
            
#             # 标记全局最优点
#             ax.scatter([optimal_cx], [optimal_cy], [best_points[-1][2]], 
#                       color='gold', s=150, marker='*', label='全局最优点')
            
#             # 标题和标签
#             if custom_font:
#                 ax.set_title('所有搜索点的3D点云视图', fontproperties=custom_font)
#                 ax.set_xlabel('X坐标', fontproperties=custom_font)
#                 ax.set_ylabel('Y坐标', fontproperties=custom_font)
#                 ax.set_zlabel('误差值', fontproperties=custom_font)
#                 ax.legend(loc='upper right', prop=custom_font)
#             else:
#                 ax.set_title('所有搜索点的3D点云视图')
#                 ax.set_xlabel('X坐标')
#                 ax.set_ylabel('Y坐标')
#                 ax.set_zlabel('误差值')
#                 ax.legend(loc='upper right')
            
#             # 保存点云图
#             if save_dir:
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(save_dir, 'search_points_cloud_3d.png'), dpi=300)
            
#             plt.close()
#     except Exception as e:
#         print(f"点云可视化创建失败: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用动态步长网格搜索寻找最优旋转中心')
    parser.add_argument('--dir', type=str, required=True, help='包含旋转图像的目录')
    parser.add_argument('--roi', type=int, default=1280, help='感兴趣区域的大小')
    parser.add_argument('--cx_min', type=float, default=1531, help='X坐标最小值')
    parser.add_argument('--cx_max', type=float, default=1536, help='X坐标最大值')
    parser.add_argument('--cy_min', type=float, default=1014, help='Y坐标最小值')
    parser.add_argument('--cy_max', type=float, default=1017, help='Y坐标最大值')
    parser.add_argument('--max_step', type=float, default=0.5, help='最大搜索步长')
    parser.add_argument('--min_step', type=float, default=0.005, help='最小搜索步长')
    parser.add_argument('--zoom_factor', type=float, default=0.25, help='缩放因子，控制每轮搜索范围缩小程度')
    parser.add_argument('--n_jobs', type=int, default=None, help='并行处理的作业数')
    parser.add_argument('--cache', action='store_true', help='是否缓存计算结果')
    args = parser.parse_args()
    
    # 加载图像序列
    print("加载图像序列...")
    images, angles = load_images_from_directory(args.dir)
    
    if not images:
        print("没有找到有效图像，程序终止。")
        return
    
    print(f"已加载 {len(images)} 张图像，角度范围: {min(-np.array(angles))}° 到 {max(-np.array(angles))}°")
    
    # 设置搜索范围
    h, w = images[0].shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 如果未指定搜索范围，使用默认值（图像中心周围的范围）
    cx_range = (args.cx_min if args.cx_min is not None else center_x - 50, 
                args.cx_max if args.cx_max is not None else center_x + 50)
    cy_range = (args.cy_min if args.cy_min is not None else center_y - 50, 
                args.cy_max if args.cy_max is not None else center_y + 50)
    
    # 创建结果目录
    results_dir = os.path.join(args.dir, "dynamic_grid_search_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 检查是否有缓存
    cache_file = os.path.join(results_dir, "dynamic_grid_search_cache.pkl")
    if args.cache and os.path.exists(cache_file):
        print(f"发现缓存文件，正在加载: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            optimal_cx = cache_data['optimal_cx']
            optimal_cy = cache_data['optimal_cy']
            error_maps = cache_data['error_maps']
            search_history = cache_data['search_history']
            min_error = cache_data['min_error']
            ref_idx = cache_data['ref_idx']
            ref_angle = cache_data['ref_angle']
            
            print("成功加载缓存数据")
        except Exception as e:
            print(f"加载缓存失败: {e}，将重新计算")
            optimal_cx, optimal_cy, error_maps, search_history, min_error, ref_idx, ref_angle = None, None, None, None, None, None, None
    else:
        optimal_cx, optimal_cy, error_maps, search_history, min_error, ref_idx, ref_angle = None, None, None, None, None, None, None
    
    # 如果没有缓存数据，执行动态步长网格搜索
    if optimal_cx is None:
        # 使用动态步长网格搜索寻找最优旋转中心
        print("开始动态步长网格搜索...")
        optimal_cx, optimal_cy, error_maps, search_history, min_error, ref_idx, ref_angle = dynamic_grid_search(
            images, angles, cx_range, cy_range, 
            max_step=args.max_step, min_step=args.min_step, 
            roi_size=args.roi, n_jobs=args.n_jobs,
            zoom_factor=args.zoom_factor)
        
        # 缓存结果
        if args.cache:
            cache_data = {
                'optimal_cx': optimal_cx,
                'optimal_cy': optimal_cy,
                'error_maps': error_maps,
                'search_history': search_history,
                'min_error': min_error,
                'ref_idx': ref_idx,
                'ref_angle': ref_angle
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"搜索结果已缓存到: {cache_file}")
    
    # 输出结果
    print("\n========== 最终结果 ==========")
    print(f"最优旋转中心: ({optimal_cx:.4f}, {optimal_cy:.4f})")
    print(f"与图像中心({center_x}, {center_y})的偏移量: ({optimal_cx-center_x:.4f}, {optimal_cy-center_y:.4f})")
    print(f"最小误差值: {min_error:.4f}")
    print(f"参考图像角度: {-ref_angle:.1f}°（索引 {ref_idx}）")
    print(f"搜索轮数: {len(search_history)}")
    print("==============================\n")
    
    # 可视化搜索过程
    print("生成搜索过程可视化...")
    visualize_search_process(error_maps, search_history, optimal_cx, optimal_cy, results_dir)
    
    # 可视化旋转结果
    print("生成旋转结果可视化...")
    visualize_results(args.dir, images, angles, (optimal_cx, optimal_cy), ref_idx, ref_angle, args.roi)
    
    # 保存结果到文件
    report_file = os.path.join(results_dir, "dynamic_grid_search_report.txt")
    with open(report_file, 'w') as f:
        f.write("========== 搜索参数 ==========\n")
        f.write(f"初始搜索范围: X: {cx_range[0]} 到 {cx_range[1]}, Y: {cy_range[0]} 到 {cy_range[1]}\n")
        f.write(f"最大步长: {args.max_step}, 最小步长: {args.min_step}, 缩放因子: {args.zoom_factor}\n")
        f.write(f"评估使用的ROI大小: {args.roi}x{args.roi} 像素\n\n")
        
        f.write("========== 搜索过程 ==========\n")
        for i, data in enumerate(search_history):
            f.write(f"轮次 {i+1}: 步长 = {data['step']:.4f}\n")
            f.write(f"  搜索范围: X: {data['cx_range'][0]:.4f} 到 {data['cx_range'][1]:.4f}, "
                   f"Y: {data['cy_range'][0]:.4f} 到 {data['cy_range'][1]:.4f}\n")
            f.write(f"  最优点: ({data['best_cx']:.4f}, {data['best_cy']:.4f}), 误差: {data['min_error']:.4f}\n\n")
        
        f.write("========== 最终结果 ==========\n")
        f.write(f"参考图像角度: {-ref_angle:.1f}°（索引 {ref_idx}）\n")
        f.write(f"最优旋转中心: ({optimal_cx:.4f}, {optimal_cy:.4f})\n")
        f.write(f"图像中心: ({center_x}, {center_y})\n")
        f.write(f"偏移量: ({optimal_cx-center_x:.4f}, {optimal_cy-center_y:.4f})\n")
        f.write(f"最小误差值: {min_error:.4f}\n")
    
    print(f"报告已保存到: {report_file}")
    print("处理完成!")

if __name__ == "__main__":
    main()