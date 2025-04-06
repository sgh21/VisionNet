import os
import cv2
import numpy as np
import glob
from scipy.optimize import minimize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

def load_images_from_directory(directory):
    """加载目录中的所有旋转图像并按角度排序"""
    # 获取所有PNG图像
    image_files = glob.glob(os.path.join(directory, "*.png"))
    
    # 解析角度信息
    angle_info = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # 假设文件名格式为 rotation_+XX.X_YY.png
        try:
            angle_str = filename.split("_")[1]
            angle = float(angle_str)
            angle_info.append((angle, img_path))
        except (ValueError, IndexError):
            print(f"跳过无法解析角度的文件: {filename}")
    
    # 按角度排序
    angle_info.sort(key=lambda x: x[0])
    
    # 加载图像和对应角度
    images = []
    angles = []
    for angle, img_path in angle_info:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            angles.append(-angle)  # 角度取反
        else:
            print(f"无法加载图像: {img_path}")
    
    return images, angles

def find_reference_image_index(angles):
    """找出角度最接近0的图像索引"""
    absolute_angles = np.abs(angles)
    ref_idx = np.argmin(absolute_angles)
    return ref_idx

def rotate_image_around_center(image, angle, center=None, scale=1.0, visualize=False):
    """
    绕给定中心点旋转图像
    
    Args:
        image: 输入图像
        angle: 旋转角度（度）
        center: 旋转中心点坐标(x, y)，None则使用图像中心
        scale: 缩放因子
        visualize: 是否可视化旋转结果
        
    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    
    if center is None:
        # 默认以图像中心为旋转中心
        center = (w / 2, h / 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 应用旋转变换
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # 可视化旋转效果（如果要求的话）
    if visualize:
        # 在原始图像和旋转后图像上标记旋转中心
        marked_original = image.copy()
        marked_rotated = rotated.copy()
        
        # 旋转中心点
        center_int = (int(center[0]), int(center[1]))
        
        # 在原始图像上标记旋转中心
        cv2.circle(marked_original, center_int, 10, (0, 0, 255), -1)  # 红色圆点
        cv2.line(marked_original, (center_int[0]-20, center_int[1]), 
                (center_int[0]+20, center_int[1]), (0, 0, 255), 2)  # 水平线
        cv2.line(marked_original, (center_int[0], center_int[1]-20), 
                (center_int[0], center_int[1]+20), (0, 0, 255), 2)  # 垂直线
        
        # 在旋转后的图像上标记旋转中心
        cv2.circle(marked_rotated, center_int, 10, (0, 0, 255), -1)
        cv2.line(marked_rotated, (center_int[0]-20, center_int[1]), 
                (center_int[0]+20, center_int[1]), (0, 0, 255), 2)
        cv2.line(marked_rotated, (center_int[0], center_int[1]-20), 
                (center_int[0], center_int[1]+20), (0, 0, 255), 2)
        
        # 添加格网线帮助观察旋转效果
        for i in range(0, w, 100):
            cv2.line(marked_original, (i, 0), (i, h), (128, 128, 128), 1)
            cv2.line(marked_rotated, (i, 0), (i, h), (128, 128, 128), 1)
        for i in range(0, h, 100):
            cv2.line(marked_original, (0, i), (w, i), (128, 128, 128), 1)
            cv2.line(marked_rotated, (0, i), (w, i), (128, 128, 128), 1)
        
        # 添加角度文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(marked_original, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(marked_rotated, f"Rotated {angle}°", (10, 30), font, 1, (0, 255, 0), 2)
        
        # 水平拼接原始图像和旋转后的图像
        comparison = np.hstack((marked_original, marked_rotated))
        
        # 添加标题
        title_text = f"Original vs. Rotated {angle}° around ({center_int[0]}, {center_int[1]})"
        title_size = cv2.getTextSize(title_text, font, 1, 2)[0]
        title_position = ((comparison.shape[1] - title_size[0]) // 2, 60)
        cv2.putText(comparison, title_text, title_position, font, 1, (0, 255, 255), 2)
        
        # 显示对比图像
        window_name = "Rotation Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, comparison)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    return rotated

def evaluate_rotation_center(params, reference_img, ref_angle, angles, images, roi_size=560):
    """评估给定旋转中心的对齐误差，使用中心区域评估"""
    cx, cy = params
    
    # 获取图像尺寸
    h, w = reference_img.shape[:2]
    
    # 使用图像中心作为ROI中心
    center_x, center_y = w // 2, h // 2
    
    # 确保ROI不会超出图像边界
    roi_half = min(roi_size // 2, center_x, center_y, w - center_x, h - center_y)
    
    # 提取参考图像中心区域
    roi_x_start = w // 2 - roi_half
    roi_x_end = w // 2 + roi_half
    roi_y_start = h // 2 - roi_half
    roi_y_end = h // 2 + roi_half
    
    roi_ref = reference_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
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
            # 使用归一化互相关来测量相似度 (更稳健)
            # 转换为灰度图像以减少计算量
            if len(roi_ref.shape) > 2:  # 彩色图像
                gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
                gray_rotated = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
            else:
                gray_ref = roi_ref
                gray_rotated = roi_rotated
            
            # 标准化后的互相关，范围为[-1, 1]，1表示完全匹配
            result = cv2.matchTemplate(gray_rotated, gray_ref, cv2.TM_CCOEFF_NORMED)
            # 将相关系数转换为误差值 (值越小越好)
            similarity = np.max(result)
            error = 1.0 - similarity
            total_error += error
        else:
            # 如果ROI超出边界，给予高惩罚值
            total_error += 1e6
    
    return total_error
# def evaluate_rotation_center(params, reference_img, ref_angle, angles, images, roi_size=560):
#     """评估给定旋转中心的对齐误差，使用中心区域评估，以MSE作为度量"""
#     cx, cy = params
    
#     # 获取图像尺寸
#     h, w = reference_img.shape[:2]
    
#     # 使用图像中心作为ROI中心
#     center_x, center_y = w // 2, h // 2
    
#     # 确保ROI不会超出图像边界
#     roi_half = min(roi_size // 2, center_x, center_y, w - center_x, h - center_y)
    
#     # 提取参考图像中心区域
#     roi_x_start = w // 2 - roi_half
#     roi_x_end = w // 2 + roi_half
#     roi_y_start = h // 2 - roi_half
#     roi_y_end = h // 2 + roi_half
    
#     roi_ref = reference_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
#     # 将参考ROI转换为灰度图像（如果是彩色的）
#     if len(roi_ref.shape) > 2:  # 彩色图像
#         gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_ref = roi_ref
    
#     # 计算与参考图像的总误差
#     total_error = 0
    
#     for i, (angle, img) in enumerate(zip(angles, images)):
#         # 如果当前图像就是参考图像，则跳过
#         if angle == ref_angle:
#             continue
        
#         # 计算相对于参考角度的旋转角度
#         relative_angle = angle - ref_angle
        
#         # 计算需要应用的反向旋转角度
#         reverse_angle = -relative_angle
        
#         # 绕指定中心旋转图像
#         rotated = rotate_image_around_center(img, reverse_angle, center=(cx, cy), visualize=False)
        
#         # 提取相同位置的ROI
#         roi_rotated = rotated[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
#         # 计算均方误差
#         if roi_ref.shape == roi_rotated.shape:
#             # 将旋转后的ROI转换为灰度图像（如果是彩色的）
#             if len(roi_rotated.shape) > 2:  # 彩色图像
#                 gray_rotated = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray_rotated = roi_rotated
            
#             # 确保尺寸匹配
#             if gray_ref.shape != gray_rotated.shape:
#                 # 如果大小不匹配，调整大小
#                 gray_rotated = cv2.resize(gray_rotated, (gray_ref.shape[1], gray_ref.shape[0]))
            
#             # 计算MSE
#             # 使用浮点数类型计算以提高精度
#             diff = gray_ref.astype(np.float32) - gray_rotated.astype(np.float32)
#             mse = np.mean(diff * diff)
            
#             # 将MSE添加到总误差
#             total_error += mse
#         else:
#             # 如果ROI超出边界，给予高惩罚值
#             total_error += 1e6
    
#     return total_error

def find_optimal_rotation_center(images, angles, initial_center=None, roi_size=560):
    """找到最优的旋转中心"""
    h, w = images[0].shape[:2]
    
    if initial_center is None:
        # 默认以图像中心为初始猜测
        initial_center = [w // 2, h // 2]
    
    # 找出角度最接近0的图像作为参考
    ref_idx = find_reference_image_index(angles)
    reference_img = images[ref_idx]
    ref_angle = angles[ref_idx]
    
    print(f"使用角度为 {-ref_angle}° 的图像作为参考 (索引 {ref_idx})")
    
    # 定义误差函数
    def error_func(params):
        return evaluate_rotation_center(params, reference_img, ref_angle, angles, images, roi_size)
    
    # 使用优化算法寻找最优中心
    result = minimize(error_func, initial_center, method='Nelder-Mead', 
                     options={'maxiter': 200, 'disp': True, 'adaptive': True})
    
    # 如果优化不成功，尝试其他方法
    if not result.success:
        print("Nelder-Mead 优化不成功，尝试使用 Powell 方法...")
        result = minimize(error_func, initial_center, method='Powell',
                         options={'maxiter': 200, 'disp': True})
    
    optimal_cx, optimal_cy = result.x
    return optimal_cx, optimal_cy, result.fun, ref_idx, ref_angle

def visualize_results(directory, images, angles, optimal_center, ref_idx, ref_angle, roi_size=560):
    """可视化优化结果"""
    # 创建保存目录
    results_dir = os.path.join(directory, "rotation_center_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 使用参考图像
    reference_img = images[ref_idx]
    h, w = reference_img.shape[:2]
    
    # 图像中心和ROI边界
    center_x, center_y = w // 2, h // 2
    roi_half = roi_size // 2
    roi_x_start = w // 2 - roi_half
    roi_x_end = w // 2 + roi_half
    roi_y_start = h // 2 - roi_half
    roi_y_end = h // 2 + roi_half
    
    # 中心点坐标
    cx, cy = optimal_center
    
    # 在图像上标记旋转中心和ROI区域
    marked_ref = reference_img.copy()
    # 绘制ROI区域
    cv2.rectangle(marked_ref, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 2)
    # 标记旋转中心
    cv2.circle(marked_ref, (int(cx), int(cy)), 10, (0, 0, 255), -1)
    # 标记图像中心点
    cv2.circle(marked_ref, (center_x, center_y), 5, (0, 255, 0), -1)
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(marked_ref, f"Reference Image (Angle: {-ref_angle:.1f}°)", (10, 30), font, 1, (0, 255, 0), 2)
    
    # 保存标记了旋转中心的参考图像
    cv2.imwrite(os.path.join(results_dir, "reference_image_marked.jpg"), marked_ref)
    
    # 创建旋转修正后的图像
    print("生成修正后的图像...")
    for i, (angle, img) in enumerate(tqdm(zip(angles, images))):
        # 计算相对于参考角度的旋转角度
        relative_angle = angle - ref_angle
        
        # 绕最优中心旋转到基准位置
        corrected = rotate_image_around_center(img, -relative_angle, center=(cx, cy))
        
        # 在图像上标记中心点和ROI区域
        cv2.rectangle(corrected, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 2)
        cv2.circle(corrected, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.circle(corrected, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # 保存修正后的图像
        cv2.imwrite(os.path.join(results_dir, f"corrected_{i:02d}_{-angle:+.1f}.jpg"), corrected)
    
    # 创建差异图
    print("生成差异图...")
    for i, (angle, img) in enumerate(tqdm(zip(angles, images))):
        # 跳过参考图像自身
        if i == ref_idx:
            continue
        
        # 计算相对于参考角度的旋转角度
        relative_angle = angle - ref_angle
        
        # 绕最优中心旋转到基准位置
        corrected = rotate_image_around_center(img, -relative_angle, center=(cx, cy))
        
        # 计算ROI区域内的差异
        ref_roi = reference_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        corr_roi = corrected[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 创建完整差异图
        diff = cv2.absdiff(reference_img, corrected)
        
        # 突出显示ROI区域
        diff_highlighted = diff.copy()
        # 在ROI区域周围绘制边框
        cv2.rectangle(diff_highlighted, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 2)
        
        # 保存差异图
        cv2.imwrite(os.path.join(results_dir, f"diff_{i:02d}_{-angle:+.1f}.jpg"), diff_highlighted)
        
        # 单独保存ROI区域的差异
        roi_diff = cv2.absdiff(ref_roi, corr_roi)
        cv2.imwrite(os.path.join(results_dir, f"roi_diff_{i:02d}_{-angle:+.1f}.jpg"), roi_diff)
    
    # 创建旋转中心评估报告
    with open(os.path.join(results_dir, "rotation_center_report.txt"), 'w') as f:
        f.write(f"参考图像角度: {-ref_angle:.1f}°（索引 {ref_idx}）\n")
        f.write(f"最优旋转中心: ({cx:.2f}, {cy:.2f})\n")
        f.write(f"图像中心: ({center_x}, {center_y})\n")
        f.write(f"偏移量: ({cx-center_x:.2f}, {cy-center_y:.2f})\n")
        f.write(f"评估使用的ROI大小: {roi_size}x{roi_size} 像素\n")
        f.write(f"ROI区域: ({roi_x_start}, {roi_y_start}) 到 ({roi_x_end}, {roi_y_end})\n")

def main():
    # 获取目录路径
    import argparse
    parser = argparse.ArgumentParser(description='寻找图像序列的最优旋转中心')
    parser.add_argument('--dir', type=str, required=True, help='包含旋转图像的目录')
    parser.add_argument('--roi', type=int, default=672, help='感兴趣区域的大小')
    parser.add_argument('--initial_x', type=float, default=None, help='初始旋转中心X坐标')
    parser.add_argument('--initial_y', type=float, default=None, help='初始旋转中心Y坐标')
    args = parser.parse_args()
    
    # 加载图像序列
    print("加载图像序列...")
    images, angles = load_images_from_directory(args.dir)
    
    print(f"angles: {angles}")
    if not images:
        print("没有找到有效图像，程序终止。")
        return
    
    print(f"已加载 {len(images)} 张图像，角度范围: {min(-np.array(angles))}° 到 {max(-np.array(angles))}°")
    
    # 初始旋转中心猜测
    h, w = images[0].shape[:2]
    if args.initial_x is not None and args.initial_y is not None:
        initial_center = [args.initial_x, args.initial_y]
        print(f"使用指定的初始旋转中心: ({initial_center[0]}, {initial_center[1]})")
    else:
        initial_center = [w // 2, h // 2]
        print(f"使用图像中心作为初始旋转中心: ({initial_center[0]}, {initial_center[1]})")
    
    # 寻找最优旋转中心
    print("寻找最优旋转中心...")
    cx, cy, error, ref_idx, ref_angle = find_optimal_rotation_center(
        images, angles, initial_center, args.roi)
    
    print(f"最优旋转中心: ({cx:.2f}, {cy:.2f})")
    print(f"与图像中心({w//2}, {h//2})的偏移量: ({cx-w//2:.2f}, {cy-h//2:.2f})")
    print(f"最终误差值: {error:.2f}")
    print(f"参考图像角度: {-ref_angle:.1f}°（索引 {ref_idx}）")
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_results(args.dir, images, angles, (cx, cy), ref_idx, ref_angle, args.roi)
    
    print("处理完成!")

if __name__ == "__main__":
    main()