import numpy as np
import cv2
import os
def create_black_image(width=640, height=480, channels=3):
    """
    创建指定尺寸的纯黑色图像
    
    Args:
        width (int): 图像宽度，默认为640
        height (int): 图像高度，默认为480
        channels (int): 图像通道数，1为灰度图，3为RGB彩色图，默认为3
        
    Returns:
        numpy.ndarray: 黑色图像
    """
    # 创建全零数组（黑色图像）
    black_image = np.zeros((height, width, channels), dtype=np.uint8)
    
    return black_image

# 使用示例
if __name__ == "__main__":
    # 创建640x480的RGB黑色图像
    black_img = create_black_image(640, 480)
    
    # 保存图像
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'black_image.png'), black_img)
    print(f"黑色图像已保存至: {os.path.join(output_dir, 'black_image.png')}")