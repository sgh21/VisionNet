from utils.touch_weight_map import *
from utils.VisionUtils import rotate_images
import cv2
import re

def load_all_images_from_folder(folder_path):
    """
    Load all images from a folder and return them as a list of PIL images.
    按照文件名中的数字排序，适用于_number.png格式。
    
    Args:
        folder_path (str): Path to the folder containing images.
        
    Returns:
        List[Image]: List of PIL Image objects.
    """
    image_list = []
    file_list = []
    
    # 获取所有图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            file_list.append(filename)
    
    # 按照文件名中的数字进行排序
    def extract_number(filename):
        # 提取文件名中的数字部分
        numbers = re.findall(r'_(\d+)\.', filename)
        if numbers:
            return int(numbers[0])
        return 0  # 如果没有找到数字，则返回0
    
    # 按数字排序
    file_list = sorted(file_list, key=extract_number)
    
    # 按排序后的顺序加载图片
    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        image = load_image(img_path)
        if image is not None:
            image_list.append(image)
    
    return image_list, file_list

def main():
    img_path = "/home/sgh/data/WorkSpace/BTBInsertionV2/documents/image_templates/image_background.png"
    img = load_image(img_path)
    img = crop_center(img, 560, 560)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('image_crop_background.png', img)
    # folder_path = '/home/sgh/data/WorkSpace/BTBInsertionV2/documents/calib_data/images/touch'
    
    # img_template_dir = os.path.join(folder_path, 'template')
    # img_template_path = os.path.join(img_template_dir, 'gel_image_raw1.png')
    # # rotate_images(img_template_dir)
    # output_dir = os.path.join(folder_path, 'diff_map')
    # os.makedirs(output_dir, exist_ok=True)
    # # Load all images from the specified folder
    # images, file_list = load_all_images_from_folder(folder_path)
    # # Load the template image
    # template_img = load_image(img_template_path)
    # template_img = crop_center(template_img, 448, 448)

    # assert template_img.shape == images[0].shape, "Template image and first image must have the same shape"
    
    # for i, image in enumerate(images):
    #     # 计算差异图
    #     diff_map = compute_difference_map(template_img, image, method='cv2')
    #     # # 应用双边滤波
    #     diff_map = cv2.bilateralFilter(diff_map, d=5, sigmaColor=75, sigmaSpace=75)
    #     # 应用NLM滤波
    #     filtered_diff = apply_nlm_filter(diff_map, h=4, template_window_size=5, search_window_size=15)

    #     # 使用直方图均衡化增强对比度
    #     _, equalization_diff = adaptive_range_enhancement(diff_map, percentile_low=0, percentile_high=100, visualize=False)
        
    #     # 保存差异图
    #     output_path = os.path.join(output_dir, file_list[i])
    #     cv2.imwrite(output_path, equalization_diff)
    #     print(f"Saved difference image to {output_path}")
    #     # 显示差异图
    #     cv2.imshow('Difference Map', equalization_diff)
    #     cv2.waitKey(1000)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()