import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('copy_files.log')
        ]
    )
    return logging.getLogger(__name__)

def get_all_image_files(directory):
    """获取目录下所有图片文件的路径和文件名"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.txt']
    files_dict = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                # 使用文件名(不含扩展名)作为键
                file_basename = os.path.splitext(file)[0]
                files_dict[file_basename] = {
                    'path': file_path,
                    'name': file,
                    'rel_dir': os.path.relpath(root, directory)
                }
    
    return files_dict

def create_matching_dir_structure(source_dir, target_dir, output_base_dir, copy_nonmatching=False):
    """
    创建与源目录结构相同的输出目录，并复制目标目录中文件名匹配的图片
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径（包含要匹配的文件名）
        output_base_dir: 输出基础目录路径
        copy_nonmatching: 是否复制未匹配的文件
    """
    logger = logging.getLogger(__name__)
    
    # 确保路径是绝对路径
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    
    # 获取源目录和目标目录中所有图片文件
    logger.info(f"正在扫描源目录: {source_dir}")
    source_files = get_all_image_files(source_dir)
    logger.info(f"找到 {len(source_files)} 个源文件")
    
    logger.info(f"正在扫描目标目录: {target_dir}")
    target_files = get_all_image_files(target_dir)
    logger.info(f"找到 {len(target_files)} 个目标文件")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 计数器
    matched_count = 0
    not_found_count = 0
    
    # 记录未找到匹配的文件
    not_found_files = []
    
    # 复制匹配的文件
    for base_name, target_info in tqdm(target_files.items(), desc="正在处理文件"):
        if base_name in source_files:
            # 源文件信息
            source_info = source_files[base_name]
            
            # 创建目标文件的相对目录
            output_rel_dir = source_info['rel_dir']
            output_dir = os.path.join(output_base_dir, output_rel_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 复制文件
            output_file = os.path.join(output_dir, source_info['name'])
            shutil.copy2(source_info['path'], output_file)
            matched_count += 1
        else:
            not_found_count += 1
            not_found_files.append(target_info['path'])
            
            if copy_nonmatching:
                # 如果需要复制未匹配的文件，则从目标目录复制
                output_dir = os.path.join(output_base_dir, target_info['rel_dir'])
                os.makedirs(output_dir, exist_ok=True)
                
                output_file = os.path.join(output_dir, target_info['name'])
                shutil.copy2(target_info['path'], output_file)
    
    # 输出统计信息
    logger.info(f"处理完成!")
    logger.info(f"匹配并复制的文件数: {matched_count}")
    logger.info(f"未找到匹配的文件数: {not_found_count}")
    
    # 保存未找到匹配的文件列表
    if not_found_files:
        with open(os.path.join(output_base_dir, 'not_found_files.txt'), 'w') as f:
            for file_path in not_found_files:
                f.write(f"{file_path}\n")
        logger.info(f"未找到匹配的文件列表已保存到: {os.path.join(output_base_dir, 'not_found_files.txt')}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='复制与目标目录中文件名匹配的源目录图片，保持源目录结构')
    parser.add_argument('--source', '-s', default='H:\WorkSpace\BTBInsertion\experiments\\vision_data_0513\\bright\labels', help='源目录路径')
    parser.add_argument('--target', '-t', default='H:\WorkSpace\BTBInsertion\experiments\\vision_data_0513\\dark\\rgb_images', help='目标目录路径')
    parser.add_argument('--output', '-o', default='H:\WorkSpace\BTBInsertion\experiments\\vision_data_0513\\dark\labels', help='输出目录路径')
    parser.add_argument('--copy-nonmatching', action='store_true', help='复制未匹配的文件')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 执行主要函数
    create_matching_dir_structure(
        args.source, 
        args.target, 
        args.output,
        args.copy_nonmatching
    )

if __name__ == '__main__':
    main()