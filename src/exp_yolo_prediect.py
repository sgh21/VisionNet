from utils.YoloDection import YOLODetection
from config import PARAMS
import os
import cv2
import numpy as np
import pandas as pd

PLUG_MODEL_PATH = PARAMS['yolo_model_path']
YOLO = YOLODetection(model_path=PLUG_MODEL_PATH)
def load_all_files(dir_path ,endwith='.png',sort = False):
    files = os.listdir(dir_path)
    files = [file for file in files if file.endswith(endwith)]
    if sort:
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def get_column_data(csv_file,column_name):
    '''
    从csv文件中获取指定列的数据
    Args:
        csv_file: str
            CSV文件路径
        column_name: str
            列名
    Returns:
        column_data_numpy: numpy.ndarray
    '''
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Check if the column exists
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")
        
        # Extract the specified column
        column_data = data[column_name]
        
        # Ensure the data is numeric and drop NaN values
        column_data = pd.to_numeric(column_data, errors='coerce').dropna()

        column_data_numpy = column_data.to_numpy()
        return column_data_numpy
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{csv_file}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__=="__main__":
    serials = PARAMS['serials']
    dataset_dir = os.path.join(PARAMS['dataset_dir'],\
                            'exp_data_collector_0107')
    scale_factor = 1
    for serial in serials:
        data_dir = os.path.join(dataset_dir,f'image_{serial}')
        csv_file = os.path.join(dataset_dir,f'tcp_position_{serial}.csv')
        result_dir = os.path.join(dataset_dir,f'image_result_{serial}')
        labels_dir = os.path.join(dataset_dir,f'labels_{serial}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        files = load_all_files(data_dir,sort = True)

        # 读取csv文件
        tcp_x = get_column_data(csv_file,'tcp_x')*1000
        tcp_y = get_column_data(csv_file,'tcp_y')*1000
        tcp_rz = get_column_data(csv_file,'tcp_rz')
        for i,file in enumerate(files):
            file_path = os.path.join(data_dir,file)
            img = YOLO.load_image(file_path)
            YOLO.predict(img,padding=5,conf = 0.5 ,agnostic_nms = True,retina_masks = True)
           
            cut_img_result = YOLO.cut_image(box_size=(512,256))[0]
            # cut_img_result.scaled(scale_factor=scale_factor,interpolation=cv2.INTER_CUBIC)
            cut_img = cut_img_result.img
            cut_img_xyxy = cut_img_result.xyxy

            # 保存切割图片和缩放因子
            file_prefix = file.split('.')[0]
            file_name_list = file_prefix.split('_')
            xyxy_str = '['+','.join(map(str,cut_img_xyxy)) +']'
            file_name_list.insert(0,'roi')
            file_name_list.insert(4,str(scale_factor))
            file_name_list.insert(5,xyxy_str)
            file_name = '_'.join(file_name_list)
            # 保存切割后的图像
            cv2.imwrite(f'{result_dir}/{file_name}.png',cut_img)
            # 保存标签信息
            label_file = f'{labels_dir}/{file_name}.txt'
            with open(label_file,'w') as f:
                f.write(f'{tcp_x[i]},{tcp_y[i]},{tcp_rz[i]}')
            # 显示切割后的图像
            # YOLO.show_cut_image(timeout=400)
            # yolo.show_result(timeout=400,show_mask=True)