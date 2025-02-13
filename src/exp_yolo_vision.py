# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from config import PARAMS
import os
from tqdm import tqdm
from utils.YoloDection import YOLODetection
import logging


# ===== 配置 logging 与 tqdm 集成 =====
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
    
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

logger.addHandler(TqdmLoggingHandler())
# ========================================

# TODO: 1. 加载原始图像，使用YOLO模型进行预测
# TODO: 2. 对预测结果进行处理，包括裁剪，缩放等
# TODO: 3. 使用二级YOLO进行预测，获得针脚中心位置
# TODO: 4. 对不同针脚进行主成分分析，获得针脚的角度
# TODO: 5. 将视觉解决写入CSV文件


PIN_MODEL_PATH = PARAMS['pin_model_path']


YOLO_LOW_LEVEL = YOLODetection(model_path=PIN_MODEL_PATH)

def load_all_files(dir_path, endwith='.png'):
    files = os.listdir(dir_path)
    # 按_number.png排序
   
    # files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files = [file for file in files if file.endswith(endwith)]
    return files

def get_centers_by_mask(masks,xyxy=[0,0,0,0],scale_factor=1.0):
    centers = []
    
    for mask in masks:
        
        mask = mask.cpu().detach().numpy()
        mask = mask.astype(np.uint8)*255
        mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor,interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        
        # 计算图像矩
        M = cv2.moments(binary)
        

        if M["m00"] != 0:
            # 计算质心坐标
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            # 如果掩码为空，质心设为边界框中心或默认值
            bbox = xyxy
            cX = (bbox[0] + bbox[2]) / 2
            cY = (bbox[1] + bbox[3]) / 2
        
        # 调整质心坐标到原始图像坐标系
        bbox = xyxy
        adjusted_cX = int(cX / scale_factor**2 + bbox[0])
        adjusted_cY = int(cY / scale_factor**2 + bbox[1])
        
        centers.append([adjusted_cX, adjusted_cY])
   
    return centers

def get_centers(boxes, xyxy=[0,0,0,0], scale_factor=1.0):
    centers = []
    for box in boxes:
        x = (box[0]+box[2])/(2*scale_factor)+xyxy[0]
        y = (box[1]+box[3])/(2*scale_factor)+xyxy[1]
        centers.append([x,y])
    return centers

def mathematical_analysis(points, LDA=False):
    # 计算中心点
    center = np.mean(points, axis=0)
    points = np.array(points)
    
    # PCA分析
    pca = PCA(n_components=2)
    pca.fit(points)
    main_direction = pca.components_[0]
    
    if LDA:
        # 构造直线方程 v2x - v1y = c
        v1, v2 = main_direction
        c = v2*center[0] - v1*center[1]
        
        # 计算点到直线的距离作为分类依据
        distances = v2*points[:,0] - v1*points[:,1] - c
        labels = (distances > 0).astype(int)
        
        # LDA优化方向
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(points, labels)
        # 将LDA方向向量旋转90度得到分类边界方向
        lda_direction = lda.coef_[0]
        main_direction = np.array([-lda_direction[1], lda_direction[0]])  # 旋转90度
        
    # 计算角度
    angle = np.arctan2(main_direction[1], main_direction[0]) * 180 / np.pi
    
    return center, angle, main_direction,labels

def parse_image_info(img_name):
        img_name = img_name.split('.')[0]
        img_info = img_name.split('_')
        scale_factor = float(img_info[4]) # TODO:暂时不用
        xyxy = list(map(float,img_info[5][1:-1].split(',')))
       

        return xyxy
def main():
    serials = ['4024P','4030P','4034P','4040P']
    dataset_dir = os.path.join(PARAMS['dataset_dir'],\
                            'train_data_0115','val')
    scale_factor = 2.0
    img_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir,'labels')
    files = load_all_files(img_dir)
    csv_file = os.path.join(dataset_dir, f'vision_experiment.csv')
    csv_data = []
    for file in files:
        
        file_name = file.split('.')[0]
        file_path = os.path.join(img_dir, file)
        xyxy = parse_image_info(file)
        # 加载原始图像
        img = YOLO_LOW_LEVEL.load_image(file_path)
        img_scaled = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
        local_result = YOLO_LOW_LEVEL.predict(img_scaled, padding=1, conf=0.5, agnostic_nms=True, retina_masks=True)
        masks = local_result.masks
        boxes = local_result.xyxy
        centers = get_centers(boxes, xyxy=xyxy, scale_factor=2.0)
        # centers = get_centers_by_mask(masks, xyxy=xyxy, scale_factor=scale_factor)
        if len(centers) == 0:
            logger.warning(f"文件 {file} 没有检测到针脚。")
            continue
            
        center, angle, main_direction,labels = mathematical_analysis(centers,LDA=True)
        label = os.path.join(label_dir,f'{file_name}.txt')
        # 读取label文件
        if os.path.exists(label):
            with open(label, 'r') as f:
                label_data = f.readline().strip().split(',')
                x, y, rz = map(float, label_data)
        else:
            logger.warning(f"文件 {label} 不存在")
            x = y = rz = 0.0
        raw_data = {
            'vision_x': center[0]*PARAMS["intrinsic"][0], 
            'vision_y': center[1]*PARAMS["intrinsic"][0],
            'vision_angle': angle,
            'file': file,
            'label_x': x,
            'label_y': y,
            'label_rz': rz
        }
            
        # 调整坐标
        center[0] = (center[0] - xyxy[0]) * scale_factor
        center[1] = (center[1] - xyxy[1]) * scale_factor
    
        result_img = img_scaled.copy()
        for i, (box, label) in enumerate(zip(boxes, labels)):
            color = (0,0,255) if label == 0 else (255,0,0)  # 红色和蓝色
            cv2.rectangle(result_img, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        color, 2)
            # for box in boxes:
            #     cv2.rectangle(result_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)
            cv2.circle(result_img, (int(center[0]), int(center[1])), 2, (0,255,0), -1)
            cv2.line(result_img,
                    (int(center[0]), int(center[1])),
                    (int(center[0] + main_direction[0]/np.linalg.norm(main_direction)*50), int(center[1] + main_direction[1]/np.linalg.norm(main_direction)*50)),
                    (255,0,0), 2)
            
        # 如果需要显示图像，可以取消注释以下两行
        cv2.imshow("result", result_img)
        cv2.waitKey(20)

        csv_data.append(raw_data)
        
    # 保存数据到CSV文件
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    logger.info(f"所有数据已保存到 {csv_file}")

if __name__ == "__main__":
    main()
