from ultralytics import YOLO
import cv2
import numpy as np
from  config import PLUG_MODEL_PATH,DATA_DIR,PIN_MODEL_PATH

class Result:
    def __init__(self,**kwargs):
        self.img = getattr(kwargs, 'img', None)
        self.xyxy = getattr(kwargs, 'xyxy', None)
        self.boxes = getattr(kwargs, 'boxes', None)
        self.masks = getattr(kwargs, 'masks', None)
        self.classes = getattr(kwargs, 'classes', None)
    
    def result_from_yolo(self, yolo_result,img=None):
        self.img = img
        self.xyxy = yolo_result.boxes.xyxy
        self.boxes = yolo_result.boxes
        self.masks = yolo_result.masks.data
        self.classes = yolo_result.boxes.cls
        return self
    
    def result_filter_with_area(self,area_threshold=1e4):
        filtered_xyxy = []
        for i in range(len(self.xyxy)):
            box = [int(j.item()) for j in self.xyxy[i].cpu().numpy().flatten()]
            _area = (box[2] - box[0]) * (box[3] - box[1])
            if area_threshold <= _area:
                filtered_xyxy.append(box)
        self.xyxy = filtered_xyxy
        return self
    
    def result_box_padding(self,padding = 20):
        padding_box = []
        for j in range(len(self.xyxy)):
            box = self.xyxy[j]
            if box[0] - padding >= 0:
                box[0] -= padding
            if box[1] - padding >= 0:
                box[1] -= padding
            if box[2] + padding <= self.img.shape[1]:
                box[2] += padding
            if box[3] + padding <= self.img.shape[0]:
                box[3] += padding
            padding_box.append(box)
        self.xyxy = padding_box
        return self
    
class CutImage:
    def __init__(self, img, xyxy):
        self.img = img
        self.xyxy = xyxy
        
    def scaled(self, scale_factor=2.0, interpolation=cv2.INTER_LINEAR):
        # 记录原始尺寸
        original_height, original_width = self.img.shape[:2]
        
        
        # 先将图像放大
        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)
        img_scaled = cv2.resize(self.img, (scaled_width, scaled_height), interpolation=interpolation)
        
        # 改正xyxy
        self.xyxy[2] = self.xyxy[0]+scaled_width
        self.xyxy[3] = self.xyxy[1]+scaled_height
        
        return img_scaled
        
class YOLODetection:
    def __init__(self, model_path=PLUG_MODEL_PATH):
        self.model = YOLO(model_path, task='segment')
        self.result = Result()
        
    def load_image(self, img_path):
        """
        加载图像，返回cv.Image对象

        Args:
            img_path (str): 图像路径

        Returns:
            cv.Image: 图像对象
        """
        src = cv2.imread(img_path)
        return src
    
    def predict(self, img,padding = 20,area_threshold=1e2,**kwargs):
        """
        使用yolo模型进行预测，返回Result对象

        Args:
            img (cv.Image): input image
            padding (int, optional): padding the image to some pixel. Defaults to 20.
            area_threshold (float, optional): fliter the image that smaller than threshold. Defaults to 1e2.

        Returns:
            Result: the struct of result
        """
        result = self.model.predict(img,**kwargs)[0]
        
        # 初始化1个Result对象
        self.result = self.result.result_from_yolo(result,img)
        # 过滤面积小于area_threshold的box
        self.result = self.result.result_filter_with_area(area_threshold)
        # 对box进行padding
        self.result = self.result.result_box_padding(padding)
        return self.result
    
    def cut_image(self, box_size=None):
        """
        将检测到的box进行裁剪，返回裁剪后的图像列表
        
        Args:
            box_size (tuple): 目标box大小 (width, height)，如果指定则将图像调整为该大小
            
        Returns:
            list[[cv.Image,xyxy]]: 图像列表，每个数据为一个裁剪后的图像和对应的坐标
        """
        cut_img_list = []
        img_height, img_width = self.result.img.shape[:2]
        
        for xyxy in self.result.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            
            if box_size is not None:
                target_w, target_h = box_size
                # 计算原始box的中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 计算新的xyxy坐标
                new_x1 = max(0, center_x - target_w // 2)
                new_y1 = max(0, center_y - target_h // 2)
                new_x2 = min(img_width, new_x1 + target_w)
                new_y2 = min(img_height, new_y1 + target_h)
                
                # 如果超出图像边界，进行调整
                if new_x2 - new_x1 < target_w:
                    if new_x1 == 0:
                        new_x2 = min(img_width, target_w)
                    else:
                        new_x1 = max(0, img_width - target_w)
                        new_x2 = img_width
                        
                if new_y2 - new_y1 < target_h:
                    if new_y1 == 0:
                        new_y2 = min(img_height, target_h)
                    else:
                        new_y1 = max(0, img_height - target_h)
                        new_y2 = img_height
                
                # 裁剪图像
                cutted_img = self.result.img[new_y1:new_y2, new_x1:new_x2]
                
                # 更新xyxy坐标
                xyxy = [new_x1, new_y1, new_x2, new_y2]
            else:
                cutted_img = self.result.img[y1:y2, x1:x2]
                
            cut_img = CutImage(cutted_img, xyxy)
            cut_img_list.append(cut_img)
        
        return cut_img_list
            
    def draw_mask(self,alpha = 0.3,detach = False):
        
        img = self.result.img.copy()

        mask_all = np.zeros_like(self.result.img)
        for i in range(len(self.result.xyxy)):
            mask = self.result.masks[i].cpu().detach().numpy()
            colored_mask = np.zeros_like(self.result.img)
            colored_mask[mask > 0] = [0, 250, 154]
            mask_all += colored_mask
        if detach:
            img = cv2.addWeighted(mask_all, alpha, img, 1-alpha, 0, img)
        else:
            self.result.img = cv2.addWeighted(mask_all, alpha, img, 1-alpha, 0, img)

        return img
    
    def show_result(self,scaled = 1,timeout = 0,show_mask = False):
        img = self.result.img.copy()
        mask_all = np.zeros_like(img)
        for i in range(len(self.result.xyxy)):
            box = self.result.xyxy[i]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            if show_mask:
                # 将掩码转换为三通道彩色图像
                mask = self.result.masks[i].cpu().detach().numpy()
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0] = [0, 250, 154]  # 使用绿色标记掩码区域

                # 添加掩码到原图
                mask_all += colored_mask
        if show_mask:
            cv2.addWeighted(mask_all, 0.3, img, 0.7, 0, img)
        if scaled != 1:
            img = cv2.resize(img, (int(img.shape[1] * scaled), int(img.shape[0] * scaled)))
        cv2.imshow('result', img)
        cv2.waitKey(timeout)
       
    def show_cut_image(self,timeout = 0):
        cut_img_list = self.cut_image()
        
        # 获取图像列表
        images = [cut_img.img for cut_img in cut_img_list]
        
        for img in images:
            cv2.imshow('result',img)
            cv2.waitKey(timeout)
            
if __name__ == '__main__':
    yolo_detection = YOLODetection(model_path=PLUG_MODEL_PATH)
    img_path = f'{DATA_DIR}/test_image[0,0,0].png'
    img = yolo_detection.load_image(img_path)
    result = yolo_detection.predict(img,padding=0)
    # print(result.xyxy)
    yolo_detection.show_result(scaled=0.5)
    # yolo_detection.show_cut_image()