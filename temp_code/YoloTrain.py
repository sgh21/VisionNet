# Description: Train a YOLOv8 model on a custom dataset
from ultralytics import YOLO



data_dir = "D:/WorkSpace/VisionNet/dataset/yolo_1016"

if __name__ == "__main__":
    # Load a model
    # model = YOLO("yolov8s-seg.yaml")
    # model.load(weights='yolov8s-seg.pt', weights_only=True)
    model = YOLO("yolo11n-seg.pt").load(weights='yolo11n-seg.pt')  # build a new model from scratch
    # model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # 设置数据增强参数
    data_augmentation_params = {  
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.9, #0.4
        'degrees': 0.0, 
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'copy_paste': 0.0
    }
   
    # Use the model
    model.train(
        data=f"{data_dir}/config.yaml", 
        epochs=100,
        batch = 56,
        workers=12,  # 根据你的 CPU 核心数调整
        **data_augmentation_params)  # train the model
    