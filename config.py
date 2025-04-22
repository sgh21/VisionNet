import os
import numpy as np
WORK_DIR  = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(WORK_DIR, 'weights')
# Yolo for object detection
PLUG_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'connector_plug.pt')
PIN_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'connector_pin.pt')
MAE_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'mae_vit_large_patch16_bs64.pth')
EXPERIMENT_DIR = os.path.join(WORK_DIR, 'experiment')
DATA_DIR = os.path.join(WORK_DIR, 'documents')
DATASET_DIR = os.path.join(WORK_DIR, 'dataset')

# 在相机上的初始位姿势[x,y,z,r,p,y] mm,deg
# ! 请注意：这里的单位与机器人控制时并不统一，只是为了便于调试
ROBOT_INIT_POSE = [-106,-548,250,-180,0,90]
# 相机内参 rx = focal_length / pixel_width
# 相机内参 ry = focal_length / pixel_height
# INTRINSIC = [-0.024175,-0.024225]
# INTRINSIC = [-0.0206,-0.0207]
INTRINSIC = [0.023860, 0.023987, -1.0]
CXCY = [-0.0102543, -0.0334525]  # 相机坐标系下的中心点坐标 560
M = np.float32([
                [1.3391, 0, 12.5791],
                [0, 1.3391, -21.2115]
            ])
# CXCY = [-0.0090657, -0.0335761]  # 相机坐标系下的中心点坐标 672
# 连接器型号枚举
SERIALS = ['4024P','4030P','4034P','4040P']
# HSV颜色空间的下限和上限
HSV_LOWER_BOUND = [0, 0, 0]
HSV_UPPER_BOUND = [180, 255, 55]

PARAMS = {
  'work_dir': WORK_DIR,
  'weights_dir': WEIGHTS_DIR,
  'yolo_model_path': PLUG_MODEL_PATH,
  'pin_model_path': PIN_MODEL_PATH,
  'robot_init_pose': ROBOT_INIT_POSE,
  'mae_model_path': MAE_MODEL_PATH,
  'experiment_dir': EXPERIMENT_DIR,
  'data_dir': DATA_DIR,
  'dataset_dir': DATASET_DIR,
  'intrinsic': INTRINSIC,
  'cxcy': CXCY,
  'm': M,
  'serials' : SERIALS,
  'hsv_lower_bound': HSV_LOWER_BOUND,
  'hsv_upper_bound': HSV_UPPER_BOUND
}

print("All parameters are loaded successfully!")
# print(PARAMS)