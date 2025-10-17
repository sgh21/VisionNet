import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

checkpoint = "D:/WorkSpace/VisionNet/weights/sam_vit_l_0b3195.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry["vit_l"](checkpoint=checkpoint).to(device)
predictor = SamPredictor(sam)

image_path = "D:/WorkSpace/VisionNet/dataset/train_data_0218/train/rgb_images/image_4024P_0.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

points = []

def show_mask(ax, image_rgb, mask):
    ax.clear()
    ax.imshow(image_rgb)
    if mask is not None:
        # 创建一个纯色掩码（如绿色）
        color_mask = np.zeros_like(image_rgb)
        color_mask[..., 1] = 255  # 绿色通道
        # 只在掩码区域显示颜色
        mask_bool = mask.astype(bool)
        show_img = image_rgb.copy()
        show_img[mask_bool] = color_mask[mask_bool]
        ax.imshow(show_img, alpha=0.5)
    if points:
        pts = np.array(points)
        ax.scatter(pts[:, 0], pts[:, 1], c='red', s=40)
    ax.set_title("点击添加提示点，实时分割，按F退出")
    ax.axis('off')
    plt.draw()

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        points.append([x, y])
        input_points = np.array(points)
        input_labels = np.ones(len(points), dtype=np.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(image_rgb)
            masks, _, _ = predictor.predict(input_points, input_labels)
        mask = masks[0].astype(np.uint8) * 255
        show_mask(ax, image_rgb, mask)

def onkey(event):
    if event.key.lower() == 'f':
        plt.close()

fig, ax = plt.subplots()
show_mask(ax, image_rgb, None)
cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()