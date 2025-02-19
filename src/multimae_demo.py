import os 
import sys
sys.path.append('..')
import numpy as np
import torch
from einops import rearrange
from functools import partial
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from models.multimae.input_adapters import PatchedInputAdapter
from models.multimae.output_adapters import SpatialOutputAdapter
from models.multimae.multimae import pretrain_multimae_base

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title='',normalize=True):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    if normalize:
        image = image * imagenet_std + imagenet_mean
    plt.imshow(torch.clip(image * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def show_images_side_by_side(img1, img2, normalize1=True, normalize2=True, title1='Image 1', title2='Image 2', figsize=(10,5)):
    # 创建1行2列的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    if normalize1:
        img1 = img1 * imagenet_std + imagenet_mean
    if normalize2:
        img2 = img2 * imagenet_std + imagenet_mean

    img1 = torch.clip(img1 * 255, 0, 255).int()
    img2 = torch.clip(img2 * 255, 0, 255).int()
    
    # 显示第一张图
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax1.axis('off')
    
    # 显示第二张图
    ax2.imshow(img2)
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    return
def denormalize(image):
    return image * imagenet_std + imagenet_mean


def plot_four_images(img1, img2, img3, img4, 
                    title1='Image 1', title2='Image 2', 
                    title3='Image 3', title4='Image 4', 
                    figsize=(10,10)):
    
    # 创建2x2布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    # img1 = torch.clip(img1 * 255, 0, 255).int()
    # img2 = torch.clip(img2 * 255, 0, 255).int()
    # img3 = torch.clip(img3 * 255, 0, 255).int()
    # img4 = torch.clip(img4 * 255, 0, 255).int()
    # 显示第一张图
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax1.axis('off')
    
    # 显示第二张图
    ax2.imshow(img2)
    ax2.set_title(title2)
    ax2.axis('off')
    
    # 显示第三张图
    ax3.imshow(img3)
    ax3.set_title(title3)
    ax3.axis('off')
    
    # 显示第四张图
    ax4.imshow(img4)
    ax4.set_title(title4)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_predictions(input_dict, preds, masks, image_size=224):
    pred_rgb = preds['rgb'][0].permute(1,2,0)
    pred_touch = preds['touch'][0].permute(1,2,0)
    input_rgb = input_dict['rgb'][0].cpu().permute(1,2,0)
    input_touch = input_dict['touch'][0].cpu().permute(1,2,0)

    pred_rgb = torch.clip(denormalize(pred_rgb) * 255, 0, 255).int()
    pred_touch = torch.clip(pred_touch * 255, 0, 255).int()
    
    input_rgb = torch.clip(denormalize(input_rgb) * 255, 0, 255).int()
    input_touch = torch.clip(input_touch * 255, 0, 255).int()

    plot_four_images(input_rgb, pred_rgb, input_touch, pred_touch)


def run_one_image(input_dict, model, alphas, num_encoder_tokens):
    # 将alphas转换为字典格式
    if isinstance(alphas, (int, float)):
        alphas = [alphas] * 2
    preds, masks =model.forward(
        input_dict,
        mask_inputs=True,
        num_encoded_tokens=num_encoder_tokens,
        alphas=alphas,
    )
    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    res = plot_predictions(input_dict, preds, masks)

DOMAIN_CONF = {
    'rgb': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
    },
    'touch': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
    },
}
DOMAINS = ['rgb', 'touch']

input_adapters = {
    domain: dinfo['input_adapter'](
        patch_size_full=16,
    )
    for domain, dinfo in DOMAIN_CONF.items()
}
output_adapters = {
    domain: dinfo['output_adapter'](
        patch_size_full=16,
        dim_tokens=256,
        use_task_queries=True,
        depth=2,
        context_tasks=DOMAINS,
        task=domain
    )
    for domain, dinfo in DOMAIN_CONF.items()
}

multimae = pretrain_multimae_base(
    input_adapters=input_adapters,
    output_adapters=output_adapters,
)

# 指定参数路径
checkpoint_path = './weights/pretrained_multimae_base_multivit_finetune.pth'

# 加载参数
checkpoint_model = torch.load(checkpoint_path, map_location='cpu')

# 加载模型参数
msg = multimae.load_state_dict(checkpoint_model['model'], strict=False)
# print(msg)

# 将模型放到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimae.to(device)

root_img_dir = './dataset/multimae_eval'
rgb_dir = os.path.join(root_img_dir, 'rgb')
touch_dir = os.path.join(root_img_dir, 'touch')
img_num = 5
plug_serial = '4030P'
rgb_path = os.path.join(rgb_dir, f'image_{plug_serial}_{img_num}.png')
touch_path = os.path.join(touch_dir, f'gel_image_{plug_serial}_{img_num}.png')
rgb_img = Image.open(rgb_path).convert('RGB')
touch_img = Image.open(touch_path).convert('RGB')

rgb_img = rgb_img.resize((224, 224))
touch_img = touch_img.resize((224, 224))

rgb_img = np.array(rgb_img)/255.
assert rgb_img.shape == (224, 224, 3) 

rgb_img = rgb_img - imagenet_mean
rgb_img = rgb_img / imagenet_std

# rgb_img = TF.normalize(TF.to_tensor(rgb_img),mean=imagenet_mean, std=imagenet_std).unsqueeze(0)
rgb_img = torch.Tensor(rgb_img).permute(2,0,1).unsqueeze(0)
touch_img = TF.to_tensor(touch_img).unsqueeze(0)

input_dict = {
    'rgb': rgb_img,
    'touch': touch_img,
}
input_dict = {k: v.to(device) for k, v in input_dict.items()}
run_one_image(input_dict, multimae, 1, 98)