import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class CrossMAEDataset(Dataset):
    """CrossMAE训练数据集"""
    def __init__(self, config, is_train=True, transform=None, is_eval = False, use_fix_template = False):
        self.is_train = is_train
        self.is_eval = is_eval
        root = os.path.join(config.data_path, 'train' if is_train else 'val')
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.img_dir):
            class_name = img_file.split('_')[1]  # 根据文件名获取类别
            if class_name not in self.class_to_imgs:
                self.class_to_imgs[class_name] = []
            self.class_to_imgs[class_name].append(img_file)
            
        # 生成所有可能的图片对
        self.pairs = self._generate_pairs(use_fix_template = use_fix_template)
        self.transform = transform

    def _generate_pairs(self, use_fix_template = False):
        """生成训练/验证图片对"""
        pairs = []
        for class_name, imgs in self.class_to_imgs.items():

            if use_fix_template:
                # : 使用第0张图片作为固定模板
                image_template = 'image_' + class_name + '_0.png'
                # index0 = int(imgs[0].split('_')[-1].split('.')[0])
                # assert index0 == 0, 'The first image should be the template'
                class_pairs = [(image_template, imgs[i]) 
                          for i in range(1,len(imgs))]
            else:
                class_pairs = [(imgs[i], imgs[j]) 
                            for i in range(len(imgs))
                            for j in range(i+1, len(imgs))]
            
            if self.is_train or self.is_eval:  # 训练集下采样
                num_samples = int(len(class_pairs) * self.sample_ratio)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)
            else: # 测试集按固定比例采样
                num_samples = int(len(class_pairs) * 0.1)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)
            pairs.extend(class_pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)
    def _remove_prefix(self,filename):
        if filename.startswith('gel_'):
            return filename[4:]
        return filename
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        
        # 加载图片
        img1 = Image.open(os.path.join(self.img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.img_dir, img2_name)).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img1_name).replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img2_name).replace('.png', '.txt')))
        if self.is_eval:
            return img1, img2, label1, label2, img1_name, img2_name
        
        return img1, img2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)
    

class MultiCrossMAEDataset(Dataset):
    """
    MultiCrossMAE训练数据集
    format:
        root_dir/train/rgb_images/image_type_index.png
        root_dir/train/touch_images/gel_image_type_index.png
        root_dir/train/labels/image_type_index.txt

        root_dir/val/rgb_images/image_type_index.png
        root_dir/val/touch_images/gel_image_type_index.png
        root_dir/val/labels/image_type_index.txt
    return:
        rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2
    """
    def __init__(self, config, is_train=True, rgb_transform=None, touch_transform=None, is_eval=False, use_fix_template = False):
        self.is_train = is_train
        root = os.path.join(config.data_path, 'train' if is_train else 'val')
        self.rgb_img_dir = os.path.join(root, 'rgb_images')
        self.touch_img_dir = os.path.join(root, 'touch_images')
        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        self.is_eval = is_eval

        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.rgb_img_dir):
            class_name = img_file.split('_')[1]  # 根据文件名获取类别
            if class_name not in self.class_to_imgs:
                self.class_to_imgs[class_name] = []
            self.class_to_imgs[class_name].append(img_file)
            
        # 生成所有可能的图片对
        self.rgb_pairs,self.touch_pairs = self._generate_pairs(use_fix_template = use_fix_template)
        self.rgb_transform = rgb_transform
        self.touch_transform = touch_transform

    def _generate_pairs(self, use_fix_template = False):
        """生成训练/验证图片对"""
        rgb_pairs = []
        touch_pairs = []
        for class_name, imgs in self.class_to_imgs.items():
            if use_fix_template:
                # : 使用第0张图片作为固定模板
                img_template = 'image_' + class_name + '_0.png'
                # index0 = int(imgs[0].split('_')[-1].split('.')[0])
                # assert index0 == 0, 'The first image should be the template'
                class_pairs = [(img_template, imgs[i]) 
                          for i in range(1,len(imgs))]
            else:
                class_pairs = [(imgs[i], imgs[j]) 
                            for i in range(len(imgs))
                            for j in range(i+1, len(imgs))]
            
            if self.is_train or self.is_eval:  # 训练集下采样
                num_samples = int(len(class_pairs) * self.sample_ratio)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)
            else:
                # 验证集固定比例采样
                num_samples =  int(len(class_pairs)*0.1)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)

            rgb_pairs.extend(class_pairs)
        
        touch_pairs = [('gel_' + img1, 'gel_' + img2) for img1, img2 in rgb_pairs]

        return rgb_pairs,touch_pairs
    
    def __len__(self):
        return len(self.rgb_pairs)
    
    def __getitem__(self, idx):
        rgb_img1_name, rgb_img2_name = self.rgb_pairs[idx]
        touch_img1_name, touch_img2_name = self.touch_pairs[idx]

        # 加载图片
        rgb_img1 = Image.open(os.path.join(self.rgb_img_dir, rgb_img1_name)).convert('RGB')
        rgb_img2 = Image.open(os.path.join(self.rgb_img_dir, rgb_img2_name)).convert('RGB')
        touch_img1 = Image.open(os.path.join(self.touch_img_dir, touch_img1_name)).convert('RGB')
        touch_img2 = Image.open(os.path.join(self.touch_img_dir, touch_img2_name)).convert('RGB')

        if self.rgb_transform:
            rgb_img1 = self.rgb_transform(rgb_img1)
            rgb_img2 = self.rgb_transform(rgb_img2)
        if self.touch_transform:
            touch_img1 = self.touch_transform(touch_img1)
            touch_img2 = self.touch_transform(touch_img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, rgb_img1_name.replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, rgb_img2_name.replace('.png', '.txt')))

        if self.is_eval:
            return rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2, rgb_img1_name, rgb_img2_name
        
        return rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)