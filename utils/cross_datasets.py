import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class CrossMAEDataset(Dataset):
    """CrossMAE训练数据集"""
    def __init__(self, config, is_train=True, transform=None):
        self.is_train = is_train
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
        self.pairs = self._generate_pairs()
        self.transform = transform

    def _generate_pairs(self):
        """生成训练/验证图片对"""
        pairs = []
        for class_name, imgs in self.class_to_imgs.items():
            class_pairs = [(imgs[i], imgs[j]) 
                          for i in range(len(imgs))
                          for j in range(i+1, len(imgs))]
            
            if self.is_train:  # 训练集下采样
                num_samples = int(len(class_pairs) * self.sample_ratio)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)
            pairs.extend(class_pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        
        # 加载图片
        img1 = Image.open(os.path.join(self.img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.img_dir, img2_name)).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, img1_name.replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, img2_name.replace('.png', '.txt')))
        
        return img1, img2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)