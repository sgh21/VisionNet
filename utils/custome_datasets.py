import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from utils.VisionUtils import add_radial_noise
import torchvision.transforms as transforms
from utils.TransUtils import TerraceMapGenerator
from config import EXPANSION_SIZE

expansion_size = EXPANSION_SIZE

class LocalMAEDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None, is_eval = False, use_fix_template = False):
        self.is_train = is_train
        self.is_eval = is_eval
        root = os.path.join(config.data_path, 'train' if is_train else 'val')
        self.rgb_img_dir = os.path.join(root, 'rgb_images')
        self.touch_img_dir = os.path.join(root, 'touch_masks')
        self.touch_mask_gray_dir = os.path.join(root, 'touch_masks_gray')
        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        self.high_res_size = config.high_res_size

        # TODO: 根据mask生成权重图，将触觉对齐到RGB图像
        self.terrace_map_generator = TerraceMapGenerator(
            intensity_scaling = config.intensity_scaling,
            edge_enhancement = config.edge_enhancement,
            sample_size= config.sample_size,
            expansion_size = expansion_size,
        )

        self.touch_transform = transforms.Compose([
            transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.rgb_img_dir):
            class_name = img_file.split('_')[-2]  # 根据文件名获取类别
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
                # *: use gel_image as template when train touch model
                img_template = 'image_'+class_name+'_0.png'
                index0 = int(img_template.split('_')[-1].split('.')[0])
                assert index0 == 0, 'The first image should be the template'
                class_pairs = [(imgs[i], img_template) 
                          for i in range(0,len(imgs))]
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
            # 随机打散
            random.shuffle(pairs)
            # 采样触觉图像
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
        img1 = Image.open(os.path.join(self.rgb_img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.rgb_img_dir, img2_name)).convert('RGB')

        touch_mask1 = Image.open(os.path.join(self.touch_img_dir, 'gel_' + img1_name)).convert('L')
        touch_mask2 = Image.open(os.path.join(self.touch_img_dir, 'gel_' + img2_name)).convert('L')
        touch_mask1_gray = Image.open(os.path.join(self.touch_mask_gray_dir, 'gel_' + img1_name)).convert('L')
        touch_mask2_gray = Image.open(os.path.join(self.touch_mask_gray_dir, 'gel_' + img2_name)).convert('L')

        # 触觉图像转换
        serial = img1_name.split('_')[-2]
        # TODO: 添加contour误差计算方法
        terrace_map1, sample_contour1 = self.terrace_map_generator(touch_mask1, serial = serial)
        terrace_map2, sample_contour2 = self.terrace_map_generator(touch_mask2, serial = serial)
        touch_img_mask1 = self.touch_transform(terrace_map1)
        touch_img_mask2 = self.touch_transform(terrace_map2) # (1, H, W)

        touch_mask1_gray = self.touch_transform(touch_mask1_gray)
        touch_mask2_gray = self.touch_transform(touch_mask2_gray)
        threshold = 0.01
        touch_img_mask1 = self.sample_image_with_terrace_map(touch_mask1_gray, touch_img_mask1, threshold)
        touch_img_mask2 = self.sample_image_with_terrace_map(touch_mask2_gray, touch_img_mask2, threshold)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img1_name).replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img2_name).replace('.png', '.txt')))
        if self.is_eval:
            return img1, img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2, img1_name, img2_name
        
        return img1, img2,touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)
    
    def sample_image_with_terrace_map(self, img, terrace_map, threshold=0.1):
        """
        使用灰度地形图(terrace_map)对图像进行采样，支持Tensor和NumPy格式
        
        参数:
            img: 图像数据，可以是Tensor[C,H,W]或NumPy数组[H,W,C]
            terrace_map: 灰度地形图，可以是Tensor[1,H,W]或NumPy数组[H,W]
            threshold: 灰度阈值，默认为0.1
            
        返回:
            采样后的图像，与输入格式一致
        """
        # 检查数据类型
        is_tensor = isinstance(img, torch.Tensor)
        
        if is_tensor:
            # 处理Tensor格式数据
            
            # 确保terrace_map也是Tensor格式
            if not isinstance(terrace_map, torch.Tensor):
                terrace_map = torch.from_numpy(terrace_map).float()
            
            # 处理terrace_map的形状，确保它是[H,W]或可以被压缩为[H,W]
            if terrace_map.dim() > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if terrace_map.dim() > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建二值掩码
            mask = (terrace_map > threshold).float()
            
            # 适配不同通道维度的图像
            if img.dim() == 2:  # 单通道图像 [H,W]
                # 直接乘以掩码
                sampled_img = img * mask
            elif img.dim() == 3:  # 多通道图像 [C,H,W]
                # 扩展掩码维度以匹配图像通道
                mask = mask.unsqueeze(0)  # [1,H,W]
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[1:] != img.shape[1:]:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0),  # [1,1,H,W]
                        size=img.shape[1:],
                        mode='nearest'
                    ).squeeze(0)  # [1,H,W]
                    
                # 扩展掩码的通道维度以匹配输入通道数
                mask = mask.expand_as(img)  # [C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
            else:  # 处理批次数据 [B,C,H,W]
                # 扩展掩码维度 [H,W] -> [1,1,H,W]
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[2:] != img.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=img.shape[2:],
                        mode='nearest'
                    )
                    
                # 扩展掩码的批次和通道维度以匹配输入
                mask = mask.expand(img.shape[0], img.shape[1], -1, -1)  # [B,C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
                
        else:
            # 处理NumPy格式数据
            
            # 确保terrace_map是NumPy数组
            if isinstance(terrace_map, torch.Tensor):
                terrace_map = terrace_map.detach().cpu().numpy()
                
            # 确保terrace_map是二维数组
            if len(terrace_map.shape) > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if len(terrace_map.shape) > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建掩码
            mask = (terrace_map > threshold).astype(np.float32)
            
            # 适配不同形状的图像
            if len(img.shape) == 2:  # 单通道图像
                sampled_img = img * mask
            else:  # 多通道图像
                # 确定通道维度位置
                if img.shape[-1] in [1, 3, 4]:  # [H,W,C] 格式
                    # 扩展掩码维度以匹配图像通道
                    mask_expanded = np.expand_dims(mask, axis=-1)
                    mask_expanded = np.repeat(mask_expanded, img.shape[-1], axis=-1)
                else:  # [C,H,W] 格式
                    mask_expanded = np.expand_dims(mask, axis=0)
                    mask_expanded = np.repeat(mask_expanded, img.shape[0], axis=0)
                    
                # 应用掩码
                sampled_img = img * mask_expanded
                
        return sampled_img

class EvalMAEDataset(Dataset):
    def __init__(self, config, use_fix_template = False):
        # *: 验证集根路径
        root = os.path.join(config.data_path, config.eval_data_folder)
        self.rgb_img_dir = os.path.join(root, 'rgb_images')
        self.mask_method = config.method
        if self.mask_method == 'touch_mask':
            self.mask_img_dir = os.path.join(root, 'touch_masks')
        elif self.mask_method == 'rgb_mask':
            self.mask_img_dir = os.path.join(root, 'rgb_masks')
        elif self.mask_method == 'touch_mask_gray':
            self.mask_img_dirs = [os.path.join(root, 'touch_masks'), os.path.join(root, 'touch_masks_gray')]


        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        self.high_res_size = config.high_res_size

        # TODO: 根据mask生成权重图，将触觉对齐到RGB图像
        self.terrace_map_generator = TerraceMapGenerator(
            intensity_scaling = config.intensity_scaling,
            edge_enhancement = config.edge_enhancement,
            sample_size= config.sample_size,
            expansion_size = EXPANSION_SIZE,
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.touch_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 构建高分辨率转换
        self.high_res_transform = transforms.Compose([
            transforms.Resize((self.high_res_size, self.high_res_size),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.rgb_img_dir):
            class_name = img_file.split('_')[-2]  # 根据文件名获取类别
            if class_name not in self.class_to_imgs:
                self.class_to_imgs[class_name] = []
            self.class_to_imgs[class_name].append(img_file)
            
        # 生成所有可能的图片对
        self.pairs = self._generate_pairs(use_fix_template = use_fix_template)

    def _generate_pairs(self, use_fix_template = False):
        """生成训练/验证图片对"""
        pairs = []
        for class_name, imgs in self.class_to_imgs.items():

            if use_fix_template:
                # : 使用第0张图片作为固定模板
                # *: use gel_image as template when train touch model
                img_template = 'image_'+class_name+'_0.png'
                index0 = int(img_template.split('_')[-1].split('.')[0])
                assert index0 == 0, 'The first image should be the template'
                class_pairs = [(imgs[i], img_template) 
                          for i in range(0,len(imgs))]
            else:
                class_pairs = [(imgs[i], imgs[j]) 
                            for i in range(len(imgs))
                            for j in range(i+1, len(imgs))]
            
            num_samples = int(len(class_pairs) * self.sample_ratio)
            if num_samples > 0:
                class_pairs = random.sample(class_pairs, num_samples)
        
            pairs.extend(class_pairs)
            # 随机打散
            random.shuffle(pairs)
            # 采样触觉图像
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
        img1 = Image.open(os.path.join(self.rgb_img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.rgb_img_dir, img2_name)).convert('RGB')

        if self.mask_method == 'touch_mask':
            mask1 = Image.open(os.path.join(self.mask_img_dir, 'gel_' + img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dir, 'gel_' + img2_name)).convert('L')
        elif self.mask_method == 'rgb_mask':
            mask1 = Image.open(os.path.join(self.mask_img_dir, img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dir, img2_name)).convert('L')
            mask1 = mask1.resize((560, 560),interpolation=Image.NEAREST)
            mask2 = mask2.resize((560, 560),interpolation=Image.NEAREST)
        elif self.mask_method == 'touch_mask_gray':
            mask1 = Image.open(os.path.join(self.mask_img_dirs[0], 'gel_' + img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dirs[0], 'gel_' + img2_name)).convert('L')
            mask1_gray = Image.open(os.path.join(self.mask_img_dirs[1], 'gel_' + img1_name)).convert('L')
            mask2_gray = Image.open(os.path.join(self.mask_img_dirs[1], 'gel_' + img2_name)).convert('L')

        # 触觉图像转换
        serial = img1_name.split('_')[-2]
        
        terrace_map1, sample_contour1 = self.terrace_map_generator(mask1, serial = serial)
        terrace_map2, sample_contour2 = self.terrace_map_generator(mask2, serial = serial)
        touch_img_mask1 = self.touch_transform(terrace_map1)
        touch_img_mask2 = self.touch_transform(terrace_map2)
        
        if self.mask_method == 'touch_mask_gray':
            mask1_gray = self.touch_transform(mask1_gray)
            mask2_gray = self.touch_transform(mask2_gray)
            threshold = 0.05
            touch_img_mask1 = self.sample_image_with_terrace_map(mask1_gray, touch_img_mask1, threshold)
            touch_img_mask2 = self.sample_image_with_terrace_map(mask2_gray, touch_img_mask2, threshold)
        
        # 保存高分辨率版本
        high_res_img1 = self.high_res_transform(img1)
        high_res_img2 = self.high_res_transform(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img1_name).replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img2_name).replace('.png', '.txt')))
        
        return img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)
    
    def sample_image_with_terrace_map(self, img, terrace_map, threshold=0.1):
        """
        使用灰度地形图(terrace_map)对图像进行采样，支持Tensor和NumPy格式
        
        参数:
            img: 图像数据，可以是Tensor[C,H,W]或NumPy数组[H,W,C]
            terrace_map: 灰度地形图，可以是Tensor[1,H,W]或NumPy数组[H,W]
            threshold: 灰度阈值，默认为0.1
            
        返回:
            采样后的图像，与输入格式一致
        """
        # 检查数据类型
        is_tensor = isinstance(img, torch.Tensor)
        
        if is_tensor:
            # 处理Tensor格式数据
            
            # 确保terrace_map也是Tensor格式
            if not isinstance(terrace_map, torch.Tensor):
                terrace_map = torch.from_numpy(terrace_map).float()
            
            # 处理terrace_map的形状，确保它是[H,W]或可以被压缩为[H,W]
            if terrace_map.dim() > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if terrace_map.dim() > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建二值掩码
            mask = (terrace_map > threshold).float()
            
            # 适配不同通道维度的图像
            if img.dim() == 2:  # 单通道图像 [H,W]
                # 直接乘以掩码
                sampled_img = img * mask
            elif img.dim() == 3:  # 多通道图像 [C,H,W]
                # 扩展掩码维度以匹配图像通道
                mask = mask.unsqueeze(0)  # [1,H,W]
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[1:] != img.shape[1:]:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0),  # [1,1,H,W]
                        size=img.shape[1:],
                        mode='nearest'
                    ).squeeze(0)  # [1,H,W]
                    
                # 扩展掩码的通道维度以匹配输入通道数
                mask = mask.expand_as(img)  # [C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
            else:  # 处理批次数据 [B,C,H,W]
                # 扩展掩码维度 [H,W] -> [1,1,H,W]
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[2:] != img.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=img.shape[2:],
                        mode='nearest'
                    )
                    
                # 扩展掩码的批次和通道维度以匹配输入
                mask = mask.expand(img.shape[0], img.shape[1], -1, -1)  # [B,C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
                
        else:
            # 处理NumPy格式数据
            
            # 确保terrace_map是NumPy数组
            if isinstance(terrace_map, torch.Tensor):
                terrace_map = terrace_map.detach().cpu().numpy()
                
            # 确保terrace_map是二维数组
            if len(terrace_map.shape) > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if len(terrace_map.shape) > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建掩码
            mask = (terrace_map > threshold).astype(np.float32)
            
            # 适配不同形状的图像
            if len(img.shape) == 2:  # 单通道图像
                sampled_img = img * mask
            else:  # 多通道图像
                # 确定通道维度位置
                if img.shape[-1] in [1, 3, 4]:  # [H,W,C] 格式
                    # 扩展掩码维度以匹配图像通道
                    mask_expanded = np.expand_dims(mask, axis=-1)
                    mask_expanded = np.repeat(mask_expanded, img.shape[-1], axis=-1)
                else:  # [C,H,W] 格式
                    mask_expanded = np.expand_dims(mask, axis=0)
                    mask_expanded = np.repeat(mask_expanded, img.shape[0], axis=0)
                    
                # 应用掩码
                sampled_img = img * mask_expanded
                
        return sampled_img
class TransMAEDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None, is_eval = False, use_fix_template = False):
        self.is_train = is_train
        self.is_eval = is_eval
        root = os.path.join(config.data_path, 'train' if is_train else 'val')
        self.rgb_img_dir = os.path.join(root, 'rgb_images')
        self.mask_method = config.method
        if self.mask_method == 'touch_mask':
            self.mask_img_dir = os.path.join(root, 'touch_masks')
        elif self.mask_method == 'rgb_mask':
            self.mask_img_dir = os.path.join(root, 'rgb_masks')
        elif self.mask_method == 'touch_mask_gray':
            self.mask_img_dirs = [os.path.join(root, 'touch_masks'), os.path.join(root, 'touch_masks_gray')]


        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        self.high_res_size = config.high_res_size

        # TODO: 根据mask生成权重图，将触觉对齐到RGB图像
        self.terrace_map_generator = TerraceMapGenerator(
            intensity_scaling = config.intensity_scaling,
            edge_enhancement = config.edge_enhancement,
            sample_size= config.sample_size,
            expansion_size = EXPANSION_SIZE,
        )
        
        self.touch_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 构建高分辨率转换
        self.high_res_transform = transforms.Compose([
            transforms.Resize((self.high_res_size, self.high_res_size),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.rgb_img_dir):
            class_name = img_file.split('_')[-2]  # 根据文件名获取类别
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
                # *: use gel_image as template when train touch model
                img_template = 'image_'+class_name+'_0.png'
                index0 = int(img_template.split('_')[-1].split('.')[0])
                assert index0 == 0, 'The first image should be the template'
                class_pairs = [(imgs[i], img_template) 
                          for i in range(0,len(imgs))]
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
            # 随机打散
            random.shuffle(pairs)
            # 采样触觉图像
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
        img1 = Image.open(os.path.join(self.rgb_img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.rgb_img_dir, img2_name)).convert('RGB')

        if self.mask_method == 'touch_mask':
            mask1 = Image.open(os.path.join(self.mask_img_dir, 'gel_' + img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dir, 'gel_' + img2_name)).convert('L')
        elif self.mask_method == 'rgb_mask':
            mask1 = Image.open(os.path.join(self.mask_img_dir, img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dir, img2_name)).convert('L')
            mask1 = mask1.resize((560, 560),interpolation=Image.NEAREST)
            mask2 = mask2.resize((560, 560),interpolation=Image.NEAREST)
        elif self.mask_method == 'touch_mask_gray':
            mask1 = Image.open(os.path.join(self.mask_img_dirs[0], 'gel_' + img1_name)).convert('L')
            mask2 = Image.open(os.path.join(self.mask_img_dirs[0], 'gel_' + img2_name)).convert('L')
            mask1_gray = Image.open(os.path.join(self.mask_img_dirs[1], 'gel_' + img1_name)).convert('L')
            mask2_gray = Image.open(os.path.join(self.mask_img_dirs[1], 'gel_' + img2_name)).convert('L')

        # 触觉图像转换
        serial = img1_name.split('_')[-2]
        
        terrace_map1, sample_contour1 = self.terrace_map_generator(mask1, serial = serial)
        terrace_map2, sample_contour2 = self.terrace_map_generator(mask2, serial = serial)
        touch_img_mask1 = self.touch_transform(terrace_map1)
        touch_img_mask2 = self.touch_transform(terrace_map2)
        
        if self.mask_method == 'touch_mask_gray':
            mask1_gray = self.touch_transform(mask1_gray)
            mask2_gray = self.touch_transform(mask2_gray)
            threshold = 0.05
            touch_img_mask1 = self.sample_image_with_terrace_map(mask1_gray, touch_img_mask1, threshold)
            touch_img_mask2 = self.sample_image_with_terrace_map(mask2_gray, touch_img_mask2, threshold)
        
        # 保存高分辨率版本
        high_res_img1 = self.high_res_transform(img1)
        high_res_img2 = self.high_res_transform(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # 加载标签
        label1 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img1_name).replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, self._remove_prefix(img2_name).replace('.png', '.txt')))
        if self.is_eval:
            return img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2, img1_name, img2_name
        
        return img1, img2, high_res_img1, high_res_img2, touch_img_mask1, touch_img_mask2, sample_contour1, sample_contour2, label1, label2

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)
    
    def sample_image_with_terrace_map(self, img, terrace_map, threshold=0.1):
        """
        使用灰度地形图(terrace_map)对图像进行采样，支持Tensor和NumPy格式
        
        参数:
            img: 图像数据，可以是Tensor[C,H,W]或NumPy数组[H,W,C]
            terrace_map: 灰度地形图，可以是Tensor[1,H,W]或NumPy数组[H,W]
            threshold: 灰度阈值，默认为0.1
            
        返回:
            采样后的图像，与输入格式一致
        """
        # 检查数据类型
        is_tensor = isinstance(img, torch.Tensor)
        
        if is_tensor:
            # 处理Tensor格式数据
            
            # 确保terrace_map也是Tensor格式
            if not isinstance(terrace_map, torch.Tensor):
                terrace_map = torch.from_numpy(terrace_map).float()
            
            # 处理terrace_map的形状，确保它是[H,W]或可以被压缩为[H,W]
            if terrace_map.dim() > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if terrace_map.dim() > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建二值掩码
            mask = (terrace_map > threshold).float()
            
            # 适配不同通道维度的图像
            if img.dim() == 2:  # 单通道图像 [H,W]
                # 直接乘以掩码
                sampled_img = img * mask
            elif img.dim() == 3:  # 多通道图像 [C,H,W]
                # 扩展掩码维度以匹配图像通道
                mask = mask.unsqueeze(0)  # [1,H,W]
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[1:] != img.shape[1:]:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0),  # [1,1,H,W]
                        size=img.shape[1:],
                        mode='nearest'
                    ).squeeze(0)  # [1,H,W]
                    
                # 扩展掩码的通道维度以匹配输入通道数
                mask = mask.expand_as(img)  # [C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
            else:  # 处理批次数据 [B,C,H,W]
                # 扩展掩码维度 [H,W] -> [1,1,H,W]
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                # 确保掩码与输入图像的空间维度匹配
                if mask.shape[2:] != img.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=img.shape[2:],
                        mode='nearest'
                    )
                    
                # 扩展掩码的批次和通道维度以匹配输入
                mask = mask.expand(img.shape[0], img.shape[1], -1, -1)  # [B,C,H,W]
                
                # 应用掩码
                sampled_img = img * mask
                
        else:
            # 处理NumPy格式数据
            
            # 确保terrace_map是NumPy数组
            if isinstance(terrace_map, torch.Tensor):
                terrace_map = terrace_map.detach().cpu().numpy()
                
            # 确保terrace_map是二维数组
            if len(terrace_map.shape) > 2:
                terrace_map = terrace_map.squeeze()
                
                # 如果压缩后仍不是2维，取第一个通道
                if len(terrace_map.shape) > 2:
                    terrace_map = terrace_map[0]
                    
            # 创建掩码
            mask = (terrace_map > threshold).astype(np.float32)
            
            # 适配不同形状的图像
            if len(img.shape) == 2:  # 单通道图像
                sampled_img = img * mask
            else:  # 多通道图像
                # 确定通道维度位置
                if img.shape[-1] in [1, 3, 4]:  # [H,W,C] 格式
                    # 扩展掩码维度以匹配图像通道
                    mask_expanded = np.expand_dims(mask, axis=-1)
                    mask_expanded = np.repeat(mask_expanded, img.shape[-1], axis=-1)
                else:  # [C,H,W] 格式
                    mask_expanded = np.expand_dims(mask, axis=0)
                    mask_expanded = np.repeat(mask_expanded, img.shape[0], axis=0)
                    
                # 应用掩码
                sampled_img = img * mask_expanded
                
        return sampled_img
    
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
            class_name = img_file.split('_')[-2]  # 根据文件名获取类别
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
                # *: use gel_image as template when train touch model
                img_template = 'image_'+class_name+'_0.png'
                index0 = int(img_template.split('_')[-1].split('.')[0])
                assert index0 == 0, 'The first image should be the template'
                class_pairs = [(imgs[i], img_template) 
                          for i in range(0,len(imgs))]
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
    def __init__(self, config, is_train=True, rgb_transform=None, touch_transform=None, is_eval=False, use_fix_template = False, add_noise = False, **kwargs):
        self.is_train = is_train
        root = os.path.join(config.data_path, 'train' if is_train else 'val')
        self.rgb_img_dir = os.path.join(root, 'rgb_images')
        self.touch_img_dir = os.path.join(root, 'touch_images')
        self.label_dir = os.path.join(root, 'labels')
        self.sample_ratio = config.pair_downsample
        self.is_eval = is_eval
        self.add_noise = add_noise
        # TODO: 从**kwargs中获取其他参数
        self.noise_ratio = kwargs.get('noise_ratio', 0.1)
        self.noise_level = kwargs.get('noise_level', 0.1)
        # *:用于测试添加噪声比例的计数器
        

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
        self._validate_images()
        
    def _validate_images(self):
        """验证所有图像文件的完整性，使用更加宽容的验证方法"""
        valid_pairs = []
        valid_touch_pairs = []
        
        # 打印进度信息
        print(f"验证图像文件完整性... 总共 {len(self.rgb_pairs)} 对图像")
        
        for idx, (rgb_pair, touch_pair) in enumerate(zip(self.rgb_pairs, self.touch_pairs)):
            try:
                # 检查RGB图像
                rgb_img1_path = os.path.join(self.rgb_img_dir, rgb_pair[0])
                rgb_img2_path = os.path.join(self.rgb_img_dir, rgb_pair[1])
                # 检查触觉图像
                touch_img1_path = os.path.join(self.touch_img_dir, touch_pair[0])
                touch_img2_path = os.path.join(self.touch_img_dir, touch_pair[1])
                
                # 尝试打开图像但不使用verify()
                valid = True
                for img_path in [rgb_img1_path, rgb_img2_path, touch_img1_path, touch_img2_path]:
                    if not os.path.exists(img_path) or os.path.getsize(img_path) < 100:  # 文件不存在或过小
                        valid = False
                        print(f"警告: 文件不存在或过小: {img_path}")
                        break
                    try:
                        # 尝试简单打开而不是验证
                        with Image.open(img_path) as img:
                            _ = img.size  # 尝试访问属性以确保可以打开
                    except Exception as e:
                        valid = False
                        print(f"警告: 无法打开图像 {img_path}: {str(e)}")
                        break
                        
                # 如果所有图像都有效，添加到有效对列表
                if valid:
                    valid_pairs.append(rgb_pair)
                    valid_touch_pairs.append(touch_pair)
                
            except Exception as e:
                print(f"警告: 跳过图像对: {rgb_pair} - {touch_pair}, 错误: {str(e)}")
                continue
                
            # 打印进度
            if idx % 100 == 0:
                print(f"已验证 {idx}/{len(self.rgb_pairs)} 对图像")
        
        print(f"验证完成。有效图像对: {len(valid_pairs)}/{len(self.rgb_pairs)}")
        self.rgb_pairs = valid_pairs
        self.touch_pairs = valid_touch_pairs
    
    def _safe_load_image(self, path):
        """安全加载图像文件"""
        try:
            with Image.open(path) as img:
                return img.convert('RGB')
        except Exception as e:
            print(f"警告: 加载图像失败 {path}: {str(e)}")
            # 返回一个空白图像作为替代
            return Image.new('RGB', (224, 224), 'black')
    def _generate_pairs(self, use_fix_template = False):
        """生成训练/验证图片对"""
        rgb_pairs = []
        touch_pairs = []
        for class_name, imgs in self.class_to_imgs.items():
            if use_fix_template:
                img_template = 'image_'+class_name+'_0.png'
                index0 = int(img_template.split('_')[-1].split('.')[0])
                assert index0 == 0, 'The first image should be the template'
                class_pairs = [(img_template, imgs[i]) 
                          for i in range(0,len(imgs))]
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
        try:
            rgb_img1_name, rgb_img2_name = self.rgb_pairs[idx]
            touch_img1_name, touch_img2_name = self.touch_pairs[idx]
            
            # 加载并验证图像
            rgb_img1 = self._safe_load_image(os.path.join(self.rgb_img_dir, rgb_img1_name))
            rgb_img2 = self._safe_load_image(os.path.join(self.rgb_img_dir, rgb_img2_name))
            touch_img1 = self._safe_load_image(os.path.join(self.touch_img_dir, touch_img1_name))
            touch_img2 = self._safe_load_image(os.path.join(self.touch_img_dir, touch_img2_name))
            
            # 应用转换
            if self.rgb_transform:
                rgb_img1 = self.rgb_transform(rgb_img1)
                rgb_img2 = self.rgb_transform(rgb_img2)
            if self.touch_transform:
                touch_img1 = self.touch_transform(touch_img1)
                touch_img2 = self.touch_transform(touch_img2)
            noise_sampler = random.uniform(0,1)

            if self.add_noise and noise_sampler < self.noise_ratio:
                noise_level = random.uniform(0.1, self.noise_level)
                rgb_img1 = add_radial_noise(rgb_img1, noise_level)
               
            # 加载标签
            label1 = self._load_label(os.path.join(self.label_dir, rgb_img1_name.replace('.png', '.txt')))
            label2 = self._load_label(os.path.join(self.label_dir, rgb_img2_name.replace('.png', '.txt')))
            
            if self.is_eval:
                return rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2, rgb_img1_name, rgb_img2_name
            if self.add_noise:
                return rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2, noise_level
            
            return rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2
            
        except Exception as e:
            print(f"错误: 加载索引 {idx} 的数据时出错: {str(e)}")
            # 返回数据集中的其他有效样本
            return self.__getitem__((idx + 1) % len(self))
        
    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)