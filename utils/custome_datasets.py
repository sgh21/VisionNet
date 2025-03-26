import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from utils.VisionUtils import add_radial_noise


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