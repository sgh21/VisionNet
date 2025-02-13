import os
import yaml
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from datetime import datetime
from models.SiameseNet import SiameseNetwork
# from models.ResNet import resnet18
from torch.utils.tensorboard import SummaryWriter

class CustomDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        self.img_dir = config['train_image_dir'] if is_train else config['test_image_dir']
        self.label_dir = config['train_label_dir'] if is_train else config['test_label_dir']
        self.sample_ratio = config.get('pair_downsample', 1.0)  # 如果未设置，默认使用全部数据
        
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.img_dir):
            # 从文件名中提取类别 (4024P, 4030P 等)
            class_name = img_file.split('_')[2]  # roi_image_4024P_x.png -> 4024P
            if class_name not in self.class_to_imgs:
                self.class_to_imgs[class_name] = []
            self.class_to_imgs[class_name].append(img_file)
            
        # 生成所有可能的同类图片对并进行下采样
        self.pairs = []
        for class_name, imgs in self.class_to_imgs.items():
            class_pairs = []
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    class_pairs.append((imgs[i], imgs[j]))
            
            # 对每个类别进行下采样
            if self.is_train:  # 只在训练集上进行下采样
                num_samples = int(len(class_pairs) * self.sample_ratio)
                if num_samples > 0:
                    class_pairs = random.sample(class_pairs, num_samples)
            
            self.pairs.extend(class_pairs)
        
        self.transform = transforms.Compose([
            # transforms.ColorJitter(
            #     brightness=config['aug_brightness'],
            #     contrast=config['aug_contrast'], 
            #     saturation=config['aug_saturation']
            # ) if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def parse_image_info(self, img_name):
        img_name = img_name.split('.')[0]
        img_info = img_name.split('_')
        scale_factor = float(img_info[4]) # TODO:暂时不用
        xyxy = list(map(float,img_info[5][1:-1].split(',')))

        return torch.tensor(xyxy)
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        
        # 加载图片
        img1_path = os.path.join(self.img_dir, img1_name)
        img2_path = os.path.join(self.img_dir, img2_name)
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 加载额外信息
        xyxy1 = self.parse_image_info(img1_name)
        xyxy2 = self.parse_image_info(img2_name)

        # 应用变换
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        # 加载对应的标签
        label1_path = os.path.join(self.label_dir, img1_name.replace('.png', '.txt'))
        label2_path = os.path.join(self.label_dir, img2_name.replace('.png', '.txt'))
        
        label1 = self._load_label(label1_path)
        label2 = self._load_label(label2_path)
        
        return img1, img2, xyxy1, xyxy2, label1, label2

    def _load_label(self, label_path):
        """加载标签文件"""
        with open(label_path, 'r') as f:
            content = f.read().strip()
            x, y, rz = map(float, content.split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)

def setup_logger(config, experiment_name):
    log_dir = os.path.join(config.get('log_dir', 'logs'), experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    processed_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch_idx, (img1, img2, embedding_info1, embedding_info2, label1, label2) in enumerate(pbar):
        batch_size = img1.size(0)
        img1, img2 = img1.to(device), img2.to(device)
        embedding_info1, embedding_info2 = embedding_info1.to(device), embedding_info2.to(device)
        label1, label2 = label1.to(device), label2.to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2, embedding_info1, embedding_info2)
        delta_label = label2 - label1
        loss = criterion(output, delta_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        processed_batches += 1
        
        pbar.set_postfix({
            'batch_loss': f'{loss.item():.4f}',
            'avg_loss': f'{(total_loss/((batch_idx+1)*batch_size)):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    pbar = tqdm(val_loader, desc='Validating', leave=False)
    
    with torch.no_grad():
        for img1, img2, embedding_info1, embedding_info2, label1, label2 in pbar:
            batch_size = img1.size(0)
            img1, img2 = img1.to(device), img2.to(device)
            embedding_info1, embedding_info2 = embedding_info1.to(device), embedding_info2.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            
            output = model(img1, img2, embedding_info1, embedding_info2)
            delta_label = label2 - label1
            loss = criterion(output, delta_label)
            
            total_loss += loss.item() * batch_size
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader.dataset)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def main():
    # 输入训练实验名称
    experiment_name = input("请输入训练实验名称: ")
    
    # 读取配置文件
    with open('./config/SiameseNet.yaml', 'r') as f:
        config = yaml.safe_load(f)   
    
    # 设置GPU
    gpu_id = config.get('gpu_id', 0)  # 默认使用GPU 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU: {torch.cuda.get_device_name(gpu_id)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
        # 设置日志和设备

    logger = setup_logger(config, experiment_name)
    logger.info(f'Using device: {device}')
    
    # 创建数据加载器
    train_dataset = CustomDataset(config, is_train=True)
    test_dataset = CustomDataset(config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型、优化器和损失函数
    model = SiameseNetwork(
        # customize_network=resnet18,
        pretrained=config['pretrained'],
        feature_dim=config['feature_dim'],
        layer_width=config['layer_width'],
        embedding_width=config['embedding_width'],
        using_embedding = config['using_embedding']
    ).to(device)
    
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(),
        **config['optimizer_params']
    )
    
    criterion = nn.MSELoss()
    
    # 创建保存目录
    checkpoint_dir = os.path.join(config['checkpoint_dir'], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建 SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], experiment_name))
    
    # 训练循环
    best_val_loss = float('inf')
    epoch_pbar = tqdm(range(config['epochs']), desc='Training')
    
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss = validate(model, test_loader, criterion, device, epoch, writer)
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            logger.info(f'Saved best model with val_loss: {val_loss:.4f}')
    
    writer.close()

if __name__ == '__main__':
    main()