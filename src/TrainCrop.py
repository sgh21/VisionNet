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

from models.ResNet import resnet18
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
            class_name = img_file.split('_')[1]  # roi_image_4024P_x.png -> 4024P
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
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        
        # 加载图片
        img1_path = os.path.join(self.img_dir, img1_name)
        img2_path = os.path.join(self.img_dir, img2_name)
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 应用变换
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        # 加载对应的标签
        label1_path = os.path.join(self.label_dir, img1_name.replace('.png', '.txt'))
        label2_path = os.path.join(self.label_dir, img2_name.replace('.png', '.txt'))
        
        label1 = self._load_label(label1_path)
        label2 = self._load_label(label2_path)
        
        return img1, img2, label1, label2

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
def label_normalize(label,weight=[5,2.5,10]):

    weight = torch.tensor(weight).to(label.device)
    label[:,:1] = label[:,:1]/(-24.175)
    label[:,1:2] = label[:,1:2]/(-24.225)
    label[:,2:3] = label[:,2:3]/180*np.pi
    label = label * weight
    return label

def label_denormalize(label,weight=[5,2.5,10]):
    weight = torch.tensor(weight).to(label.device)
    label = label / weight
    label[:,:1] = label[:,:1]*(-24.175)
    label[:,1:2] = label[:,1:2]*(-24.225)
    label[:,2:3] = label[:,2:3]/np.pi*180
    return label

def calculate_dim_mae(pred, target):
       # 计算每个维度的MAE
    mae_x = torch.mean(torch.abs(pred[:,0] - target[:,0]))
    mae_y = torch.mean(torch.abs(pred[:,1] - target[:,1]))
    mae_rz = torch.mean(torch.abs(pred[:,2] - target[:,2]))
    return mae_x, mae_y, mae_rz
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    model.train()
    total_loss = 0
    processed_batches = 0
    loss_norm = config.get('loss_norm', [10, 5, 20])
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)


    for batch_idx, (img1, img2, label1, label2) in enumerate(pbar):
        batch_size = img1.size(0)
        img1, img2 = img1.to(device), img2.to(device)
        
        label1, label2 = label1.to(device), label2.to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        delta_label = label2 - label1
        delta_label = label_normalize(delta_label,weight=loss_norm )
        
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

def validate(model, val_loader, criterion, device, epoch, writer, config):
    model.eval()
    total_loss = 0
    loss_norm = config.get('loss_norm', [10, 5, 20])
    pbar = tqdm(val_loader, desc='Validating', leave=False)
    total_x_mae = 0
    total_y_mae = 0
    total_rz_mae = 0
    with torch.no_grad():
        for img1, img2, label1, label2 in pbar:
            batch_size = img1.size(0)
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            
            output = model(img1, img2)
            delta_label = label2 - label1
            delta_label = label_normalize(delta_label,weight=loss_norm)

            loss = criterion(output, delta_label)
            delta_label = label_denormalize(delta_label,weight=loss_norm)
            output = label_denormalize(output,weight=loss_norm)

            total_loss += loss.item() * batch_size
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            # 计算每个维度的MAE并记录
            mae_x, mae_y, mae_rz = calculate_dim_mae(output, delta_label)
            total_x_mae += mae_x * batch_size
            total_y_mae += mae_y * batch_size
            total_rz_mae += mae_rz * batch_size
            
    avg_x_mae = total_x_mae / len(val_loader.dataset)
    avg_y_mae = total_y_mae / len(val_loader.dataset)
    avg_rz_mae = total_rz_mae / len(val_loader.dataset)
    writer.add_scalar('MAE/x', avg_x_mae, epoch)
    writer.add_scalar('MAE/y', avg_y_mae, epoch)
    writer.add_scalar('MAE/rz', avg_rz_mae, epoch)
    avg_loss = total_loss / len(val_loader.dataset)
    writer.add_scalar('Loss/val', avg_loss, epoch)

    return avg_loss

def main():
    # 输入训练实验名称
    experiment_name = input("请输入训练实验名称: ")
    
    # 读取配置文件
    with open('./config/SiameseNetCrop.yaml', 'r') as f:
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
        layer_width=config['layer_width']
    ).to(device)
    
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(),
        **config['optimizer_params']
    )
    
    criterion = nn.SmoothL1Loss()
    
    # 创建保存目录
    checkpoint_dir = os.path.join(config['checkpoint_dir'], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建 SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], experiment_name))
    
    # 训练循环
    best_val_loss = float('inf')
    epoch_pbar = tqdm(range(config['epochs']), desc='Training')
    
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,config)
        val_loss = validate(model, test_loader, criterion, device, epoch, writer,config)
        
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