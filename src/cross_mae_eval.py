import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import models.TestCrossMAE as crossmae
from pathlib import Path
class EvalDataset(Dataset):
    def __init__(self, args, transform=None):
        root = args.data_path
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.transform = transform
        
        # 按类别组织图片
        self.class_to_imgs = {}
        for img_file in os.listdir(self.img_dir):
            class_name = img_file.split('_')[1]
            if class_name not in self.class_to_imgs:
                self.class_to_imgs[class_name] = []
            self.class_to_imgs[class_name].append(img_file)
            
        self.pairs = self._generate_pairs()
        import random
        num_samples = int(len(self.pairs) * args.sample_ratio)
        self.pairs = random.sample(self.pairs, num_samples)
    def _generate_pairs(self):
        pairs = []
        for class_name, imgs in self.class_to_imgs.items():
            class_pairs = [(imgs[i], imgs[j]) 
                          for i in range(len(imgs))
                          for j in range(i+1, len(imgs))]
            pairs.extend(class_pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        
        img1 = Image.open(os.path.join(self.img_dir, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(self.img_dir, img2_name)).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label1 = self._load_label(os.path.join(self.label_dir, img1_name.replace('.png', '.txt')))
        label2 = self._load_label(os.path.join(self.label_dir, img2_name.replace('.png', '.txt')))
        
        return img1, img2, label1, label2, img1_name, img2_name

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            x, y, rz = map(float, f.read().strip().split(','))
        return torch.tensor([x, y, rz], dtype=torch.float32)
def get_default_args():
    """获取默认参数"""
    parser = get_args_parser()
    default_args = parser.parse_args([])
    return default_args

def load_yaml_config(yaml_path):
    """加载yaml配置并与命令行参数合并"""
    args = get_default_args()
    
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    if yaml_cfg:
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('CrossMAE eval', add_help=False)
    # 基本参数
    parser.add_argument('--config', type=str, default='',
                        help='path to yaml config file')
    parser.add_argument('--weights', default='', type=str,
                        help='checkpoint path')
    
    # 模型参数
    parser.add_argument('--model', default='crossmae_vit_base', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--embed_dim', default=1024, type=int)
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--encoder_num_heads', default=16, type=int)
    parser.add_argument('--cross_num_heads', default=16, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--feature_dim', default=3, type=int)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--mask_ratio', default=1, type=float)
    
    # 评估参数
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./eval_results')
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    
    return parser



def visualize_results(img1_path, img2_path, pred, gt, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示图片对
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    ax1.imshow(img1)
    ax2.imshow(img2)
    
    # 添加预测值和真实值标题
    fig.suptitle(f'Pred: dx={pred[0]:.2f}, dy={pred[1]:.2f}, drz={pred[2]:.2f}\n' + 
                 f'GT: dx={gt[0]:.2f}, dy={gt[1]:.2f}, drz={gt[2]:.2f}')
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def calculate_dim_mae(pred, target):
    mae_x = torch.abs(pred[:,0] - target[:,0])
    mae_y = torch.abs(pred[:,1] - target[:,1])
    mae_rz = torch.abs(pred[:,2] - target[:,2])
    return mae_x, mae_y, mae_rz

def main(args):
    print(f"Loading checkpoint from {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # 创建模型
    model = crossmae.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        pretrained_path=args.mae_pretrained,
        qkv_bias=args.qkv_bias,
    )
    
    # 加载权重
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    print(msg)
    
    model = model.to(args.device)
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    dataset = EvalDataset(args, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建结果保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录评估结果
    all_maes = []
    # 添加进度条
    pbar = tqdm(dataloader, desc='Evaluating', ncols=100)
    with torch.no_grad():
        for batch_idx, (img1, img2, label1, label2, img1_name, img2_name) in enumerate(dataloader):
            img1, img2 = img1.to(args.device), img2.to(args.device)
            label1, label2 = label1.to(args.device), label2.to(args.device)
            
            # 预测
            pred = model(img1, img2, mask_ratio=args.mask_ratio)
            delta_label = label2 - label1
            
            # 反归一化预测值和真实值
            # pred_real = label_denormalize(label_normalize(pred, loss_norm), loss_norm)
            # delta_label_real = label_denormalize(label_normalize(delta_label, loss_norm), loss_norm)
            
            # 计算MAE
            mae_x, mae_y, mae_rz = calculate_dim_mae(pred, delta_label)
            all_maes.append([mae_x.item(), mae_y.item(), mae_rz.item()])
            
            # 更新进度条描述
            pbar.update(1)
            pbar.set_postfix({
                'MAE_X': f'{mae_x.item():.4f}',
                'MAE_Y': f'{mae_y.item():.4f}', 
                'MAE_Rz': f'{mae_rz.item():.4f}'
            })

            # 可视化结果
            for i in range(img1.size(0)):
                img1_path = os.path.join(args.data_path, 'images', img1_name[i])
                img2_path = os.path.join(args.data_path, 'images', img2_name[i])
                save_path = os.path.join(args.output_dir, f'pair_{batch_idx}_{i}.png')
                
                visualize_results(
                    img1_path, 
                    img2_path,
                    pred[i].cpu().numpy(),
                    delta_label[i].cpu().numpy(),
                    save_path
                )
    
    # 计算并打印平均MAE
    all_maes = np.array(all_maes)
    avg_maes = all_maes.mean(axis=0)
    std_maes = all_maes.std(axis=0)
    three_sigmas = 3 * std_maes + avg_maes
    print(f'Average MAE:')
    print(f'X: {avg_maes[0]:.4f} mm')
    print(f'Y: {avg_maes[1]:.4f} mm')
    print(f'Rz: {avg_maes[2]:.4f} deg')

    print(f'mu + 3 * std(X):{three_sigmas[0]:.4f} mm')
    print(f'mu + 3 * std(Y):{three_sigmas[1]:.4f} mm')
    print(f'mu + 3 * std(Rz):{three_sigmas[2]:.4f} deg')
    pbar.close()
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    if args.config:
        args = load_yaml_config(args.config)
        
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)