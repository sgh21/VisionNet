import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import random
import math
import yaml
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm.optim.optim_factory as optim_factory
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models.CrossMAE import create_crossmae_model
from utils.custome_datasets import CrossMAEDataset

def get_default_args():
    """获取默认参数"""
    parser = get_args_parser()
    default_args = parser.parse_args([])
    return default_args

def load_yaml_config(yaml_path):
    """加载yaml配置并与命令行参数合并"""
    # 获取默认参数
    args = get_default_args()
    
    # 读取yaml配置
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    # 更新参数
    if yaml_cfg:
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('CrossMAE training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--input_size', default=224, type=int)

    # Model parameters
    parser.add_argument('--model', default='crossmae', type=str, metavar='MODEL')
    parser.add_argument('--embed_dim', default=1024, type=int) 
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--encoder_num_heads', default=16, type=int)
    parser.add_argument('--cross_num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--feature_dim', default=3, type=int)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--pair_downsample', type=float, default=1.0)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--log_dir', default='./output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    
    # 预训练权重
    parser.add_argument('--mae_pretrained', default='', type=str)
    parser.add_argument('--resume', default='', type=str)

    # Use yaml config
    parser.add_argument('--config', type=str, default='',
                        help='path to yaml config file')
    return parser
loss_norm = [2.5, 2.5, 1] # x,y,rz的归一化权重
def label_normalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label * weight
    return label

def label_denormalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label / weight
    return label

def calculate_dim_mae(pred, target):
    mae_x = torch.mean(torch.abs(pred[:,0] - target[:,0]))
    mae_y = torch.mean(torch.abs(pred[:,1] - target[:,1]))
    mae_rz = torch.mean(torch.abs(pred[:,2] - target[:,2]))
    return mae_x, mae_y, mae_rz

def train_one_epoch(model: torch.nn.Module, data_loader, optimizer: torch.optim.Optimizer, criterion, 
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train()  # 设置模型为训练模式
    metric_logger = misc.MetricLogger(delimiter="  ")  # 用于记录和打印指标
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 学习率
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # 打印频率

    accum_iter = args.accum_iter  # 梯度累积

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (img1, img2, label1, label2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 每累积一定步数调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label1 = label1.to(device, non_blocking=True)
        label2 = label2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 混合精度训练
            pred = model(img1, img2, args.mask_ratio)  # 模型输出
            delta_label = label2 - label1  # 计算标签的差值
            
            # 归一化标签
            delta_label = label_normalize(delta_label, weight=loss_norm)
            pred = label_normalize(pred, weight=loss_norm)
            loss = criterion(pred, delta_label)  # 计算损失

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter  # 梯度累积
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  # 梯度归零

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)  # 更新损失

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)  # 更新学习率

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # 计算全局损失
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()  # 跨进程同步
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, data_loader, criterion, device, epoch, log_writer=None, args=None):
    model.eval()  # 设置模型为评估模式
    metric_logger = misc.MetricLogger(delimiter="  ")  # 用于记录和打印指标
    header = 'Test:'

    total_loss = 0
    total_x_mae = 0
    total_y_mae = 0
    total_rz_mae = 0

    with torch.no_grad():  # 在验证过程中不计算梯度
        for img1, img2, label1, label2 in metric_logger.log_every(data_loader, 20, header):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            batch_size = img1.size(0)
            
            with torch.cuda.amp.autocast():  # 混合精度验证
                pred = model(img1, img2, args.mask_ratio)
                delta_label = label2 - label1
                
                # 归一化标签
                delta_label_norm = label_normalize(delta_label, loss_norm)
                pred_norm = label_normalize(pred, loss_norm)
                loss = criterion(pred_norm, delta_label_norm)
                
                # 反归一化并计算MAE
                delta_label_real = label_denormalize(delta_label_norm, loss_norm)
                pred_real = label_denormalize(pred_norm, loss_norm)
                mae_x, mae_y, mae_rz = calculate_dim_mae(pred_real, delta_label_real)
                
                total_x_mae += mae_x.item() * batch_size
                total_y_mae += mae_y.item() * batch_size
                total_rz_mae += mae_rz.item() * batch_size
                total_loss += loss.item() * batch_size

            metric_logger.update(loss=loss.item())  # 更新损失

    # 计算平均值
    num_samples = len(data_loader.dataset)
    avg_loss = total_loss / num_samples
    avg_x_mae = total_x_mae / num_samples
    avg_y_mae = total_y_mae / num_samples
    avg_rz_mae = total_rz_mae / num_samples
    
    # 记录到tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/loss', avg_loss, epoch)
        log_writer.add_scalar('val/mae_x_mm', avg_x_mae, epoch)
        log_writer.add_scalar('val/mae_y_mm', avg_y_mae, epoch)
        log_writer.add_scalar('val/mae_rz_deg', avg_rz_mae, epoch)

    metric_logger.update(mae_x=avg_x_mae)
    metric_logger.update(mae_y=avg_y_mae)
    metric_logger.update(mae_rz=avg_rz_mae)

    # 同步并打印结果
    metric_logger.synchronize_between_processes()
    print('* Avg loss {:.3f}, MAE x {:.2f}mm, y {:.2f}mm, rz {:.2f}deg'
          .format(metric_logger.loss.global_avg,
                 metric_logger.mae_x.global_avg,
                 metric_logger.mae_y.global_avg,
                 metric_logger.mae_rz.global_avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建数据增强
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = CrossMAEDataset(args, is_train=True, transform=transform_train)
    dataset_val = CrossMAEDataset(args, is_train=False, transform=transform_val)

    # 使用普通随机采样器
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 创建模型
    model = create_crossmae_model(
        img_size=args.input_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        encoder_num_heads=args.encoder_num_heads,
        cross_num_heads=args.cross_num_heads,
        mlp_ratio=args.mlp_ratio,
        feature_dim=args.feature_dim,
        pretrained_path=args.mae_pretrained,
        qkv_bias=args.qkv_bias,
    )

    model.to(device)

    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    # 使用AdamW优化器
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.SmoothL1Loss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, criterion, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        val_stats = validate(
            model, data_loader_val,
            criterion, device, epoch, log_writer,
            args=args
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model,model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,save_full_state=False
            )

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config:
        args = load_yaml_config(args.config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
