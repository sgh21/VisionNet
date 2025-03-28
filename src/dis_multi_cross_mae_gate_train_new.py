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
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # Import DistributedSampler
from PIL import Image

import timm.optim.optim_factory as optim_factory
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import models.TestMultiCrossMAEGate as MultiCrossMAEGate
from utils.custome_datasets import MultiCrossMAEDataset

# Ensure the following lines are added to enable distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training(args):
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.local_rank == -1 and hasattr(args, 'local-rank'):
        args.local_rank = getattr(args, 'local-rank')
    
    if args.local_rank != -1:
        args.distributed = True
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.device = torch.device(f'cuda:{args.local_rank}')
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.local_rank
        )
    else:
        args.distributed = False
        args.device = torch.device('cuda:0')

def cleanup():
    """Clean up distributed training processes."""
    if dist.is_initialized():
        dist.destroy_process_group()

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
    parser.add_argument('--use_fix_template', action='store_true')

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
    parser.add_argument('--cross_attention', action='store_true', 
                        help='Use cross attention in the model')
    
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
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--noise_ratio', default=0.1,type=float)
    parser.add_argument('--noise_level', default=0.1, type=float)
    
    # 预训练权重
    parser.add_argument('--rgb_pretrained', default='', type=str)
    parser.add_argument('--touch_pretrained', default='', type=str)
    parser.add_argument('--resume', default='', type=str)

    # Use yaml config
    parser.add_argument('--config', type=str, default='',
                        help='path to yaml config file')
    
    # Distributed training parameters
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='local rank for DistributedDataParallel')  # 兼容性参数
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    # 添加分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    return parser

loss_norm = [2.5, 2.5, 1]  # x,y,rz的归一化权重

def label_normalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label * weight
    return label

def label_denormalize(label, weight=[10, 5, 20]):
    weight = torch.tensor(weight).to(label.device)
    label = label / weight
    return label

def calculate_dim_mae(pred, target):
    mae_x = torch.mean(torch.abs(pred[:, 0] - target[:, 0]))
    mae_y = torch.mean(torch.abs(pred[:, 1] - target[:, 1]))
    mae_rz = torch.mean(torch.abs(pred[:, 2] - target[:, 2]))
    return mae_x, mae_y, mae_rz

def calculate_correlation(noise_levels, lambda_values):
    """计算Pearson相关系数"""
    if len(noise_levels) == 0 or len(lambda_values) == 0:
        return 0.0
    noise_tensor = torch.tensor(noise_levels)
    lambda_tensor = torch.tensor(lambda_values)
    return torch.corrcoef(torch.stack([noise_tensor, lambda_tensor]))[0,1].item()

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
    noise_levels = []
    lambda_values = []

    for data_iter_step, (rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2, noise_level) \
    in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 每累积一定步数调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        rgb_img1 = rgb_img1.to(device, non_blocking=True)
        rgb_img2 = rgb_img2.to(device, non_blocking=True)
        touch_img1 = touch_img1.to(device, non_blocking=True)
        touch_img2 = touch_img2.to(device, non_blocking=True)
        label1 = label1.to(device, non_blocking=True)
        label2 = label2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 混合精度训练
            pred, rgb1_weight = model(rgb_img1, rgb_img2, touch_img1, touch_img2, args.mask_ratio)  # 模型输出
            delta_label = label2 - label1  # 计算标签的差值
            lambda_values.extend(rgb1_weight.cpu().numpy())
            noise_levels.extend(noise_level.cpu().numpy())
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

    # 在epoch结束时计算相关系数并记录
    if log_writer is not None:
        correlation = calculate_correlation(noise_levels, lambda_values)
        log_writer.add_scalar('noise_lambda/correlation', correlation, epoch)
        
        # 添加散点图
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(noise_levels, lambda_values, alpha=0.5)
        plt.xlabel('Noise Level')
        plt.ylabel('Lambda Value')
        plt.title(f'Epoch {epoch}, Correlation: {correlation:.3f}')
        log_writer.add_figure('noise_lambda/scatter_plot', fig, epoch)
        plt.close()
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
        for rgb_img1, rgb_img2, touch_img1, touch_img2, label1, label2, noise_level in metric_logger.log_every(data_loader, 20, header):
            
            rgb_img1 = rgb_img1.to(device, non_blocking=True)
            rgb_img2 = rgb_img2.to(device, non_blocking=True)
            touch_img1 = touch_img1.to(device, non_blocking=True)
            touch_img2 = touch_img2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            batch_size = args.batch_size
            print("\033[1;36;40m Batch_size per GPU:\033[0m", batch_size)
            with torch.cuda.amp.autocast():  # 混合精度验证
                pred, rgb1_weights = model(rgb_img1, rgb_img2, touch_img1, touch_img2, args.mask_ratio)
                delta_label = label2 - label1
                
                # 归一化标签
                delta_label_norm = label_normalize(delta_label, loss_norm)
                pred_norm = label_normalize(pred, loss_norm)
                loss = criterion(pred_norm, delta_label_norm)
                loss_value = loss.item()

                # 反归一化并计算MAE
                delta_label_real = label_denormalize(delta_label_norm, loss_norm)
                pred_real = label_denormalize(pred_norm, loss_norm)
                mae_x, mae_y, mae_rz = calculate_dim_mae(pred_real, delta_label_real)
                
                if args.distributed:
                    # !: 在计算loss时需要同步
                    torch.distributed.barrier()
                    loss_all_reduce = misc.all_reduce_mean(loss_value)
                    mae_x = misc.all_reduce_mean(mae_x.item())
                    mae_y = misc.all_reduce_mean(mae_y.item())
                    mae_rz = misc.all_reduce_mean(mae_rz.item())

                total_x_mae += mae_x * batch_size * args.world_size
                total_y_mae += mae_y * batch_size * args.world_size
                total_rz_mae += mae_rz * batch_size * args.world_size
                total_loss += loss_all_reduce * batch_size * args.world_size

            metric_logger.update(loss=loss_value)  # 更新损失

    # 计算平均值
    
    num_samples = len(data_loader.dataset)
    print("\033[1;36;40m Total dataset len:\033[0m", num_samples)
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

# Modify the main function to support distributed training
def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Initialize distributed training
    setup_distributed_training(args)

    # Set the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build data transforms
    rgb_transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    touch_transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rgb_transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    touch_transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = MultiCrossMAEDataset(args, is_train=True, 
                                        rgb_transform=rgb_transform_train, 
                                        touch_transform=touch_transform_train,  
                                        use_fix_template=args.use_fix_template,
                                        add_noise= args.add_noise,
                                        noise_ratio = args.noise_ratio,
                                        noise_level = args.noise_level)
    
    dataset_val = MultiCrossMAEDataset(args, is_train=False, 
                                       rgb_transform=rgb_transform_val, 
                                       touch_transform=touch_transform_val,  
                                       use_fix_template=args.use_fix_template,
                                       add_noise= args.add_noise,
                                       noise_ratio = args.noise_ratio,
                                       noise_level = args.noise_level)

    # Use DistributedSampler for distributed training
    sampler_train = DistributedSampler(dataset_train, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    sampler_val = DistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    if args.log_dir and dist.get_rank() == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    # 修正batch_size计算
    total_batch_size = args.batch_size
    args.batch_size = int(total_batch_size / args.world_size)
    print(f"\033[1;32;40mTotal batch size: {total_batch_size}, "
          f"Per GPU batch size: {args.batch_size}\033[0m")
    
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
        batch_size=args.batch_size ,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Create the model
    model = MultiCrossMAEGate.__dict__[args.model](
        cross_num_heads=args.cross_num_heads,
        feature_dim=args.feature_dim,
        rgb_pretrained_path=args.rgb_pretrained,
        touch_pretrained_path=args.touch_pretrained,
        qkv_bias=args.qkv_bias,
        cross_attention=args.cross_attention
        )

    model.to(args.device)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[args.local_rank])

    print("Model = %s" % str(model))

    eff_batch_size = total_batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    # Use AdamW optimizer
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
            optimizer, criterion, args.device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Update sampler to ensure that each GPU sees different data
        sampler_train.set_epoch(epoch)

        val_stats = validate(
            model, data_loader_val,
            criterion, args.device, epoch, log_writer,
            args=args
        )

        if args.output_dir and dist.get_rank() == 0 and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model.module,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, save_full_state=False
            )

        # Log statistics (only on rank 0)
        if dist.get_rank() == 0:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
            }
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    cleanup()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.config:
        args = load_yaml_config(args.config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

