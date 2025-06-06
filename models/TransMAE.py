import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed
from models.Footshone import MAEEncoder, CrossAttention, MaskPatchPooling
from extensions.chamfer_dist import *
from utils.TransUtils import GlobalIlluminationAlignment as IlluminationAlignment
from utils.TransUtils import SSIM

class MAEEncoder(nn.Module):

    """ Masked Autoencoder with VisionTransformer backbone
    """
    """
    MAE encoder specifics
    input: img (N, 3, 224, 224)
    output: latent (N, 14*14+1, 1024),mask (N, 14*14),ids_restore (N, 14*14)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 pretrained_path = None, remove_class_token=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        self.remove_class_token = remove_class_token
        if pretrained_path is not None:
            self.load_mae_weights(pretrained_path)
        else:
            # 弹出警告 带颜色的警告
            print("\033[1;31;40mWarning: No pretrained MAE weights loaded!\033[0m")
            self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_mae_weights(self, pretrained_path):
        """加载预训练的MAE权重"""
        checkpoint = torch.load(pretrained_path, map_location='cpu',weights_only=True)
        model_dict = checkpoint['model']
        filtered_dict = {k: v for k, v in model_dict.items() 
                if not k.startswith('decoder')}
        msg = self.load_state_dict(filtered_dict, strict=False)

        print(f"Loading MAE checkpoint: {msg}")

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] # 16
        h = w = int(x.shape[1]**.5) # 14*14
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, keep_mask = None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if keep_mask is None or mask_ratio == 0.0 :
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]

        else:
            assert keep_mask['ids_keep'].shape == (N, len_keep) and keep_mask['ids_restore'].shape == (N, L)
            ids_keep = keep_mask['ids_keep']
            ids_restore = keep_mask['ids_restore']

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, keep_mask = None):
        # embed patches
        x = self.patch_embed(x) # (N, L, D) (1, 14*14, 1024)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, keep_mask = keep_mask)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.remove_class_token:
            x = x[:, 1:, :]  # remove class token
        return x, mask, ids_restore, ids_keep

    def forward(self, imgs, mask_ratio=0.75, keep_mask = None):
        latent, mask, ids_restore, ids_keep = self.forward_encoder(imgs, mask_ratio, keep_mask = keep_mask)
        return latent, mask, ids_restore, ids_keep
    
class TransMAE(nn.Module):
    def __init__(
        self,
        img_size=224,
        mask_size=560,
        patch_size=16,
        in_chans=3, 
        embed_dim=1024,
        depth=24,
        encoder_num_heads=16,
        cross_num_heads=8,
        mlp_ratio=4.,
        remove_class_token=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        feature_dim=3,
        drop_rate=0.1,
        qkv_bias=False,
        pretrained_path=None,
        illumination_alignment=False,
        use_chamfer_dist=False,
        chamfer_dist_type='L2',
        use_mask_weight=False,
        use_ssim_loss=False,
        pool_mode='mean',
        match_variance=True,
    ):
        super().__init__()
    
        # MAE Encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pretrained_path=pretrained_path,
            remove_class_token=remove_class_token
        )
        # !: 是否使用SSIM, 使用默认参数
        self.use_ssim_loss = use_ssim_loss
        if self.use_ssim:
            self.ssim_loss = SSIM(
                win_size=11,
                sigma=1.5,
                k1=0.01,
                k2=0.03,
                gaussian_weights=True,
                use_sample_covariance=True,
            )

        # !: 是否使用mask权重
        self.use_mask_weight = use_mask_weight
        if self.use_mask_weight:
            mask_ratio = mask_size / img_size
            self.mask_patch_pooling = MaskPatchPooling(
                img_size=mask_size,
                patch_size=int(patch_size*mask_ratio),
                pool_mode=pool_mode
            )
        # 交叉注意力模块
        # !: 需要检查注意力模块儿是对谁进行注意力计算的
        self.cross_attention = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        
        
        self.fc_norm = norm_layer(embed_dim)
        self.feat_norm = norm_layer(embed_dim*2)

        # !: 测试新的回归头的效果
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim*2, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        self.use_chamfer_dist = use_chamfer_dist
        if self.use_chamfer_dist:
            self.chamfer_dist = ChamferDistanceL2() if chamfer_dist_type == 'L2' else ChamferDistanceL1()

        self.illumination_alignment = None
        if illumination_alignment:
            self.illumination_alignment = IlluminationAlignment(
                match_variance=match_variance,
                per_channel=True,
            )
            
        self.initialize_weights() 

    def initialize_weights(self):
        """初始化新增网络层的参数"""
        # 初始化CrossAttention中的线性层
        for name, p in self.cross_attention.named_parameters():
            if 'weight' in name:
                if len(p.shape) > 1:  # 线性层权重
                    torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.constant_(p, 0)

        # 初始化LayerNorm
        torch.nn.init.constant_(self.fc_norm.bias, 0)
        torch.nn.init.constant_(self.fc_norm.weight, 1.0)
        torch.nn.init.constant_(self.feat_norm.bias, 0)
        torch.nn.init.constant_(self.feat_norm.weight, 1.0)

        # 初始化regressor
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def downsample_mask(self, mask, target_size=(224, 224)):
        """
        将掩码从原始尺寸下采样到目标尺寸
        
        Args:
            mask (torch.Tensor): 原始掩码, 形状为 [B, 1, H, W]
            target_size (tuple): 目标尺寸, 形式为 (height, width)
            
        Returns:
            torch.Tensor: 下采样后的掩码, 形状为 [B, 1, target_height, target_width]
        """
        return torch.nn.functional.interpolate(
            mask, 
            size=target_size, 
            mode='bilinear',  # 对于掩码，建议使用'nearest'或'bilinear'
            align_corners=False
        )
    def forward_pred(self, x1, x2, mask_ratio=0.75, scale_factors=None, **kwargs):
        """
        预测变换参数
        
        Args:
            x1: 第一张图像 [B, C, H, W]
            x2: 第二张图像 [B, C, H, W]
            mask_ratio: MAE掩码比例
            scale_factors: 参数缩放系数 (None或tensor[5])，控制预测参数的范围
                           theta, cx, cy, tx, ty的缩放系数
                           
        Returns:
            pred: 预测的变换参数 [B, 5] (theta, cx, cy, tx, ty)
        """
        mask1 = kwargs.get('mask1', None) # [B, 1, H, W]
        mask2 = kwargs.get('mask2', None) # [B, 1, H, W]
        if mask1 is not None and mask2 is not None:
            # 下采样掩码
            mask1_ds = self.downsample_mask(mask1, target_size=(x1.shape[2], x1.shape[3]))
            mask2_ds = self.downsample_mask(mask2, target_size=(x2.shape[2], x2.shape[3]))
            # 仅保留掩码的第一个通道，作为灰度掩码
            threshold = 0.05
            mask1_ds = (mask1_ds > threshold).float()
            mask2_ds = (mask2_ds > threshold).float()
            x1 = x1 * mask1_ds
            x2 = x2 * mask2_ds

        # Encoder features
        feat1, _mask1, _id_restore1, _ids_keep1 = self.encoder(x1, mask_ratio)  # [B, N, C] 
        keep_mask = {
            'ids_keep': _ids_keep1,
            'ids_restore': _id_restore1
        }
        feat2, _mask2, _id_restore2, _ids_keep2 = self.encoder(x2, mask_ratio, keep_mask)  # [B, N, C]
        
        assert torch.sum(_mask1-_mask2) < 1e-6

        # Cross attention 互相算注意力更有效
        # !: 根据官方的实现，在交叉注意力前加入了LayerNorm
        # *： 测试表明，加入LayerNorm后效果没有明显变化
        # !: 归一化层似乎用错了，对于query和context应该使用不同的归一化层
        feat1_cross = self.cross_attention(feat1, feat2)  # [B, N, C] 
        feat2_cross = self.cross_attention(feat2, feat1)  # [B, N, C]
        
        # Feature fusion
        feat1_fusion = feat1_cross.mean(dim=1)  # [B, C]
        feat1_fusion = self.fc_norm(feat1_fusion) # [B, C]
        feat2_fusion = feat2_cross.mean(dim=1)
        feat2_fusion = self.fc_norm(feat2_fusion)

        feat_fusion = torch.cat([feat1_fusion, feat2_fusion], dim=1)  # [B, 2C]
        feat_fusion = self.feat_norm(feat_fusion)  # [B, 2C] # !: 测试归一化的效果

        # 原始预测
        pred = self.regressor(feat_fusion)  # [B, 3]
        
        # 设置默认缩放系数
        if scale_factors is None:
            # 默认缩放系数: [theta, cx, cy, tx, ty]
            # theta的默认范围为[-π, π]
            # 其他参数的默认范围为[-1, 1]
            scale_factors = torch.tensor([7.5/180*torch.pi, 1.0, 1.0], 
                                        device=pred.device)
        else:
            # 确保scale_factors是tensor并且在正确的设备上
            if not isinstance(scale_factors, torch.Tensor):
                scale_factors = torch.tensor(scale_factors, device=pred.device)
            elif scale_factors.device != pred.device:
                scale_factors = scale_factors.to(pred.device)
        
        # 应用缩放系数
        # 扩展维度以适应批次大小
        scale_factors = scale_factors.view(1, -1)
        pred = pred * scale_factors
        
        return pred
    def forward_transfer(self, x, params, CXCY=None):
        """
        使用3参数[theta,tx,ty]应用仿射变换到输入图像
        
        Args:
            x (Tensor): 输入数据，[B, C, H, W]
            params (Tensor): 变换参数，[B, 3] (theta, tx, ty)
                其中theta是旋转角度(-π, π)
                [tx, ty]构成平移向量(-1, 1)
            CXCY:[cx, cy]为旋转中心坐标(-1, 1)
        Returns:
            Tensor: 变换后的图像，[B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 提取参数
        theta = params[:, 0]  # 旋转角度，(-π, π)范围
        tx = params[:, 1]  # x方向平移，(-1, 1)范围
        ty = params[:, 2]  # y方向平移，(-1, 1)范围
        if CXCY is not None:
            # 转化为tensor
            cx = torch.full((B, 1, 1), CXCY[0], device=device)
            cy = torch.full((B, 1, 1), CXCY[1], device=device)
        else:
            # 计算旋转中心
            cx = torch.zeros(B, 1, 1, device=device)
            cy = torch.zeros(B, 1, 1, device=device)

        # 构建旋转矩阵
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 旋转矩阵元素
        a = cos_theta
        b = -sin_theta
        c = sin_theta
        d = cos_theta
        
        # 创建归一化网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 扩展网格坐标到批次维度
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        
        # 计算行列式(稳定性检查)
        det = a * d - b * c
        eps = 1e-6
        safe_det = torch.where(torch.abs(det) < eps, 
                           torch.ones_like(det) * eps * torch.sign(det), 
                           det)
        
        # 计算逆变换矩阵(用于逆向映射)
        inv_a = d / safe_det
        inv_b = -b / safe_det
        inv_c = -c / safe_det
        inv_d = a / safe_det
        
        # 将参数调整为[B,1,1]形状，方便广播
        inv_a = inv_a.view(B, 1, 1)
        inv_b = inv_b.view(B, 1, 1)
        inv_c = inv_c.view(B, 1, 1)
        inv_d = inv_d.view(B, 1, 1)
        tx = tx.view(B, 1, 1)
        ty = ty.view(B, 1, 1)
        
        # 逆向映射坐标计算（从输出找输入）:
        # 1. 先应用平移的逆变换
        x_after_trans = grid_x - tx  
        y_after_trans = grid_y - ty
        
        # 2. 将坐标相对于旋转中心
        x_centered = x_after_trans - cx
        y_centered = y_after_trans - cy
        
        # 3. 应用旋转的逆变换
        x_unrotated = inv_a * x_centered + inv_b * y_centered
        y_unrotated = inv_c * x_centered + inv_d * y_centered
        
        # 4. 加回旋转中心
        x_in = x_unrotated + cx
        y_in = y_unrotated + cy
        
        # 组合成采样网格
        grid = torch.stack([x_in, y_in], dim=-1)  # [B, H, W, 2]
        
        # 使用grid_sample实现双线性插值 bicubic
        return torch.nn.functional.grid_sample(
            x, 
            grid, 
            mode='bilinear',      
            padding_mode='zeros', 
            align_corners=True    
        )
    def forwrd_weights_map(self, shape, device, **kwargs):
        """
        生成根据不同方法创建的权重图，用于在图像变换过程中赋予不同区域不同的重要性
        
        Args:
            shape (tuple): 输出权重图的形状，格式为(B, C, H, W)，其中B是批次大小，
                        C是通道数(通常为1)，H和W是图像高度和宽度
            device (torch.device): 权重图应该在的设备(CPU或GPU)
            **kwargs: 额外的参数，根据method不同而变化
                - 对于'gaussian'，可以提供'sigma'参数控制高斯分布的扩散程度:
                    - sigma (list/float): [sigma_x, sigma_y]或单个值，控制x和y方向的方差
                - 对于'touch_mask'，需要提供'touch_mask'参数:
                    - touch_mask (Tensor): 形状为[1, 1, H, W]的掩码张量
        
        Returns:
            torch.Tensor: 批次化的权重图，形状为[B, 1, H, W]，可用于后续的权重损失计算
        
        注意:
            - 对于'gaussian'方法，基础权重图会被缓存，只在必要时重新计算
            - 高斯权重图会被归一化，使总权重和等于像素数量(H*W)
            - 触摸掩码方法需要外部提供掩码，适合用户交互场景
        """
        B, C, H, W = shape

        
        # 创建基础权重图（只计算一次并缓存）
        if not hasattr(self, 'base_weight_map') \
        or self.base_weight_map.shape[2:] != (H, W) \
        or self.base_weight_map.device != device :
            
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )

            sigma = kwargs.get('sigma', [0.5, 0.5])
            sigma_x, sigma_y = sigma
            # 计算到图像中心的距离
            # 对 x,y 分别计算高斯权重衰减
            dist_x = x_grid.pow(2) / (2 * sigma_x**2)
            dist_y = y_grid.pow(2) / (2 * sigma_y**2)
            
            # 使用高斯函数生成权重图
            base_weights = torch.exp(-(dist_x + dist_y))
            
            # 归一化权重，使权重总和为像素数量
            base_weights = base_weights * (H * W) / base_weights.sum()
            
            # 保存为单通道图像
            self.base_weight_map = base_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 将基础权重图扩展到批次大小
        batch_weight_map = self.base_weight_map.expand(B, 1, H, W)
        return batch_weight_map

    def forward_loss(self, x1, x2, params, CXCY=None, method = 'gaussian', **kwargs):
        """
        计算两个图像之间的MSE损失，权重图会随图像变换而变换
        
        Args:
            x1 (Tensor): 输入图像1，形状为[B, C, H, W]
            x2 (Tensor): 输入图像2，形状为[B, C, H, W]
            params (Tensor): 变换参数，[B, 3] (theta, tx, ty)
            sigma (List(float)): 高斯权重的标准差
            CXCY (list): 旋转中心坐标 [cx, cy]
                
        Returns:
            Tensor: 加权MSE损失
        """
        B, C, H, W = shape =  x1.shape
        device = x1.device

        # 将基础权重图扩展到批次大小
        if method == 'gaussian':
            batch_weight_map = self.forwrd_weights_map(shape, device, method=method, **kwargs)
            # 应用变换到权重图
            transformed_weight_map = self.forward_transfer(batch_weight_map, params, CXCY=CXCY)
        
        elif method == 'touch_mask' or method == 'rgb_mask' or method == 'touch_mask_gray':
            mask1 = kwargs.get('mask1', None) # [B, 1, H, W]
            mask2 = kwargs.get('mask2', None) # [B, 1, H, W]
            if mask1 is None or mask2 is None:
                raise ValueError("For 'touch_mask' method, 'mask1' and 'mask2' must be provided.")
            
            transformed_mask2 = self.forward_transfer(mask2, params, CXCY=CXCY)
            # 将两张图像的掩码结合
            transformed_weight_map = torch.max(mask1, transformed_mask2)
            
            # 归一化权重图，使权重总和为像素数量
            pixel_count = H * W
            current_sum = transformed_weight_map.sum(dim=(2, 3), keepdim=True)
            
            # 归一化系数 = 期望总和 / 当前总和
            scale_factor = pixel_count / (current_sum + 1e-8)
            # 应用归一化
            transformed_weight_map = transformed_weight_map * scale_factor
        
        # 扩展到匹配通道数
        weights = transformed_weight_map.expand(B, C, H, W)
        
        # 计算MSE损失并应用权重
        if self.use_ssim_loss:
            # 使用SSIM损失
            ssim_loss, loss_map = self.ssim_loss(x1, x2, full=True, wo_light=True, loss=True)
        else:
            loss_map = torch.nn.functional.mse_loss(x1, x2, reduction='none')
        loss = (loss_map * weights).sum() / (B * C * H * W)
        
        return loss
    
    def forward_chamfer_loss(self, xy1, xy2):
        """
        计算Chamfer距离损失
        
        Args:
            xy1 (Tensor): 输入图像1，形状为[B, N, 2]
            xy2 (Tensor): 输入图像2，形状为[B, N, 2]
        
        Returns:
            Tensor: Chamfer距离损失
        """
        # 计算Chamfer距离
        xyz1 = torch.cat([xy1, torch.zeros_like(xy1[:,:, :1])], dim=-1)  # [B, N, 3]
        xyz2 = torch.cat([xy2, torch.zeros_like(xy2[:,:, :1])], dim=-1)  # [B, N, 3]
        chamfer_loss = self.chamfer_dist(xyz1, xyz2)
        return chamfer_loss
    
    def add_zero_noise(self, x, noise_ratio=0.0, noise_level=0.0):
        """
        以一定概率给图像添加零值噪声，噪声位置在各通道间保持一致（完全向量化实现）
        
        Args:
            x (torch.Tensor): 输入图像，形状 [B, C, H, W]
            noise_ratio (float): 应用噪声的概率，范围 [0, 1]
            noise_level (float): 噪声的强度，即将多少比例的像素点(而非像素值)置为0，范围 [0, 1]
            
        Returns:
            torch.Tensor: 应用噪声后的图像，形状与输入相同 [B, C, H, W]
        """
        if noise_ratio <= 0.0 or noise_level <= 0.0:
            return x  # 如果噪声比例或噪声等级为0，直接返回原图像
        
        device = x.device
        B, C, H, W = x.shape
        
        # 创建输出图像的副本
        noisy_x = x.clone()
        
        # 为每个样本生成随机概率，决定是否应用噪声
        apply_noise = torch.rand(B, device=device) < noise_ratio
        
        if not apply_noise.any():
            return noisy_x  # 如果没有样本需要应用噪声，直接返回
        
        # 计算需要置为0的像素点数量
        num_pixels = H * W
        num_zeros = int(num_pixels * noise_level)
        
        # 创建批量掩码 [B, 1, H, W]，初始值全为1（保留所有像素）
        batch_mask = torch.ones((B, 1, H, W), device=device)
        
        # 对需要添加噪声的样本创建噪声掩码
        noise_indices = torch.nonzero(apply_noise, as_tuple=True)[0]
        for idx in noise_indices:
            # 为当前样本生成随机索引
            flat_indices = torch.randperm(num_pixels, device=device)[:num_zeros]
            h_indices = (flat_indices // W)
            w_indices = flat_indices % W
            
            # 在相应位置设置掩码值为0
            batch_mask[idx, 0, h_indices, w_indices] = 0.0
        
        # 应用掩码到所有通道
        noisy_x = noisy_x * batch_mask
        
        return noisy_x
    
    def forward(self, x1, x2, 
                high_res_x1=None, high_res_x2=None, 
                sample_contourl1=None, sample_contourl2=None,
                mask_ratio=0.75, CXCY=None, 
                method='gaussian', **kwargs):
        """
        模型前向传播，包含高分辨率损失计算
        
        Args:
            x1, x2: 低分辨率输入图像 (224x224)
            high_res_x1, high_res_x2: 高分辨率输入图像 (可选，用于高精度损失计算)
        """
                
        # !: 归一化光照
        if self.illumination_alignment is not None:
            x1 = self.illumination_alignment(x1, x2)
        
        # !: 添加噪声
        noise_ratio = kwargs.get('noise_ratio', 0.0)
        noise_level = kwargs.get('noise_level', 0.0)
        x1 = self.add_zero_noise(x1, noise_ratio, noise_level)
        # 从低分辨率图像预测变换参数
        pred = self.forward_pred(x1, x2, mask_ratio=mask_ratio, **kwargs)

        # 计算损失 - 优先使用高分辨率图像
        if high_res_x1 is not None and high_res_x2 is not None:
            # 应用相同的变换参数到高分辨率图像
            high_res_x2_trans = self.forward_transfer(high_res_x2, pred, CXCY=CXCY)

            # 在高分辨率上计算损失
            trans_diff_loss = self.forward_loss(
                high_res_x1, 
                high_res_x2_trans, 
                params=pred, 
                CXCY=CXCY,
                method=method,
                **kwargs,
                )
            
            x2_trans = high_res_x2_trans
        else:
            # 应用变换到低分辨率图像（用于可视化）
            x2_trans = self.forward_transfer(x2, pred, CXCY=CXCY)
            # 回退到低分辨率损失
            trans_diff_loss = self.forward_loss(
                x1, 
                x2_trans, 
                params=pred, 
                CXCY=CXCY,
                method=method,
                **kwargs,
                )
        chamfer_loss = 0
        if self.use_chamfer_dist and sample_contourl1 is not None and sample_contourl2 is not None:
            # 计算Chamfer距离损失
            chamfer_loss = self.forward_chamfer_loss(sample_contourl1, sample_contourl2)

        return pred, trans_diff_loss, chamfer_loss, x2_trans


def create_transmae_model(
    img_size=224,
    patch_size=16,
    in_chans=3, 
    embed_dim=1024,
    depth=24,
    encoder_num_heads=16,
    cross_num_heads=16,
    mlp_ratio=4.,
    remove_class_token=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    feature_dim=3,
    drop_rate=0.1,
    qkv_bias=False,
    pretrained_path=None,
    **kwargs
):
    # 捕获所有参数的值并打印
    print("\033[1;36;40mCreating CrossMAE model......\033[0m")
    print(f"img_size: {img_size}")
    print(f"patch_size: {patch_size}")
    print(f"in_chans: {in_chans}")
    print(f"embed_dim: {embed_dim}")
    print(f"depth: {depth}")
    print(f"encoder_num_heads: {encoder_num_heads}")
    print(f"cross_num_heads: {cross_num_heads}")
    print(f"mlp_ratio: {mlp_ratio}")
    print(f"remove_class_token: {remove_class_token}")
    print(f"norm_layer: {norm_layer}")
    print(f"feature_dim: {feature_dim}")
    print(f"drop_rate: {drop_rate}")
    print(f"qkv_bias: {qkv_bias}")
    print(f"pretrained_path: {pretrained_path}")

    model = TransMAE(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans, 
        embed_dim=embed_dim,
        depth=depth,
        encoder_num_heads=encoder_num_heads,
        cross_num_heads=cross_num_heads,
        mlp_ratio=mlp_ratio,
        remove_class_token=remove_class_token,
        norm_layer=norm_layer,
        feature_dim=feature_dim,
        drop_rate=drop_rate,
        qkv_bias=qkv_bias,
        pretrained_path=pretrained_path,
        **kwargs
    )
    return model
def transmae_vit_large_patch16_224(pretrained_path=None, **kwargs):
    return create_transmae_model(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        encoder_num_heads=16,
        mlp_ratio=4.,
        pretrained_path=pretrained_path,
        **kwargs
    )

def transmae_vit_base_patch16_224(pretrained_path=None, **kwargs):
    return create_transmae_model(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        encoder_num_heads=12,
        mlp_ratio=4.,
        pretrained_path=pretrained_path, 
        **kwargs
        )

transmae_vit_large = transmae_vit_large_patch16_224
transmae_vit_base = transmae_vit_base_patch16_224

if __name__ == "__main__":
    from config import PARAMS
    mae_model_path = PARAMS['mae_model_path']
    # 测试代码
    model = create_transmae_model(feature_dim=3,pretrained_path=mae_model_path)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    output = model(x1, x2)
    print(f"Output shape: {output.shape}")  # [2, 3]