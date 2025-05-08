import torch
import torch.nn as nn
from functools import partial
from models.Footshone import MAEEncoder, CrossAttention

class CrossMAE(nn.Module):
    def __init__(
        self,
        img_size=224,
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
        
        # 交叉注意力模块
        # !: 需要检查注意力模块儿是对谁进行注意力计算的
        self.cross_attention = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        
        
        self.fc_norm = norm_layer(embed_dim)
        self.feat_norm = norm_layer(embed_dim*2)
        # 回归头 目前最为有效的回归头
        # self.regressor = nn.Sequential(
        #     nn.Linear(embed_dim * 2, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(embed_dim, embed_dim//4),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(embed_dim//4, feature_dim)
        # )

        # !: 测试新的回归头的效果
        self.regressor = nn.Sequential(
            # nn.Linear(embed_dim * 2, embed_dim),
            # nn.GELU(),
            # nn.Dropout(drop_rate),
            nn.Linear(embed_dim*2, feature_dim)
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
    
    def forward(self, x1, x2, mask_ratio=0.75, **kwargs):
        noise_ratio = kwargs.get('noise_ratio', 0.0)
        noise_level = kwargs.get('noise_level', 0.0)
        x1 = self.add_zero_noise(x1, noise_ratio, noise_level)
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
        # feat_fusion = self.feat_norm(feat_fusion)  # [B, 2C] # !: 测试归一化的效果

        pred = self.regressor(feat_fusion)  # [B, 3]
        
        return pred

def create_crossmae_model(
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

    model = CrossMAE(
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
def crossmae_vit_large_patch16_224(pretrained_path=None, **kwargs):
    return create_crossmae_model(
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

def crossmae_vit_base_patch16_224(pretrained_path=None, **kwargs):
    return create_crossmae_model(
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

crossmae_vit_large = crossmae_vit_large_patch16_224
crossmae_vit_base = crossmae_vit_base_patch16_224

if __name__ == "__main__":
    from config import PARAMS
    mae_model_path = PARAMS['mae_model_path']
    # 测试代码
    model = create_crossmae_model(feature_dim=3,pretrained_path=mae_model_path)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    output = model(x1, x2)
    print(f"Output shape: {output.shape}")  # [2, 3]