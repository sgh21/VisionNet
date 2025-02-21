import torch
import torch.nn as nn
from functools import partial
from Footshone import MAEEncoder, CrossAttention

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
        
        self.query_norm = norm_layer(embed_dim)
        self.context_norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

        # !: 测试新的回归头的效果 证明回归头简单反而效果更好
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),# !: 使用GELU激活函数，进行测试
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, feature_dim)
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
        torch.nn.init.constant_(self.query_norm.bias, 0)
        torch.nn.init.constant_(self.query_norm.weight, 1.0)
        torch.nn.init.constant_(self.context_norm.bias, 0)
        torch.nn.init.constant_(self.context_norm.weight, 1.0)
        torch.nn.init.constant_(self.fc_norm.bias, 0)
        torch.nn.init.constant_(self.fc_norm.weight, 1.0)

        # 初始化regressor
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, mask_ratio=0.75):
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
        feat1_cross = self.cross_attention(self.query_norm(feat1), self.context_norm(feat2))  # [B, N, C] 
        feat2_cross = self.cross_attention(self.query_norm(feat2), self.context_norm(feat1))  # [B, N, C]
        
        # Feature fusion
        feat1_fusion = feat1_cross.mean(dim=1)  # [B, C]
        feat1_fusion = self.fc_norm(feat1_fusion) # [B, C]
        feat2_fusion = feat2_cross.mean(dim=1)
        feat2_fusion = self.fc_norm(feat2_fusion)

        feat_fusion = torch.cat([feat1_fusion, feat2_fusion], dim=1)  # [B, 2C]


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