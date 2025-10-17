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
        
        # 回归头 目前最为有效的回归头
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim//4, feature_dim)
        )
        # !: 测试新的回归头的效果
        # self.regressor = nn.Sequential(
        #     nn.Linear(embed_dim * 2, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(embed_dim, feature_dim)
        # )

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

        # 初始化fc_norm
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
        feat1, _mask1, _id_restore1, _ = self.encoder(x1, mask_ratio)  # [B, N, C] 
        feat2, _mask2, _id_restore2, _ = self.encoder(x2, mask_ratio)  # [B, N, C]
        
        # Cross attention 互相算注意力更有效
        feat1_cross = self.cross_attention(feat1, feat2)  # [B, N, C] 
        feat2_cross = self.cross_attention(feat2, feat1)  # [B, N, C]
        
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
    cross_num_heads=8,
    mlp_ratio=4.,
    remove_class_token=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    feature_dim=3,
    drop_rate=0.1,
    qkv_bias=False,
    pretrained_path=None,
    **kwargs
):
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
        embed_dim=1024,
        depth=24,
        encoder_num_heads=16,
        cross_num_heads=16,
        mlp_ratio=4.,
        remove_class_token=True,
        feature_dim=3,
        pretrained_path=pretrained_path,
        **kwargs
    )

crossmae_vit_large = crossmae_vit_large_patch16_224

if __name__ == "__main__":
    from config import PARAMS
    mae_model_path = PARAMS['mae_model_path']
    # 测试代码
    model = create_crossmae_model(feature_dim=3,pretrained_path=mae_model_path)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    output = model(x1, x2)
    print(f"Output shape: {output.shape}")  # [2, 3]