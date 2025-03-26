import torch
import torch.nn as nn
from functools import partial
from models.Footshone import MAEEncoderGate, CrossAttention

class MultiCrossMAEGate(nn.Module):
    def __init__(
        self,
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
        rgb_pretrained_path = None,
        touch_pretrained_path = None,
        cross_attention = True,
    ):
        super().__init__()
        # MAE Encoder
        self.rgb_encoder = MAEEncoderGate(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            pretrained_path=rgb_pretrained_path,
            remove_class_token=remove_class_token
        )
        # !: 对于触觉传感器，gate返回值始终为1.0
        self.touch_encoder = MAEEncoderGate(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            pretrained_path=touch_pretrained_path,
            remove_class_token=remove_class_token,
            keep_constant= 1.0
        )
        # TODO: 实现参数不更新时部分模块的报错，以及实现验证时将图片mask的功能
        # 交叉注意力模块
        self.cross_attention = cross_attention
        if self.cross_attention:
            # 是否使用跨模态交叉注意力
            self.crossmodal_cross_attention = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        
        self.unimodal_cross_attention = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        
        # 层归一化
        self.fc_norm = norm_layer(2*embed_dim)
        self.feat_norm = norm_layer(embed_dim)
        self.rgb_norm = norm_layer(embed_dim)
        self.touch_norm = norm_layer(embed_dim)

        # !: 测试新的回归头的效果 证明回归头简单反而效果更好
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),# !: 使用GELU激活函数，进行测试
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, feature_dim)
        )
        
        # 初始化新增网络层的参数
        self.initialize_weights()

    def initialize_weights(self):
        """初始化新增网络层的参数"""
        # 初始化LayerNorm
        torch.nn.init.constant_(self.fc_norm.bias, 0)
        torch.nn.init.constant_(self.fc_norm.weight, 1.0)
        torch.nn.init.constant_(self.feat_norm.bias, 0)
        torch.nn.init.constant_(self.feat_norm.weight, 1.0)
        torch.nn.init.constant_(self.rgb_norm.bias, 0)
        torch.nn.init.constant_(self.rgb_norm.weight, 1.0)
        torch.nn.init.constant_(self.touch_norm.bias, 0)
        torch.nn.init.constant_(self.touch_norm.weight, 1.0)

        # 初始化regressor
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    
    def forward_fusion_modal(self, rgb_img, touch_img, mask_ratio=0.75, keep_mask = None, mask_rgb = False):
        """
        融合同一组数据的两个模态的特征

        Args:
            rgb_img (Tensor): 视觉模态 (B, N, L)
            touch_img (Tensor): 视觉模态 (B, N, L)
            mask_ratio (float, optional):是否使用mask. Defaults to 0.75.
            keep_mask (dict, optional): 如果使用mask,为了使不用数据间可比，应该使用相同的mask. Defaults to None.

        Returns:
            fusion_feat (Tensor): 融合后的视触觉特征 (B, 2*N, L)
        """
        if keep_mask is None:
            # RGB Encoder
            rgb_latent, rgb_mask, rgb_ids_restore, rgb_ids_keep, rgb_lambda = self.rgb_encoder(rgb_img, mask_ratio, keep_mask=keep_mask)
            keep_mask = {
                'ids_keep': rgb_ids_keep,
                'ids_restore': rgb_ids_restore,
            }
        else :
            rgb_latent, rgb_mask, rgb_ids_restore, rgb_ids_keep, rgb_lambda = self.rgb_encoder(rgb_img, mask_ratio, keep_mask=keep_mask)

        # Touch Encoder
        if mask_rgb:
            mask_ratio = 0.0
            
        touch_latent, touch_mask, touch_ids_restore, touch_ids_keep, touch_lambda = self.touch_encoder(touch_img, mask_ratio, keep_mask=keep_mask)
        
        # # Cross-Modal Cross-Attention
        # context = self.context_norm(torch.cat((rgb_latent, touch_latent), dim=1))

        # !:不使用跨模态交叉注意力
        if not self.cross_attention:
            rgb_query = self.rgb_norm(rgb_latent)
            touch_query = self.touch_norm(touch_latent)
        else:
            rgb_query = self.crossmodal_cross_attention(self.rgb_norm(rgb_latent), self.touch_norm(touch_latent))
            touch_query = self.crossmodal_cross_attention(self.touch_norm(touch_latent), self.rgb_norm(rgb_latent))
        
        # !: 根据图片的重要性进行加权 尝试softmax,不进行cat等
        rgb_weight = rgb_lambda/(rgb_lambda + touch_lambda)
        touch_weight = touch_lambda/(rgb_lambda + touch_lambda)
        rgb_query = rgb_query * rgb_weight
        touch_query = touch_query * touch_weight
        
        # Fusion
        fusion_latent = torch.cat((rgb_query, touch_query), dim=1)
        fusion_latent = self.feat_norm(fusion_latent)

        return fusion_latent, keep_mask

    
    def forward(self, rgb_img1, rgb_img2, touch_img1, touch_img2, mask_ratio=0.75,mask_rgb = False):
        # 融合跨模态特征
        fusion_feat1, keep_mask = self.forward_fusion_modal(rgb_img1, touch_img1, mask_ratio, keep_mask=None, mask_rgb = mask_rgb)
        fusion_feat2, _  = self.forward_fusion_modal(rgb_img2, touch_img2, mask_ratio, keep_mask=keep_mask, mask_rgb = mask_rgb)
       

        # 对比组间特征
        feat1_cross = self.unimodal_cross_attention(fusion_feat1, fusion_feat2)
        feat2_cross = self.unimodal_cross_attention(fusion_feat2, fusion_feat1)

        # 对特征降维
        feat1_downsample = feat1_cross.mean(dim=1) # [B, C]
        feat2_downsample = feat2_cross.mean(dim=1) # [B, C]

        # 融合特征, 并进行回归
        feat_fusion = torch.cat([feat1_downsample,feat2_downsample],dim = 1) # [B, 2*C]
        feat_fusion = self.fc_norm(feat_fusion)

        # 回归
        pred = self.regressor(feat_fusion)

        return pred

def create_multicrossmaegate_model(
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
        pretrained_path= None,
        cross_attention = True,
        **kwargs
):
        # 捕获所有参数的值并打印
    print("\033[1;36;40mCreating MultiCrossMAE model......\033[0m")
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
    print(f"cross_attention: {cross_attention}")
    
    model = MultiCrossMAEGate(
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
        cross_attention=cross_attention,
        **kwargs
    )
    return model

def multicrossmaegate_vit_large_patch16_224(pretrained_path = None, **kwargs):
    return create_multicrossmaegate_model(
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

def multicrossmaegate_vit_base_patch16_224(pretrained_path = None, **kwargs):
    return create_multicrossmaegate_model(
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

multicrossmaegate_vit_base = multicrossmaegate_vit_base_patch16_224
multicrossmaegate_vit_large = multicrossmaegate_vit_large_patch16_224