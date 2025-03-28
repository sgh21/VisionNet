import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed
from functools import partial
from models.Footshone import CrossAttention

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
    
class TestMultiCrossMAEGate(nn.Module):
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
        self.rgb_encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pretrained_path=rgb_pretrained_path,
            remove_class_token=remove_class_token
        )
        # !: 对于触觉传感器，gate返回值始终为1.0
        self.touch_encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pretrained_path=touch_pretrained_path,
            remove_class_token=remove_class_token,
        )
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim *2, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, 2),
            nn.Softmax(dim=-1)
        )

        # 交叉注意力模块
        self.unimodal_cross_attention_rgb = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        self.unimodal_cross_attention_touch = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        self.cross_attention = cross_attention
        if self.cross_attention:
            # 是否使用跨模态交叉注意力
            self.crossmodal_cross_attention = CrossAttention(embed_dim, num_heads=cross_num_heads, dropout=drop_rate, qkv_bias=qkv_bias)
        # 层归一化
        self.fc_norm = norm_layer(2*embed_dim)
        self.feat_norm = norm_layer(2*embed_dim)


        # !: 测试新的回归头的效果 证明回归头简单反而效果更好
        self.regressor_rgb = nn.Sequential(
            # nn.Linear(embed_dim * 2, embed_dim),
            # nn.GELU(),# !: 使用GELU激活函数，进行测试
            # nn.Dropout(drop_rate),
            nn.Linear(embed_dim*2, feature_dim)
        )
        
        self.regressor_touch = nn.Sequential(
            # nn.Linear(embed_dim * 2, embed_dim),
            # nn.GELU(),# !: 使用GELU激活函数，进行测试
            # nn.Dropout(drop_rate),
            nn.Linear(embed_dim*2, feature_dim)
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

        # 初始化regressor
        for m in self.regressor_rgb.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.regressor_touch.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    def forward_gate(self, rgb_latent, touch_latent):
        """
        融合两个模态的特征

        Args:
            rgb_latent (Tensor): 视觉模态 (B, N, L)
            touch_latent (Tensor): 触觉模态 (B, N, L)

        Returns:
            gate (Tensor): 门控向量 (B, 2)
        """
        # !: 对视觉和触觉模态的特征进行拼接
        fusion_latent = torch.cat((rgb_latent.mean(dim=1), touch_latent.mean(dim=1)), dim=1) #(B,2*L)
        
        weight = self.gate(fusion_latent)
      
        return weight
    def forward_contrast(self, x1, x2, dtype='rgb'):
        """
        对比两个特征

        Args:
            x1 (Tensor): 特征1 (B, N, L)
            x2 (Tensor): 特征2 (B, N, L)
            dtype (str, optional): 特征的类型. Defaults to 'rgb'.

        Returns:
            contrast_feat (Tensor): 对比后的特征 (B, 2*L)
            latent1 (Tensor): 特征1 (B, N, L)
            latent2 (Tensor): 特征2 (B, N, L)
        """
        if dtype == 'rgb':
            latent1, mask1, ids_restore1, ids_keep1 = self.rgb_encoder(x1, mask_ratio=0.0)
            latent2, mask2, ids_restore2, ids_keep2 = self.rgb_encoder(x2, mask_ratio=0.0)
            # !: 是否使用不同的跨模态交叉注意力
            # (B, N, L) -> (B, L)
            contrast_feat1 = self.unimodal_cross_attention_rgb(latent1, latent2).mean(dim=1)
            contrast_feat2 = self.unimodal_cross_attention_rgb(latent2, latent1).mean(dim=1)

        else:
            latent1, mask1, ids_restore1, ids_keep1 = self.touch_encoder(x1, mask_ratio=0.0)
            latent2, mask2, ids_restore2, ids_keep2 = self.touch_encoder(x2, mask_ratio=0.0)
            # !: 是否使用不同的跨模态交叉注意力
            # (B, N, L) -> (B, L)
            contrast_feat1 = self.unimodal_cross_attention_touch(latent1, latent2).mean(dim=1)
            contrast_feat2 = self.unimodal_cross_attention_touch(latent2, latent1).mean(dim=1)
            
        
        contrast_feat = self.feat_norm(torch.cat([contrast_feat1, contrast_feat2], dim=1)) # [B, 2*L]

        return contrast_feat, latent1, latent2
   

    
    def forward(self, rgb_img1, rgb_img2, touch_img1, touch_img2, mask_ratio=0.75,mask_rgb = False):
        # 融合跨模态特征
        rgb_feat, rgb1_latent, rgb2_latent = self.forward_contrast(rgb_img1, rgb_img2, dtype='rgb') # [B, 2*L]
        touch_feat, touch1_latent, touch2_latent = self.forward_contrast(touch_img1, touch_img2, dtype='touch') # [B, 2*L]
        
        # 门控向量
        gate = self.forward_gate(rgb1_latent, touch1_latent)
        rgb_weight = gate[:, 0:1] # [B, 1]
        touch_weight = gate[:, 1:] # [B, 1]
        # 融合特征
        # fusion_feat = rgb_weight * rgb_feat + touch_weight * touch_feat # [B, 2*L]
        # fusion_feat = self.fc_norm(fusion_feat)

        # 回归
        pred_rgb = self.regressor_rgb(rgb_feat)
        pred_touch = self.regressor_rgb(touch_feat)
        pred = rgb_weight * pred_rgb + touch_weight * pred_touch
        return pred, rgb_weight

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
        rgb_pretrained_path = None, 
        touch_pretrained_path = None,
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
    print(f"rgb_pretrained_path: {rgb_pretrained_path}")
    print(f"touch_pretrained_path: {touch_pretrained_path}")
    print(f"cross_attention: {cross_attention}")
    
    model = TestMultiCrossMAEGate(
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
        rgb_pretrained_path=rgb_pretrained_path,
        touch_pretrained_path=touch_pretrained_path,
        cross_attention=cross_attention,
        **kwargs
    )
    return model

def multicrossmaegate_vit_large_patch16_224(rgb_pretrained_path = None, touch_pretrained_path = None, **kwargs):
    return create_multicrossmaegate_model(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        encoder_num_heads=16,
        mlp_ratio=4.,
        rgb_pretrained_path=rgb_pretrained_path,
        touch_pretrained_path=touch_pretrained_path,
        **kwargs
    )

def multicrossmaegate_vit_base_patch16_224(rgb_pretrained_path = None, touch_pretrained_path = None, **kwargs):
    return create_multicrossmaegate_model(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        encoder_num_heads=12,
        mlp_ratio=4.,
        rgb_pretrained_path=rgb_pretrained_path,
        touch_pretrained_path=touch_pretrained_path,
        **kwargs
   )

multicrossmaegate_vit_base = multicrossmaegate_vit_base_patch16_224
multicrossmaegate_vit_large = multicrossmaegate_vit_large_patch16_224