import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed

class MAEEncoderGate(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    """
    MAE encoder specifics
    input: img (N, 3, 224, 224)
    output: latent (N, 14*14+1, 1024),mask (N, 14*14),ids_restore (N, 14*14)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_rate=0.1,
                 pretrained_path = None, remove_class_token=True, keep_constant = -1):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.keep_constant = keep_constant

        # *: 门控网络层，用于评价输入质量
        self.gate = nn.Sequential(
            nn.Linear(embed_dim , embed_dim//4),
            nn.GELU(),# !: 使用GELU激活函数，进行测试
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim//4, 1),
            nn.Sigmoid()
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        self.remove_class_token = remove_class_token

        # 先进行默认权重初始化
        self.initialize_weights()
        # 加载预训练权重
        if pretrained_path is not None:
            self.load_mae_weights(pretrained_path)
        else:
            # 弹出警告 带颜色的警告
            print("\033[1;31;40mWarning: No pretrained MAE weights loaded!\033[0m")

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
    def forward_gate(self, x):
        """
        门控网络层

        Args:
            x (Tensor): 输入特征 (B, N, L)

        Returns:
            gate (Tensor): 门控值 (B, 1)
        """
        if self.keep_constant > 0:
            return torch.ones(x.shape[0], 1).to(x.device) * self.keep_constant
        
        mean_x = x.mean(dim = 1)
        gate = self.gate(mean_x)
        return gate
    
    def forward_encoder(self, x, mask_ratio, keep_mask = None):
        # embed patches
        x = self.patch_embed(x) # (N, L, D) (1, 14*14, 1024)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, keep_mask = keep_mask)
        
        lambda_x = self.forward_gate(x)
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
        return x, mask, ids_restore, ids_keep, lambda_x

    def forward(self, imgs, mask_ratio=0.75, keep_mask = None):
        latent, mask, ids_restore, ids_keep, lambda_x = self.forward_encoder(imgs, mask_ratio, keep_mask = keep_mask)
        return latent, mask, ids_restore, ids_keep, lambda_x
    
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
    
class CrossAttention(nn.Module):
    """使用PyTorch官方实现的多头交叉注意力"""
    # 对应的Patch的embedding进行交叉注意力计算
    def __init__(self, dim, num_heads=8, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=qkv_bias,
            add_bias_kv= qkv_bias,
            add_zero_attn=False
        )
        # 使用xavier初始化
        self._reset_parameters()

    def _reset_parameters(self):
        # q,k,v投影矩阵初始化
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight)
        
        # 输出投影矩阵初始化
        nn.init.xavier_uniform_(self.multihead_attn.out_proj.weight)
        
        # 偏置初始化
        if self.multihead_attn.in_proj_bias is not None:
            nn.init.constant_(self.multihead_attn.in_proj_bias, 0.)
            nn.init.constant_(self.multihead_attn.out_proj.bias, 0.)

    def forward(self, x1, x2):
        """
        Args:
            x1: query sequence (B, N, C)
            x2: key/value sequence (B, N, C)
        """
        # MultiheadAttention使用batch_first=True
        out, _ = self.multihead_attn(
            query=x1,    # (B, N, C)
            key=x2,      # (B, N, C)
            value=x2     # (B, N, C)
        )
        return out