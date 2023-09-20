from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

"""MLP"""


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


"""Patch Embeded"""


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size 224
        img_size = to_2tuple(img_size)
        # patch_size 16, 16 * 16  大小 的patch
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."

        # H = 224 / 16 = 14  14 *14=196个patch
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # conv(in = 3,out = 768, kernel_size = 16 stride  = 16), in = 224*224 out = 14 14*14*768 (768 = 16*16*3)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # [B,3,224,224]
        B, C, H, W = x.shape
        ''' 
        flatten 把指定维度后的维度合并到一起
        LayerNorm 归一化
        '''
        # proj [B,3,224,224] -> [B,768,14,14], flatten  [B,768,14,14] -> [B,768,196] transpose -> [B,196,768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # H,W = 14
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # return x,(14,14)
        return x, (H, W)


"""Block"""


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # print('---into block---')
        # y = x
        # x = self.attn(self.norm1(x),H,W)
        # # print('after attn : ',x.shape)
        # x = y + self.drop_path(x)
        # print('ater cancha: ',x.shape)

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('after mlp:' ,x.shape)
        # print('---end block---')

        return x


"""Attention"""


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 特征维度和头数
        self.dim = dim
        self.num_heads = num_heads
        # 每一个head 的dimension
        head_dim = dim // num_heads
        # 1 / 根号d
        self.scale = qk_scale or head_dim ** -0.5
        # 生成qkv
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 全连接层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # SRA
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        # [ B , num_patches + 1, embed_dim] # B, 197,768

        B, N, C = x.shape
        # [B,N , 8 , C // 8 ], permute [B,N,8,C//8] -> [B,8,N,C//8]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # permute -> [B,C,N] reshape ->[B,C,H,W]
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # @ 对每个headers计算 q * k^T
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class REPVTModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 depths=[3, 4, 6, 3], num_stages=4, drop_path_rate=0., drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes  # 分类数量
        self.depths = depths  # 每层的block块数
        self.num_stages = num_stages  # 一共4个阶段
        # dpr,cur负责drop_path的
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.cur = 0

        # --stage 1--
        self.patch_embed_1 = PatchEmbed(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_channels,
                                        embed_dim=embed_dims[0])
        self.num_patches_1 = self.patch_embed_1.num_patches
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.num_patches_1, embed_dims[0]))
        self.pos_drop_1 = nn.Dropout(p=drop_rate)

        self.block_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        # 移动cur
        self.cur += depths[0]

        # --stage 2--
        self.patch_embed_2 = PatchEmbed(img_size=img_size // 4,
                                        patch_size=2,
                                        in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])
        self.num_patches_2 = self.patch_embed_2.num_patches
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.num_patches_2, embed_dims[1]))
        self.pos_drop_2 = nn.Dropout(p=drop_rate)

        self.block_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        # 移动cur
        self.cur += depths[1]

        # --stage 3--
        self.patch_embed_3 = PatchEmbed(img_size=img_size // 8,
                                        patch_size=2,
                                        in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])
        self.num_patches_3 = self.patch_embed_3.num_patches
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, self.num_patches_3, embed_dims[2]))
        self.pos_drop_3 = nn.Dropout(p=drop_rate)

        self.block_3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        # 移动cur
        self.cur += depths[2]

        # --stage 4--
        self.patch_embed_4 = PatchEmbed(img_size=img_size // 16,
                                        patch_size=2,
                                        in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])
        self.num_patches_4 = self.patch_embed_4.num_patches + 1
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, self.num_patches_4, embed_dims[3]))
        self.pos_drop_4 = nn.Dropout(p=drop_rate)

        self.block_4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        # 移动cur
        self.cur += depths[3]

        # after stage 4
        self.norm = norm_layer(embed_dims[3])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.pos_embed_4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward(self, x):
        B = x.shape[0]
        # stage 1
        x, (H, W) = self.patch_embed_1(x)  # patch embedding # B,3,224,224 -> B,3136,64 H,w = 56
        x = self.pos_drop_1(x + self.pos_embed_1)  # 加上position embedding 和 drop out , pos 的张量 [1,3136,64] 会自动扩展
        # print('after postion:',x.shape) x.shape B ,3136,64
        for blk in self.block_1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage1:', x.shape)

        # stage 2
        x, (H, W) = self.patch_embed_2(x)  # // 2  # B,64,56,56 -> B,128,28,28(784) -> B,784,128
        x = self.pos_drop_2(x + self.pos_embed_2)
        for blk in self.block_2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage2:', x.shape)

        # stage 3
        x, (H, W) = self.patch_embed_3(x)  # // 2  # B,128,28,28 -> B,320,14,14 -> B,196,320
        x = self.pos_drop_3(x + self.pos_embed_3)
        for blk in self.block_3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('after stage3:', x.shape)

        # stage 4
        x, (H, W) = self.patch_embed_4(x)  # // 2  # B,320,14,14 -> B,512,7,7

        # cls and pos embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop_4(x + self.pos_embed_4)
        for blk in self.block_4:
            x = blk(x, H, W)
        # print('after stage4:', x.shape)

        # head
        x = self.norm(x)
        x = self.head(x[:, 0])
        # print('last x :', x.shape)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# # 224*224*3 模拟图片
# image = torch.randn([64, 3, 224, 224])
#
# model = REPVTModel(patch_size=4, embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8],
#               mlp_ratios=[8, 8, 4, 4], qkv_bias=True)
#
# out = model(image)
# print("最后输出：", out.shape)


@register_model
def re_pvt(pretrained=False, **kwargs):
    model = REPVTModel(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2,2,2,2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
