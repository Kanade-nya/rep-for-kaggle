import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math

__all__ = [
    'conv_pvt2mlp_tiny', 'conv_pvt2mlp_tiny_chaoyang', 'conv_pvt2mlp_tiny_flowers', 'conv_pvt2mlp_tiny_imagenet'
]
# region
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# endregion


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
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
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def  __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)# 128->64
        # self.bn1   = nn.BatchNorm2d(int(in_channels * reduction))
        # self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)# 64->32
        # self.bn2   = nn.BatchNorm2d(int(in_channels * reduction * 0.5)) 
        # self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)# 64->64
        # self.bn3   = nn.BatchNorm2d(int(in_channels * reduction))
        # self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)# 64->64
        # self.bn4   = nn.BatchNorm2d(int(in_channels * reduction))
        # self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True) # 64->128
        # self.bn5   = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
            )
            
    def forward(self, input):
        output = self.sequeezeBlock(input)
        output = output + self.shortcut(input)
        # output = F.relu(self.bn1(self.conv1(input)), inplace=False)
        # output = F.relu(self.bn2(self.conv2(output)), inplace=False)
        # output = F.relu(self.bn3(self.conv3(output)), inplace=False)
        # output = F.relu(self.bn4(self.conv4(output)), inplace=False)
        # output = F.relu(self.bn5(self.conv5(output)), inplace=False)
        # output += F.relu(self.shortcut(input), inplace=False)
        # output = F.relu(output, inplace=False)
        return output
    
class PvtConv(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chanels=3, dim=128, num_classes=1000, embed_dims=[128, 256, 512],
                 num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 6, 3], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192]):
        super().__init__()

        self.img_size = img_size
        self.mlp_ratios = mlp_ratios
        self.dim = dim
        self.n_fliter_list = n_fliter_list
        self.embed_dims = embed_dims
        self.depths = depths

        # stage1:添加第一层的conv,减少了一层卷积是为了和pvt第二层拼接上
        # region:block1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=n_fliter_list[0],out_channels=64,kernel_size=7, stride=2, padding=3),#112*112 变成1/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, stride=2, padding=1), # 56*56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, stride=2, padding=1), # 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1), #28
            )         # endregion
        self.convRearrange = Rearrange('batch channels height width -> batch (height width) channels')
        # region stage2
        # b 28*28*128
        self.pos_embed2 = nn.Parameter(torch.zeros(1,784, dim))
        # self.dropout2 = nn.Dropout(drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block2 = nn.ModuleList([Block(
            dim=dim, num_heads = num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop = drop_rate, attn_drop=attn_drop_rate, drop_path = dpr[cur+j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        cur += depths[0]
        # endregion
        
        # region stage3:
        # b 28*28*128在Transformer block中没有维度变化
        self.embed3 = PatchEmbed(img_size=img_size//8, patch_size=2, in_chans=embed_dims[0],
                                embed_dim=embed_dims[1])
        num_patches = self.embed3.num_patches
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, embed_dims[1]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads = num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop = drop_rate, attn_drop=attn_drop_rate, drop_path = dpr[cur+j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        cur += depths[1]
        # endregion
        
        # region stage4:
        # b 14*14 320->7*7*521
        self.embed4 = PatchEmbed(img_size//16, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        num_patches = self.embed4.num_patches +1 # +1:cls token
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, embed_dims[2]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)
        self.block4 = nn.ModuleList([Block(
                dim = embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[2])
                for j in range(depths[2])])
    
        self.norm = norm_layer(embed_dims[2])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.head = nn.Linear(embed_dims[2], num_classes)
        # 并行stage
        # self.parralBlock1 = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(embed_dims[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(embed_dims[0]),
        #     nn.ReLU(inplace=True), 
        # )
        # self.parralBlock2 = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dims[0]),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[1], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(embed_dims[1]),
        #     nn.ReLU(inplace=False), 
        # )
        self.parralBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )

        self.parralBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[1], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[1], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True),
        )
        # self.parralBlock1 = BasicBlock(in_channels=embed_dims[0], out_channels=embed_dims[0], stride=1)
        # self.parralBlock2 = BasicBlock(in_channels=embed_dims[0], out_channels=embed_dims[1], stride=2)
        # self.parralBlock3 = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[1], kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dims[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[2], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(embed_dims[2]),
        #     nn.ReLU(inplace=True), 
        # )
        
        # endregion
        #self.norm_temp = norm_layer(embed_dims[2])
        # region init weight like pvt
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        # endregion
    #region default code
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    # 感觉这个会报错暂时没有使用
    # endregion
    def forward_features(self, x):
        # stage1
        x = self.block1(x)
        y = x
        x = self.convRearrange(x)
        # stage2
        B, n, _ = x.shape
        x += self.pos_embed2[:,:n]
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, 28, 28)
        x = x.reshape(B, 28, 28, -1).permute(0, 3, 1, 2).contiguous()

        # parral stage 1 
        x = x + self.parralBlock1(y)
        y = x 

        # stage3
        x, (H, W) = self.embed3(x)
        B, n, _ = x.shape
        # x: 1 196 256
        #pos_embed3: 1 49 256
        x = self.pos_drop3(x + self.pos_embed3[:,:n])
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1 ,2).contiguous()
        # parral stage 2
        x = x + self.parralBlock2(y) 
        # stage4
        x, (H, W) = self.embed4(x)
        B, n,_ = x.shape
        # cls_token = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop4(x + self.pos_embed4[:, :(n)])
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm(x)
        #x = x.reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
        #self.norm_temp(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    def flops(self):
        flops = 0
        H = W = self.img_size
        # conv*3
        for i in range(3):
            flops += 3*3*self.n_fliter_list[i+1] + self.n_fliter_list[i+1]
        # 1*1conv




        
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict
@register_model
def conv_pvt2mlp_tiny_imagenet(pretrained=False, **kwargs):
    model = PvtConv(embed_dims=[128, 320, 512],num_classes=1000,
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[2, 2, 2], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model

@register_model
def conv_pvt2mlp_tiny(pretrained=False, **kwargs):
    model = PvtConv(embed_dims=[128, 320, 512],num_classes=100,
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[2, 2, 2], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model

@register_model
def conv_pvt2mlp_tiny_flowers(pretrained=False, **kwargs):
    model = PvtConv(embed_dims=[128, 320, 512],num_classes=102,
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[2, 2, 2], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model
@register_model
def conv_pvt2mlp_tiny_chaoyang(pretrained=False, **kwargs):
    model = PvtConv(embed_dims=[128, 320, 512], num_classes=4,
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[2, 2, 2], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model
@register_model
def conv_pvt2_small(pretrained=False, **kwargs):
    model = PvtConv(num_classes=1000, embed_dims=[128, 320, 512],
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[3, 6, 3], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model

@register_model
def conv_pvt2_medium(pretrained=False, **kwargs):
    model = PvtConv(num_classes=1000, embed_dims=[128, 320, 512],
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[3, 18, 3], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model

@register_model
def conv_pvt2_large(pretrained=False, **kwargs):
    model = PvtConv(num_classes=1000, embed_dims=[128, 320, 512],
                    num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[8, 27, 3], sr_ratios=[4, 2, 1], n_fliter_list=[3, 48, 96, 192])
    model.default_cfg = _cfg()

    return model
