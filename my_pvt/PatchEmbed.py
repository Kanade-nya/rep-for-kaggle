from torch import nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


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

'''
x = self.conv1(x)
print("conv1: ", x.shape)  # 3,224,224 -> 768,14,14 ;  3,224,224 -> 64,56,56

x = x.flatten(2)
print("flatten:", x.shape)  # 展平 768,14,14 -> 768,196 ; 64,56,56 -> 64,3136

x = x.transpose(1, 2)
print("transpose:", x.shape)  # 768,196 -> 196,768 ; 64,3136 -> 3136,64

x = self.norm(x)
print("norm:", x.shape)  # 没变
'''