import torch
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

#LayerNorm = partial(nn.InstanceNorm3d, affine = True)
LayerNorm = partial(ChanNorm, eps = 1e-5)

from .transformer import Transformer

def ConvNormMax(dim, dim_out):
    return nn.Sequential(nn.Conv3d(dim, dim_out, 3, padding=1),
                         LayerNorm(dim_out),
                         nn.MaxPool3d(3, stride=2, padding=1))

def block(x, patch_sizes):
    # sort image into blocks
    # (b, n, h, w, d) -> (b, #block, seqlen, n)
    p1, p2, p3 = patch_sizes
    b, n, h, w, d = x.shape
    gh, gw, gd = h//p1, w//p2, d//p3

    x = torch.reshape(x, (b, n, gh, p1, gw, p2, gd, p3))
    x = rearrange(x, 'b n gh p1 gw p2 gd p3-> b (gh gw gd) (p1 p2 p3) n')
    return x, (gh, gw, gd)

def unblock(x, patch_sizes, grid_sizes):
    # (b, #block, seqlen, n) -> (b, n, h, w, d)
    p1, p2, p3 = patch_sizes
    gh, gw, gd = grid_sizes

    x = rearrange(x, 'b (gh gw gd) (p1 p2 p3) n -> b n gh p1 gw p2 gd p3', p1=p1, p2=p2, p3=p3, gh=gh, gw=gw, gd=gd)
    x = rearrange(x, 'b n gh p1 gw p2 gd p3 -> b n (gh p1) (gw p2) (gd p3)')
    return x

class PatchEmbedding(nn.Module):
    def __init__(self, channles, init_patch_embed_sizes, embed_dim):
        super().__init__()
        self.conv = nn.Conv3d(channles, embed_dim,
                              kernel_size=init_patch_embed_sizes,
                              stride=init_patch_embed_sizes)

    def forward(self, x):
        return self.conv(x)

class NesT(nn.Module):
    def __init__(self, nest_setting):
        super().__init__()
        self.image_sizes = nest_setting.image_sizes
        self.patch_sizes = nest_setting.patch_sizes
        self.num_classes = nest_setting.num_classes
        self.channels = nest_setting.channels
        self.init_patch_embed_sizes = nest_setting.init_patch_embed_sizes
        self.embed_dim = nest_setting.embed_dim
        self.layer_heads = nest_setting.layer_heads
        self.depthes = nest_setting.depthes
        self.mlp_mult = nest_setting.mlp_mult
        self.dim_head = nest_setting.dim_head
        self.dropout = nest_setting.dropout

        self.num_hierarchies = len(self.depthes)
        x, y, z = self.image_sizes
        a, b, c = self.init_patch_embed_sizes
        assert (x % a) == 0, 'Image dimensions 1 must be divisible by the init_patch_embed_sizes.'
        assert (y % b) == 0, 'Image dimensions 2 must be divisible by the init_patch_embed_sizes.'
        assert (z % c) == 0, 'Image dimensions 3 must be divisible by the init_patch_embed_sizes.'

        image_sizes_after_embed = (x//a, y//b, z//c)

        x, y, z = image_sizes_after_embed
        a, b, c = self.patch_sizes

        assert (x % a) == 0, 'Embed dimensions 1 must be divisible by the patch size.'
        assert (y % b) == 0, 'Embed dimensions 2 must be divisible by the patch size.'
        assert (z % c) == 0, 'Embed dimensions 3 must be divisible by the patch size.'

        self.patch_embed = PatchEmbedding(self.channels, self.init_patch_embed_sizes, self.embed_dim)

        self.layers = nn.ModuleList([])
        for heads, depth in zip(self.layer_heads, self.depthes):
            self.layers.append(nn.ModuleList([
                Transformer(self.embed_dim, depth, heads, self.dim_head, self.mlp_mult, True, self.dropout),
                ConvNormMax(self.embed_dim, self.embed_dim)
            ]))

        self.mlp_head = nn.Sequential(LayerNorm(self.embed_dim),
                                      Reduce('b n h w d -> b n', 'mean'),
                                      nn.Linear(self.embed_dim, self.num_classes))

    def forward(self, x):
        x = self.patch_embed(x)
        x, gs = block(x, self.patch_sizes)
        for level, (transformer, conv_norm_max) in zip(range(self.num_hierarchies), self.layers):
            # perform transformer
            x = transformer(x)
            x = unblock(x, self.patch_sizes, gs)
            if level != self.num_hierarchies - 1:
                # Aggregate
                x = conv_norm_max(x)
                x, gs = block(x, self.patch_sizes)

        x = self.mlp_head(x)
        return x
    
    def __repr__(self):
        tmp = f'----------------------------------\n'\
            f'Deep Learning Model: NesT\n'\
            f'image_sizes: {self.image_sizes}\n'\
            f'init_patch_embed_sizes: {self.init_patch_embed_sizes}\n'\
            f'patch_sizes: {self.patch_sizes}\n'\
            f'num_classes: {self.num_classes}\n'\
            f'channels: {self.channels}\n'\
            f'embed_dim: {self.embed_dim}\n'\
            f'num_hierarchies: {self.num_hierarchies}\n'\
            f'layer_heads: {self.layer_heads}\n'\
            f'depthes: {self.depthes}\n'\
            f'mlp_mult: {self.mlp_mult}\n'\
            f'dim_head: {self.dim_head}\n'\
            f'dropout: {self.dropout}\n'
        super_str = super().__repr__()
        return tmp+super_str

class NesT2(NesT):
    def __init__(self, nest_setting):
        super().__init__(nest_setting)
        self.mlp_head = nn.Sequential(LayerNorm(self.embed_dim),
                                      Reduce('b n h w d -> b n', 'mean'),
                                      )
        # Tmp solution, adding age and gender
        self.mlp_head2 = nn.Linear(self.embed_dim+2, self.num_classes)

    def forward(self, x, x2):
        x = self.patch_embed(x)
        x, gs = block(x, self.patch_sizes)
        for level, (transformer, conv_norm_max) in zip(range(self.num_hierarchies), self.layers):
            # perform transformer
            x = transformer(x)
            x = unblock(x, self.patch_sizes, gs)
            if level != self.num_hierarchies - 1:
                # Aggregate
                x = conv_norm_max(x)
                x, gs = block(x, self.patch_sizes)

        x = self.mlp_head(x)
        x = torch.cat((x, x2), 1)
        x = self.mlp_head2(x)
        return x