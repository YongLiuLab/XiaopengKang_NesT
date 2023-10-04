import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .transformer import Transformer

class LinearEmbedding(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, out_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
    
class CnnEmbedding(nn.Module):
    def __init__(self, kernel_size, stride, in_c=1,
                 embed_dim=512, padding=0, dropout=0.):
        super().__init__()
        self.conv = nn.Conv3d(in_c, embed_dim,
                              kernel_size, stride,
                              padding)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class FancyCnnEmbedding(nn.Module):
    def __init__(self, in_c=1,
                 embed_dim=512, padding=0, dropout=0.):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16,32,3,2,1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32,64,3,2,1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        self.mlp1 = nn.Linear(768, embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp1(x)
        return x

class Vit(nn.Module):
    def __init__(self, vit_setting,
                 device=torch.device("cuda:0")):
        super().__init__()
        self.image_sizes = vit_setting.image_sizes
        self.patch_sizes = vit_setting.patch_sizes
        self.num_classes = vit_setting.num_classes
        self.embedding = vit_setting.embedding
        self.embed_dim = vit_setting.embed_dim
        self.head = vit_setting.head
        self.depth = vit_setting.depth
        self.mlp_mult = vit_setting.mlp_mult
        self.dim_head = vit_setting.dim_head
        self.dropout = vit_setting.dropout
        self.channels = vit_setting.channels
        self.device = device
        
        p1, p2, p3 = self.patch_sizes
        a = int(self.image_sizes[0] / p1)
        b = int(self.image_sizes[1] / p2)
        c = int(self.image_sizes[2] / p3)
        self.n_patches = a*b*c
        
        self.conv_embedding = nn.Sequential(
           CnnEmbedding(embed_dim=self.embed_dim,
               kernel_size=(p1,p2,p3), stride=(p1,p2,p3), dropout=self.dropout),
        )
        
        self.split = Rearrange('n a (h1 h2) (w1 w2) (d1 d2) -> n (a h1 w1 d1) h2 w2 d2', h1=a, w1=b, d1=c)
        self.fancy_conv = FancyCnnEmbedding(embed_dim=self.embed_dim, dropout=self.dropout)
        
        self.linear_embedding = nn.Sequential(
            Rearrange('n a (h p1) (w p2) (d p3) -> n (a h w d) (p1 p2 p3)',
                              p1=p1, p2=p2, p3=p3),
            nn.Linear(p1*p2*p3, self.embed_dim),
            #LinearEmbedding(p1*p2*p3, 1024, embed_dim, 0)
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        if self.embedding != 'all':
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

            self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, self.embed_dim))

            self.transformer = Transformer(self.embed_dim, self.depth,
                                           self.head, self.dim_head, self.mlp_mult, dropout=self.dropout)

            self.to_class_token = nn.Identity()

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.num_classes)
            )

            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim*3))

            self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, self.embed_dim*3))

            self.transformer = Transformer(self.embed_dim*3, self.depth,
                                           self.head, self.dim_head, self.mlp_mult, dropout=self.dropout)

            self.to_class_token = nn.Identity()

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim*3),
                nn.Linear(self.embed_dim*3, self.num_classes)
            )

            self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        if self.embedding == 'conv':
            x = self.conv_embedding(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embed_dim)
        elif self.embedding == 'fancy_conv':
            x = self.split(x)
            n, m, a, b, c = x.shape
            new_x = torch.zeros((n, m, self.embed_dim), device=self.device)
            for i in range(m):
                xi = x[:, i].view(n, 1, a, b, c)
                new_x[:, i] = self.fancy_conv(xi)
            x = new_x
        elif self.embedding == 'linear':
            x = self.linear_embedding(x)
            #x = self.norm(x)
        elif self.embedding == 'all':
            x1 = self.conv_embedding(x)
            x1 = x1.permute(0, 2, 3, 4, 1).contiguous()
            x1 = x1.view(x1.size(0), -1, self.embed_dim)
            
            x2 = self.split(x)
            n, m, a, b, c = x2.shape
            new_x = torch.zeros((n, m, self.embed_dim), device=self.device)
            for i in range(m):
                xi = x2[:, i].view(n, 1, a, b, c)
                new_x[:, i] = self.fancy_conv(xi)
            x2 = new_x
            
            x3 = self.linear_embedding(x)
            #x3 = self.norm(x3)
            
            x = torch.cat((x1, x2, x3), -1)

        n, p, c = x.shape

        class_tokens = self.class_token.expand(n, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        x += self.pos_embedding[:, :(p + 1)]

        x = self.dropout_layer(x)

        x = self.norm(x)

        x = self.transformer(x)

        x = x[:, 0]
        #x = torch.mean(x, dim=1)
        x = self.to_class_token(x)

        x = self.mlp_head(x)
        return x

    def __repr__(self) -> str:
        tmp = f'----------------------------------\n'\
            f'Deep Learning Model: ViT\n'\
            f'image_sizes: {self.image_sizes}\n'\
            f'patch_sizes: {self.patch_sizes}\n'\
            f'num_classes: {self.num_classes}\n'\
            f'embedding: {self.embedding}\n'\
            f'embed_dim: {self.embed_dim}\n'\
            f'head: {self.head}\n'\
            f'depth: {self.depth}\n'\
            f'mlp_mult: {self.mlp_mult}\n'\
            f'dim_head: {self.dim_head}\n'\
            f'dropout: {self.dropout}\n'
        super_str = super().__repr__()
        return tmp+super_str