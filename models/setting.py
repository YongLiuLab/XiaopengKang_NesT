from dataclasses import dataclass

@dataclass
class ModelSetting():
    type: str
    image_sizes: tuple
    num_classes: int
    channels: int

@dataclass
class TransformerSetting(ModelSetting):
    patch_sizes: tuple
    embed_dim: int
    mlp_mult: int
    dim_head: int
    dropout: float

@dataclass
class VitSetting(TransformerSetting):
    embedding: str
    head: int
    depth: int

@dataclass
class NestSetting(TransformerSetting):
    init_patch_embed_sizes: tuple
    layer_heads: tuple
    depthes: tuple

@dataclass
class ResNetSetting(ModelSetting):
    shortcut_type: str

@dataclass
class DanSetting(ModelSetting):
    shortcut_type: str