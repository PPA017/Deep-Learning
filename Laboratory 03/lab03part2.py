import torch
from torch import nn

import torch
from torch import nn


#PATCH EMBEDDING 

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)          
        x = x.flatten(2)                
        x = x.transpose(1, 2)           
        return x


#MSA
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()

        assert embedding_dim % num_heads == 0, "must be divisible"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape

        qkv = self.qkv(x)                            
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.fc_out(out)

        return out


#TRANSOFMER ENCODER
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_size: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#ViT
class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 3,
        embedding_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_size: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )

        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, embedding_dim)
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.encoder(x)

        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)
