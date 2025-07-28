"""
Vision Transformer (ViT) implementation for AI-generated image detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten: (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP Block"""
    
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for AI-generated image detection"""
    
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 in_channels=3, 
                 num_classes=2,
                 embed_dim=256, 
                 num_heads=8, 
                 num_layers=12, 
                 mlp_ratio=4.0, 
                 dropout=0.1, 
                 attention_dropout=0.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.norm1.weight, std=0.02)
            nn.init.normal_(block.norm1.bias, 0)
            nn.init.normal_(block.norm2.weight, std=0.02)
            nn.init.normal_(block.norm2.bias, 0)
            
        # Initialize classification head
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token = x[:, 0]  # Take the class token
        x = self.head(cls_token)
        
        return x


def create_vit_model(config):
    """Create Vision Transformer model from config"""
    model = VisionTransformer(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.NUM_CHANNELS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        mlp_ratio=config.MLP_RATIO,
        dropout=config.DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT
    )
    return model 