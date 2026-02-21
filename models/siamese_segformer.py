import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig
"""
Siamese SegFormer with multi-scale cross-attention fusion.

Architecture:
  pre  ──► MiT Encoder (shared) ──► [F0,F1,F2,F3] ─┐
                                                      ├─► CrossAttnFusion x4 ──► SegFormer Decoder
  post ──► MiT Encoder (shared) ──► [F0,F1,F2,F3] ─┘

post features = Query
pre  features = Key / Value  (captures what *changed*)
"""
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, f_post, f_pre):
        # Flatten spatial dims: (B, C, H, W) -> (B, HW, C)
        B, C, H, W = f_post.shape
        q = f_post.flatten(2).transpose(1, 2)
        kv = f_pre.flatten(2).transpose(1, 2)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Cross-Attention: Post queries Pre
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm_out(q + attn_out)
        
        # Feed Forward
        q = self.norm_ffn(q + self.ffn(q))
        
        # Reshape back to (B, C, H, W)
        return q.transpose(1, 2).view(B, C, H, W)

class SiameseSegFormer(nn.Module):
    def __init__(self, backbone="nvidia/mit-b2", num_classes=5, decoder_dim=256):
        super().__init__()
        # 1. Shared Encoder (Siamese)
        self.config = SegformerConfig.from_pretrained(backbone, output_hidden_states=True)
        self.encoder = SegformerModel.from_pretrained(backbone, config=self.config)
        
        # Get channel dimensions for the specific MiT backbone
        # B0: [32, 64, 160, 256], B2: [64, 128, 320, 512]
        dims = self.config.hidden_sizes
        
        # 2. Cross-Attention Fusion Layers (one per scale)
        self.fusion_blocks = nn.ModuleList([
            CrossAttentionFusion(dim, num_heads=max(1, dim // 64)) 
            for dim in dims
        ])

        # 3. Lightweight MLP Decoder
        self.linear_layers = nn.ModuleList([nn.Conv2d(dim, decoder_dim, 1) for dim in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.cls = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, pre, post):
        # Encode both images with shared weights
        out_pre = self.encoder(pre).hidden_states
        out_post = self.encoder(post).hidden_states

        # Fuse features at each scale
        fused_features = []
        for i, (f_pre, f_post) in enumerate(zip(out_pre, out_post)):
            fused = self.fusion_blocks[i](f_post, f_pre)
            fused_features.append(fused)

        # Decode
        c1, c2, c3, c4 = fused_features
        target_size = pre.shape[-2:]
        
        # Resize all to 1/4th resolution and concatenate
        outs = []
        for i, x in enumerate(fused_features):
            x = self.linear_layers[i](x)
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            outs.append(x)
            
        x = self.fuse(torch.cat(outs, dim=1))
        x = self.cls(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x
