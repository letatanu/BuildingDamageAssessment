import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig

# --- Cross Attention Fusion Block ---
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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
        # f_post = Query, f_pre = Key/Value
        B, C, H, W = f_post.shape
        q = f_post.flatten(2).transpose(1, 2)
        kv = f_pre.flatten(2).transpose(1, 2)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm_out(q + attn_out)
        q = self.norm_ffn(q + self.ffn(q))
        
        return q.transpose(1, 2).view(B, C, H, W)

# --- Standard SegFormer Decoder ---
class SegFormerDecoderHead(nn.Module):
    def __init__(self, in_channels_list, decoder_dim=256, num_classes=2):
        super().__init__()
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, 1) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, features):
        c1_shape = features[0].shape[-2:]
        outs = []
        for i, x in enumerate(features):
            x = self.linear_layers[i](x)
            # Upsample everything to the resolution of the first feature map (1/4 scale)
            if i > 0:
                x = F.interpolate(x, size=c1_shape, mode='bilinear', align_corners=False)
            outs.append(x)
        
        x = self.fuse(torch.cat(outs, dim=1))
        x = self.classifier(x)
        return x

# --- Dual-Head Siamese Model ---
class DualHeadSiameseSegFormer(nn.Module):
    def __init__(self, backbone="nvidia/mit-b2", num_damage_classes=4, decoder_dim=256):
        super().__init__()
        # 1. Encoder
        self.config = SegformerConfig.from_pretrained(backbone, output_hidden_states=True)
        self.encoder = SegformerModel.from_pretrained(backbone, config=self.config)
        dims = self.config.hidden_sizes
        
        # 2. Fusion
        self.fusion_blocks = nn.ModuleList([
            CrossAttentionFusion(dim, num_heads=max(1, dim // 64)) 
            for dim in dims
        ])

        # 3. Decoders
        # Loc Head: 2 classes (Background, Building)
        self.dec_loc = SegFormerDecoderHead(dims, decoder_dim, num_classes=2)
        # Cls Head: N classes (No Damage, Minor, Major, Destroyed)
        self.dec_cls = SegFormerDecoderHead(dims, decoder_dim, num_classes=num_damage_classes)

    def forward(self, pre, post):
        pre = pre.contiguous()
        post = post.contiguous()

        # Encode
        out_pre = self.encoder(pre).hidden_states
        out_post = self.encoder(post).hidden_states
        # Encode
        out_pre = self.encoder(pre).hidden_states
        out_post = self.encoder(post).hidden_states

        # Fuse
        fused = []
        for i, (f_pre, f_post) in enumerate(zip(out_pre, out_post)):
            fused.append(self.fusion_blocks[i](f_post, f_pre))

        # Decode
        logits_loc = self.dec_loc(fused)
        logits_cls = self.dec_cls(fused)
        
        # Upsample to original image size (4x up from 1/4 scale)
        target_size = pre.shape[-2:]
        logits_loc = F.interpolate(logits_loc, size=target_size, mode='bilinear', align_corners=False)
        logits_cls = F.interpolate(logits_cls, size=target_size, mode='bilinear', align_corners=False)

        return logits_loc, logits_cls