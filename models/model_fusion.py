import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerModel

class CrossViewAttention(nn.Module):
    """
    Applies Cross-Attention between RGB (Query) and Depth (Key/Value).
    RGB features attend to Depth features to incorporate geometric context.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, rgb_feat, depth_feat):
        """
        rgb_feat:   [B, C, H, W]
        depth_feat: [B, C, H, W]
        """
        B, C, H, W = rgb_feat.size()
        
        # 1. Projections
        proj_query = self.query_conv(rgb_feat).view(B, -1, W * H).permute(0, 2, 1) # B, N, C'
        proj_key   = self.key_conv(depth_feat).view(B, -1, W * H)                  # B, C', N
        proj_value = self.value_conv(depth_feat).view(B, -1, W * H)                # B, C, N

        # 2. Calculate Attention Map
        energy = torch.bmm(proj_query, proj_key) # [B, N, N]
        attention = self.softmax(energy)

        # 3. Apply Attention to Values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # [B, C, N]
        out = out.view(B, C, H, W)
        
        # 4. Residual Connection + Learnable Scale (Gamma)
        # We add the attended depth features to the original RGB features
        out = self.gamma * out + rgb_feat
        
        return out

class RGBDSegformerFusion(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        
        # Load a base model to steal the config and decoder
        self.base = SegformerForSemanticSegmentation.from_pretrained(
            model_name, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        # 1. RGB Stream (Standard Encoder)
        self.rgb_encoder = self.base.segformer
        
        # 2. Depth Stream (Separate Encoder, same architecture)
        # We share weights initially to leverage ImageNet pretraining on both
        # Note: We will repeat the 1-channel depth to 3-channels during forward pass
        self.depth_encoder = SegformerModel.from_pretrained(
            model_name, 
            use_safetensors=True
        )
        
        # 3. Cross Attention Fusion Modules
        # We need one attention module for each of the 4 feature scales.
        # Channels depend on the model size (e.g., mit-b0: [32, 64, 160, 256])
        hidden_sizes = self.base.config.hidden_sizes
        self.fusion_layers = nn.ModuleList([
            CrossViewAttention(dim) for dim in hidden_sizes
        ])
        
        # 4. Decoder (Standard MLP Decoder from SegFormer)
        self.decode_head = self.base.decode_head

    def forward(self, rgb_images, depth_images, labels=None):
        """
        rgb_images: [B, 3, H, W]
        depth_images: [B, 1, H, W] (We will repeat this to 3 channels)
        """
        
        # --- A. Encode RGB ---
        # output_hidden_states=True ensures we get features from all 4 stages
        rgb_outputs = self.rgb_encoder(
            rgb_images, 
            output_hidden_states=True
        )
        rgb_features = rgb_outputs.hidden_states # Tuple of 4 tensors
        
        # --- B. Encode Depth ---
        # Repeat 1-channel depth to 3-channels to use standard pretrained encoder
        depth_input = depth_images.repeat(1, 3, 1, 1)
        depth_outputs = self.depth_encoder(
            depth_input, 
            output_hidden_states=True
        )
        depth_features = depth_outputs.hidden_states # Tuple of 4 tensors

        # --- C. Cross-Attention Fusion ---
        fused_features = []
        for i, (rgb_f, depth_f) in enumerate(zip(rgb_features, depth_features)):
            # Apply attention: RGB attends to Depth
            fused = self.fusion_layers[i](rgb_f, depth_f)
            fused_features.append(fused)

        # --- D. Decode ---
        # The decoder expects the list of features
        logits = self.decode_head(fused_features)
        
        # Upsample logits to original image size (H, W)
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=rgb_images.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(upsampled_logits, labels)

        # Return a namespace-like object to match HF interface
        from transformers.modeling_outputs import SemanticSegmenterOutput
        return SemanticSegmenterOutput(
            loss=loss,
            logits=upsampled_logits,
            hidden_states=None,
            attentions=None
        )