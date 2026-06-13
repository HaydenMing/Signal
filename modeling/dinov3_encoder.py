"""
DINOv3 Vision Transformer encoder wrapper for multi-modal ReID.

Replaces CLIP's visual encoder with DINOv3 ViT-B/16.
Provides the same interface: forward(x, cv_embed) → (patch_tokens, cls_token).

Key differences from CLIP:
  - Feature dim: 768 (no 512 projection layer)
  - Position encoding: RoPE (flexible for any input resolution)
  - Has 4 register/storage tokens between cls and patch tokens
"""

import os
import torch
import torch.nn as nn


def load_dinov3_vitb16(pretrained=True):
    """
    Load a DINOv3 ViT-B/16 pretrained on LVD-1689M.

    Returns:
        model: DinoVisionTransformer instance
    """
    from dinov3.hub.backbones import dinov3_vitb16

    model = dinov3_vitb16(
        pretrained=pretrained,
    )
    return model


class DINOv3Encoder(nn.Module):
    """
    Wraps DINOv3 ViT to match the interface expected by Signal (same as CLIP encoder).

    Input:  image tensor [B, 3, H, W]
    Output: cls_token [B, 768], patch_tokens [B, N, 768]

    Usage:
        encoder = DINOv3Encoder()
        patch_tokens, cls_token = encoder(img, cv_embed=None)
    """

    def __init__(self, pretrained_path=None):
        super().__init__()

        # Load DINOv3 backbone
        if pretrained_path is None or pretrained_path == '':
            # No local path → download official pretrained weights
            self.backbone = load_dinov3_vitb16(pretrained=True)
            print('Loading DINOv3 from official pretrained URL')
        elif os.path.exists(pretrained_path):
            # Local .pth file provided
            self.backbone = load_dinov3_vitb16(pretrained=False)
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f'Loaded DINOv3 from local path: {pretrained_path}')
        else:
            raise FileNotFoundError(f'DINOv3 pretrained path not found: {pretrained_path}')

        self.embed_dim = 768  # ViT-B/16
        self.backbone.eval()

    @property
    def dtype(self):
        return next(self.backbone.parameters()).dtype

    def forward(self, x, cv_embed=None, modality=None):
        """
        Args:
            x:        [B, 3, H, W]  input image
            cv_embed: optional camera embedding (SIE), added to cls token
            modality: ignored (kept for API compatibility)

        Returns:
            patch_tokens: [B, N, 768]  (N = H//16 * W//16)
            cls_token:    [B, 768]
        """
        B = x.shape[0]

        # DINOv3 forward_features returns dict
        out = self.backbone.forward_features(x)

        cls_token = out['x_norm_clstoken']          # [B, 768]
        patch_tokens = out['x_norm_patchtokens']     # [B, N, 768]

        # Inject camera embedding if provided (same as CLIP SIE)
        if cv_embed is not None:
            if cv_embed.dim() == 3:
                cv_embed = cv_embed.squeeze(1)      # [B, 768]
            cls_token = cls_token + cv_embed.to(cls_token.device)

        return patch_tokens, cls_token
