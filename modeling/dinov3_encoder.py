"""
DINOv3 Vision Transformer encoder wrapper for multi-modal ReID.

Replaces CLIP's visual encoder with DINOv3 ViT-B/16.
Provides the same interface: forward(x, cv_embed) → (patch_tokens, cls_token).

Supports:
  - Official Facebook .pth format
  - HuggingFace safetensors format (auto key remapping)
"""

import os
import re
import torch
import torch.nn as nn


def load_dinov3_vitb16(pretrained=True):
    """Load a DINOv3 ViT-B/16 from official Facebook URL."""
    from dinov3.hub.backbones import dinov3_vitb16
    model = dinov3_vitb16(pretrained=pretrained)
    return model


def _remap_hf_to_fb(hf_state_dict):
    """
    Remap HuggingFace DINOv3 keys → Facebook DINOv3 keys.
    HF: layer.{i}.attention.q_proj.weight, etc.
    FB:  blocks.{i}.attn.qkv.weight (concatenated Q,K,V), etc.
    """
    fb_dict = {}
    n_layers = 12

    # ---- token-level keys (HF has extra dim, FB uses 2D) ----
    for hf_k, fb_k, squeeze_dim in [
        ('embeddings.cls_token', 'cls_token', True),
        ('embeddings.mask_token', 'mask_token', True),
        ('embeddings.register_tokens', 'storage_tokens', False),
        ('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight', False),
        ('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias', False),
    ]:
        val = hf_state_dict.get(hf_k)
        if val is not None:
            fb_dict[fb_k] = val.squeeze(1) if squeeze_dim and val.dim() == 3 else val

    # ---- per-layer keys ----
    for i in range(n_layers):
        hf_prefix = f'layer.{i}.'
        fb_prefix = f'blocks.{i}.'

        # Q, K, V → concatenated QKV
        q_w = hf_state_dict.get(f'{hf_prefix}attention.q_proj.weight')
        k_w = hf_state_dict.get(f'{hf_prefix}attention.k_proj.weight')
        v_w = hf_state_dict.get(f'{hf_prefix}attention.v_proj.weight')
        if q_w is not None and k_w is not None and v_w is not None:
            fb_dict[f'{fb_prefix}attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_state_dict.get(f'{hf_prefix}attention.q_proj.bias')
        k_b = hf_state_dict.get(f'{hf_prefix}attention.k_proj.bias')
        v_b = hf_state_dict.get(f'{hf_prefix}attention.v_proj.bias')
        if q_b is not None and k_b is not None and v_b is not None:
            fb_dict[f'{fb_prefix}attn.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        # Output projection
        for hf_k, fb_k in [
            ('attention.o_proj.weight', 'attn.proj.weight'),
            ('attention.o_proj.bias', 'attn.proj.bias'),
            ('layer_scale1.lambda1', 'ls1.gamma'),
            ('layer_scale2.lambda1', 'ls2.gamma'),
            ('norm1.weight', 'norm1.weight'),
            ('norm1.bias', 'norm1.bias'),
            ('norm2.weight', 'norm2.weight'),
            ('norm2.bias', 'norm2.bias'),
            ('mlp.up_proj.weight', 'mlp.fc1.weight'),
            ('mlp.up_proj.bias', 'mlp.fc1.bias'),
            ('mlp.down_proj.weight', 'mlp.fc2.weight'),
            ('mlp.down_proj.bias', 'mlp.fc2.bias'),
        ]:
            val = hf_state_dict.get(hf_prefix + hf_k)
            if val is not None:
                fb_dict[fb_prefix + fb_k] = val

    print(f'Remapped {len(fb_dict)} keys from HuggingFace to Facebook format')
    return fb_dict


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
        elif os.path.isdir(pretrained_path):
            # HuggingFace directory with model.safetensors → remap keys to Facebook format
            from safetensors.torch import load_file
            sf_path = os.path.join(pretrained_path, 'model.safetensors')
            if os.path.exists(sf_path):
                self.backbone = load_dinov3_vitb16(pretrained=False)
                hf_state = load_file(sf_path)
                fb_state = _remap_hf_to_fb(hf_state)
                missing, unexpected = self.backbone.load_state_dict(fb_state, strict=False)
                print(f'Loaded DINOv3 from safetensors (HF→FB remapped): {pretrained_path}')
                if missing:
                    print(f'  Missing keys: {missing}')
                if unexpected:
                    print(f'  Unexpected keys: {unexpected}')
            else:
                raise FileNotFoundError(f'No model.safetensors found in: {pretrained_path}')
        elif os.path.exists(pretrained_path):
            # Local .pth file
            self.backbone = load_dinov3_vitb16(pretrained=False)
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f'Loaded DINOv3 from .pth file: {pretrained_path}')
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
