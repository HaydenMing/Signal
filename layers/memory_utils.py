"""
Utility functions for multi-modal memory bank.

Adapted from CLIMB-ReID (AAAI-2025) for three-modal (RGB+NI+TI) object ReID.
"""

import torch
import torch.nn.functional as F
import tqdm


def extract_multimodal_features(model, dataloader, device='cuda'):
    """
    Extract per-modality global features from the entire training set.

    The model is expected to return (rgb_feat, ni_feat, ti_feat) when
    called with get_multimodal_feat=True.

    Args:
        model:      Signal model
        dataloader: DataLoader over the training set (unshuffled, with val transforms)
        device:     'cuda'

    Returns:
        rgb_features: [N, feat_dim]
        ni_features:  [N, feat_dim]
        ti_features:  [N, feat_dim]
        labels:       [N]
    """
    rgb_features = []
    ni_features = []
    ti_features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for _, (img, pid, camid, camids, target_view, _) in enumerate(
                tqdm.tqdm(dataloader, desc='Extract multi-modal features')):
            img = {
                'RGB': img['RGB'].to(device),
                'NI':  img['NI'].to(device),
                'TI':  img['TI'].to(device),
            }
            target = pid.to(device)
            camid = camid.to(device)
            view_label = target_view.to(device)

            rgb_feat, ni_feat, ti_feat = model(
                img, cam_label=camid, view_label=view_label,
                training=False, get_multimodal_feat=True
            )

            for i, (r, n, t) in enumerate(zip(rgb_feat, ni_feat, ti_feat)):
                labels.append(target[i])
                rgb_features.append(r.cpu())
                ni_features.append(n.cpu())
                ti_features.append(t.cpu())

    labels = torch.stack(labels, dim=0)
    rgb_features = torch.stack(rgb_features, dim=0)
    ni_features = torch.stack(ni_features, dim=0)
    ti_features = torch.stack(ti_features, dim=0)

    return rgb_features, ni_features, ti_features, labels
