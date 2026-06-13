"""
Multi-Modal Memory Collaboration (MMC) — adapted from CLIMB-ReID (AAAI-2025) for 3-modal ReID.

Original single-modal design:
  - One ClusterMemoryAMP with mean+hard proxy copies
  - Intra-class momentum update in backward
  - Contrastive loss: CE(mean) + CE(hard)

Multi-modal extension:
  - Three independent memory banks (RGB, NI, TI)
  - Intra-modal update: each modality updates its own bank (unchanged logic)
  - Cross-modal interaction: each modality queries other modalities' banks (loss only, no update)
  - Total = intra_loss (3 terms) + cross_loss (6 terms)
"""

import collections
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda import amp


# ==============================================================================
# Autograd Function: Forward = matmul, Backward = momentum update (mean + hard)
# ==============================================================================

class CM_Mix_mean_hard(autograd.Function):
    """
    Custom autograd function for cluster memory.
    Forward:  inputs @ features.T  →  similarity logits
    Backward: momentum-update features with mean + hard sample per class.

    features layout: first half → updated by mean, second half → updated by hardest sample.
    """

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        nums = len(ctx.features) // 2          # half: mean proxies, half: hard proxies
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # ---- step 1: all-sample momentum update (first half: mean proxies) ----
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        # ---- step 2: per-class mean→first-half, hardest→second-half ----
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            # mean update → first half
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.momentum * ctx.features[index] + (1. - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

            # hard update → second half
            hard = np.argmin(np.array(distances))   # argmin cosine-sim = hardest
            ctx.features[index + nums] = ctx.momentum * ctx.features[index + nums] + (1. - ctx.momentum) * features[hard]
            ctx.features[index + nums] /= ctx.features[index + nums].norm()

        return grad_inputs, None, None, None


def cm_mix(inputs, indexes, features, momentum=0.5):
    return CM_Mix_mean_hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


# ==============================================================================
# Single-Modal Cluster Memory (unchanged logic from CLIMB-ReID)
# ==============================================================================

class ClusterMemoryAMP(nn.Module, ABC):
    """
    Single-modal cluster memory with mean+hard proxy collaboration.

    Args:
        temp:     temperature for contrastive loss
        momentum: momentum factor for feature update
    """

    def __init__(self, temp=0.05, momentum=0.2):
        super(ClusterMemoryAMP, self).__init__()
        self.momentum = momentum
        self.temp = temp
        self.features = None   # shape: [2 * num_classes, feat_dim]

    def forward(self, inputs, targets, update_memory=True):
        """
        Args:
            inputs:        [B, feat_dim]  L2-normalised features
            targets:       [B]            ground-truth IDs
            update_memory: if True, use cm_mix (forward + backward momentum update);
                           if False, only compute logits (for cross-modal query).
        Returns:
            scalar loss
        """
        inputs = F.normalize(inputs, dim=1)

        if update_memory:
            outputs = cm_mix(inputs, targets, self.features, self.momentum)
        else:
            # cross-modal query: compute similarity WITHOUT updating memory
            outputs = inputs.mm(self.features.t())

        outputs = outputs / self.temp

        mean, hard = torch.chunk(outputs, 2, dim=1)
        loss = 0.5 * (F.cross_entropy(hard, targets) + F.cross_entropy(mean, targets))
        return loss


# ==============================================================================
# Multi-Modal Memory Collaboration (NEW)
# ==============================================================================

class MultiModalClusterMemory(nn.Module):
    """
    Three-modal memory bank for RGB + NI + TI.

    - Intra-modal update:  each modality's features update only its own memory bank
    - Cross-modal loss:    each modality's features also query the other two banks
                           (loss only, no memory update)

    Total loss = (L_intra_rgb + L_intra_ni + L_intra_ti) / 3
               + (L_cross_rgb→ni + L_cross_rgb→ti + ... ) / 6
    """

    def __init__(self, num_classes, feat_dim, temp=0.05, momentum=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.memory_rgb = ClusterMemoryAMP(temp=temp, momentum=momentum)
        self.memory_ni  = ClusterMemoryAMP(temp=temp, momentum=momentum)
        self.memory_ti  = ClusterMemoryAMP(temp=temp, momentum=momentum)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def set_features(self, rgb_features, ni_features, ti_features, labels, device='cuda'):
        """
        Initialise the three memory banks from per-modality features.

        Args:
            rgb_features: [N, feat_dim]
            ni_features:  [N, feat_dim]
            ti_features:  [N, feat_dim]
            labels:       [N]
        """
        self.memory_rgb.features = compute_cluster_centroids(rgb_features, labels).to(device)
        self.memory_ni.features  = compute_cluster_centroids(ni_features, labels).to(device)
        self.memory_ti.features  = compute_cluster_centroids(ti_features, labels).to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, rgb_feat, ni_feat, ti_feat, targets):
        """
        Compute multi-modal memory collaboration loss.

        Args:
            rgb_feat: [B, feat_dim]  RGB global features
            ni_feat:  [B, feat_dim]  NI  global features
            ti_feat:  [B, feat_dim]  TI  global features
            targets:  [B]            identity labels

        Returns:
            (loss_intra, loss_cross, loss_total)
        """
        # ---- intra-modal (update own memory) ----
        loss_intra_rgb = self.memory_rgb(rgb_feat, targets, update_memory=True)
        loss_intra_ni  = self.memory_ni(ni_feat,   targets, update_memory=True)
        loss_intra_ti  = self.memory_ti(ti_feat,   targets, update_memory=True)
        loss_intra = (loss_intra_rgb + loss_intra_ni + loss_intra_ti) / 3.0

        # ---- cross-modal (query other memories, no update) ----
        # RGB queries NI, TI memories
        loss_cross_rgb_ni = self.memory_ni(rgb_feat, targets, update_memory=False)
        loss_cross_rgb_ti = self.memory_ti(rgb_feat, targets, update_memory=False)
        # NI  queries RGB, TI memories
        loss_cross_ni_rgb = self.memory_rgb(ni_feat,  targets, update_memory=False)
        loss_cross_ni_ti  = self.memory_ti(ni_feat,  targets, update_memory=False)
        # TI  queries RGB, NI memories
        loss_cross_ti_rgb = self.memory_rgb(ti_feat,  targets, update_memory=False)
        loss_cross_ti_ni  = self.memory_ni(ti_feat,  targets, update_memory=False)

        loss_cross = (loss_cross_rgb_ni + loss_cross_rgb_ti +
                      loss_cross_ni_rgb + loss_cross_ni_ti +
                      loss_cross_ti_rgb + loss_cross_ti_ni) / 6.0

        loss_total = loss_intra + loss_cross
        return loss_intra, loss_cross, loss_total


# ==============================================================================
# Helper: compute L2-normalised cluster centroids (with mean+hard duplication)
# ==============================================================================

def compute_cluster_centroids(features, labels):
    """
    Compute L2-normalised cluster centroid for each class.
    Returns centroids repeated 2× for mean+hard proxy structure.

    Args:
        features: [N, D]
        labels:   [N]

    Returns:
        centers: [2 * num_classes, D]
    """
    num_classes = len(labels.unique()) - 1 if -1 in labels else len(labels.unique())
    centers = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32, device=features.device)
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        temp = features[idx, :]
        if len(temp.shape) == 1:
            temp = temp.reshape(1, -1)
        centers[i, :] = temp.mean(0)
    return F.normalize(centers.repeat(2, 1), dim=1)
