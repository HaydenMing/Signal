# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: Always respond in Chinese (Simplified Chinese / 简体中文). All explanations, code comments, commit messages, and interactions must be in Chinese.**

## Project Overview

**Signal** — multi-modal object ReID framework (AAAI-2026). Works with RGB + Near Infrared (NI) + Thermal Infrared (TI) images. Based on the IDEA research group's multi-modal person/vehicle ReID pipeline.

Key papers: `2511.17965v1.pdf` (Signal), `01540-YuC.pdf` (CLIMB-ReID memory method).

## Environment

- Python 3.10+ (conda), CUDA 11.8
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` then `pip install -r requirements.txt` (remove the `grad-cam` line and outdated torch version pins first)
- `pip install safetensors` for DINOv3 HF-format weights

## How to Run

**Training** (from Signal/ directory):
```bash
python train.py --config_file configs/RGBNT201/Signal.yml
```

CLI overrides YAML via `KEY VALUE` pairs (space-separated, NO `=` sign):
```bash
python train.py --config_file configs/RGBNT201/Signal.yml \
    MODEL.TRANSFORMER_TYPE dinov3_vitb16 \
    MODEL.DINOV3_PRETRAIN_PATH /path/to/weights \
    MODEL.USE_MMC True \
    SOLVER.IMS_PER_BATCH 32 SOLVER.MAX_EPOCHS 30
```

**Testing**: edit `test.py` line 51 to point to trained `.pth`, then:
```bash
python test.py --config_file configs/RGBNT201/Signal.yml
```

**Multi-GPU**: `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py ... MODEL.DIST_TRAIN True`

## Dataset Layout

Each dataset (RGBNT201/RGBNT100/MSVR310) is a stitched 768×128 image (RGB|NI|TI side-by-side). Code auto-crops: `img.crop((0,0,256,128))` for RGB, then `(256,0,512,128)` for NI, `(512,0,768,128)` for TI.

Expected directory tree (under `ROOT_DIR`):
```
512/data/RGBNT201/
  train_171/{RGB,NI,TI}/*.jpg
  test/{RGB,NI,TI}/*.jpg
```

Dataset class in `data/datasets/<name>.py` defines `dataset_dir`. `ROOT_DIR` + `dataset_dir` = full path.

## Architecture

```
Signal model (modeling/make_model.py)
├── clip_vision_encoder (modeling/meta_arch.py → build_transformer)
│   ├── CLIP ViT-B-16 (feat_dim=512, PROJECTED)
│   ├── DINOv3 ViT-B/16 (feat_dim=768, NO projection) ← OUR ADDITION
│   └── ImageNet ViT (feat_dim=768)
│   └── Returns: (patch_tokens, cls_token) per modality
│
├── SIM (modeling/AddModule/useA.py) — Selective Interaction Module
│   ├── TokenSelection: intra-modal + inter-modal token selection
│   └── ModalInteractive: cross-attention fusion of selected tokens
│
├── AlignM (modeling/AddModule/useB.py) — Global + Local Alignment
│   ├── Cls_Align (GAM): Gram matrix volume minimization (utils/volume.py)
│   └── patch_Align (LAM): deformable sampling alignment (AddModule/DAS.py)
│
└── MultiModalMemory (layers/multimodal_memory.py) — OUR ADDITION
    ├── 3× ClusterMemoryAMP (RGB/NI/TI) with mean+hard proxies
    ├── Intra-modal: momentum update within each modality
    └── Cross-modal: query other modalities' banks (loss only, no update)
```

## Key Config Switches

| Parameter | Effect |
|-----------|--------|
| `MODEL.TRANSFORMER_TYPE` | `ViT-B-16` (CLIP, feat=512), `dinov3_vitb16` (DINOv3, feat=768), `vit_base_patch16_224` (ImageNet) |
| `MODEL.DINOV3_PRETRAIN_PATH` | Local DINOv3 weights (dir with safetensors, or .pth file) |
| `MODEL.USE_A` | Enable SIM |
| `MODEL.USE_B` | Enable GAM+LAM |
| `MODEL.stageName` | `CLS` = GAM only; `together_CLS_Patch` = GAM+LAM |
| `MODEL.USE_MMC` | Enable multi-modal memory collaboration |
| `MODEL.MMC_LOSS_WEIGHT` | Memory loss weight (default 1.0) |
| `MODEL.MEMORY_MOMENTUM` | Memory update momentum (default 0.2) |
| `MODEL.DIRECT` | `1` = concat 3-modality features → single classifier; `0` = separate classifiers |
| `SOLVER.IMS_PER_BATCH` | Total batch size |

## Key Files (non-obvious)

| File | Role |
|------|------|
| `modeling/meta_arch.py` | Backbone factory — chooses CLIP/DINOv3/ImageNet ViT based on `TRANSFORMER_TYPE` |
| `modeling/dinov3_encoder.py` | DINOv3 wrapper, HF→FB key remapping for safetensors |
| `modeling/make_model_clipreid.py` | CLIP-specific loader, `load_clip_to_cpu()`, ViT-B-16.pt path |
| `layers/multimodal_memory.py` | `CM_Mix_mean_hard` autograd fn + `ClusterMemoryAMP` + `MultiModalClusterMemory` |
| `layers/memory_utils.py` | `extract_multimodal_features()` for per-epoch memory init |
| `layers/make_loss.py` | Loss function factory (ID + triplet + center) |
| `engine/processor.py` | `do_train()` — training loop with memory bank init per epoch |
| `data/datasets/make_dataloader.py` | `train_collate_fn` / `val_collate_fn` — how batch dicts are built |
| `data/datasets/bases.py` | `read_image()` — how stitched images are cropped into 3 modalities |
| `data/datasets/RGBNT201.py` | Dataset class, `dataset_dir = '512/data/RGBNT201'` |
| `solver/make_optimizer.py` | Learning rate rules (CLIP backbone gets `0.000005`, DINOv3/fc get custom) |
| `utils/metrics.py` | `R1_mAP_eval` — main evaluation metric; needs `from scipy.integrate import simpson as simps` for scipy≥1.14 |
| `utils/volume.py` | Gram matrix determinant for GAM alignment loss |

## Common Issues

- **scipy `simps` ImportError**: scipy≥1.14 renamed it. Fix: `sed -i 's/from scipy.integrate import simps/from scipy.integrate import simpson as simps/' utils/metrics.py`
- **Hardcoded paths**: `make_model_clipreid.py:175` (ViT-B-16.pt), `test.py:51` (trained model)
- **`dataloader.NUM_WORKERS`**: Set to 0 on Windows, 8-12 on Linux
- **Memory bank init**: Each epoch extracts features from the ENTIRE training set via `train_loader_normal`. This takes ~1-2 min per epoch. On small datasets this is fine; on large ones consider caching.
