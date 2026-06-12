import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import cfg
from data import make_dataloader
from modeling import make_frame

class Newdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = [1, 2, 3, 4]

    def to(self, device):
        for key in self.keys():
            self[key] = self[key].to(device)
        return self

    def size(self, k):
        data = self['RGB']
        width, height = data.size(-1), data.size(-2)
        return width if k == -1 else height

class MaskVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.masks = None
        self.hook = target_layer.register_forward_hook(self._save_masks)
        
    def _save_masks(self, module, input, output):
        if isinstance(output, tuple) and len(output) == 6:
            # 假设输出是(rgb_mask, ni_mask, ti_mask)
            self.masks = {
                'RGB': output[3].detach(),  # [bs,128,1]
                'NI': output[4].detach(),
                'TI': output[5].detach()
            }
            
        
    
    def visualize_mask_on_image(self, img_path, mask, modality, cfg):

        # 1. 
        if cfg.DATASETS.NAMES == 'RGBNT201':
            full_path = f'/media/zpp2/Datamy/lyy/512/data/RGBNT201/test/{modality}/{img_path}'
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256)) 
        elif cfg.DATASETS.NAMES == 'RGBNT100':
            img = Image.open(f'../RGBNT100/rgbir/query/{img_path}').convert('RGB')
            if modality == "RGB":
                img = img.crop((0, 0, 256, 128))
            elif modality == "NI":
                img = img.crop((256, 0, 512, 128))
            else:
                img = img.crop((512, 0, 768, 128))
            img = np.array(img)
        
        # 2. 
        mask = mask.squeeze(-1).cpu().numpy()  # [128]
        mask_grid = mask.reshape(16, 8)  
        
        # 3. 
        h, w = img.shape[:2]
        patch_h, patch_w = h // 16, w // 8
        visual_mask = np.zeros((h, w), dtype=np.float32)
        
        # 4. 
        for i in range(16):
            for j in range(8):
                y_start, y_end = i * patch_h, (i+1) * patch_h
                x_start, x_end = j * patch_w, (j+1) * patch_w
                visual_mask[y_start:y_end, x_start:x_end] = mask_grid[i, j]
        
        # 5.
        masked_img = img.copy()
        masked_img[visual_mask < 0.5] = 0  
        
        # 6. 
        output_dir = f'zmask_vis/{cfg.DATASETS.NAMES}/{modality}'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(img_path)[0]
        save_path = f'{output_dir}/{base_name}_mask.jpg'
        
        # 
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(masked_img)
        plt.title('Masked Image (white=kept, black=discarded)')
        plt.axis('off')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    def __del__(self):
        self.hook.remove()


'''
1.  pts_path
2. make_model :
            cam_label = x['cam_label']
            view_label = x['view_label']
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Mask Visualization")
    parser.add_argument("--config_file", default="configs/RGBNT201/Signal.yml", help="Path to config file", type=str)
    parser.add_argument("opts", help="Modify config options via command line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--pts_path", default="your_path/Signal_50.pth", help="Path to pth file", type=str)
    
    args = parser.parse_args()
    
    # 
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # 
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"
    
    # 
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.pts_path)
    model.eval()
    model.to(device)
    
    # 
    target_layer = model.SIM.token_selection  
    visualizer = MaskVisualizer(model, target_layer)
    
    # 
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = Newdict({
            'RGB': img['RGB'].to(device),
            'NI': img['NI'].to(device),
            'TI': img['TI'].to(device),
            'cam_label': camids.to(device),
            'view_label': target_view.to(device)
        })
        
        # 
        with torch.no_grad():
            _ = model(img)
        
        # 
        for i in range(img['RGB'].size(0)):  
            for modality in ['RGB', 'NI', 'TI']:
                mask = visualizer.masks[modality][i]  
                
                visualizer.visualize_mask_on_image(imgpath[i], mask, modality, cfg)
        
        
        if n_iter == 0:
            break
    
    print("Mask visualization completed!")
    print(f"Results saved in: mask_vis/{cfg.DATASETS.NAMES}/")