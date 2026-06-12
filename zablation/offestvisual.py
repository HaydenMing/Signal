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


class FunctionHookManager:

    def __init__(self):
        self.hooks = []
        self.original_function = None
        self.wrapped_function = None
        self.is_wrapped = False

    def register_hook(self, target_function, hook):
        if not self.is_wrapped:
            self.original_function = target_function
            self.hooks.append(hook)

            def wrapped(*args, **kwargs):
                output = self.original_function(*args, **kwargs)

                for hook_fn in self.hooks:
                    hook_fn(None, args, output)  

                return output

            self.wrapped_function = wrapped
            self.is_wrapped = True
            return wrapped
        else:
            self.hooks.append(hook)
            return self.wrapped_function

    def remove_hooks(self):
        self.hooks = []
        self.is_wrapped = False
        return self.original_function


class PointsVisualizer:
    def __init__(self, model, target_function):
        self.model = model
        self.target_function = target_function
        self.feats = None
        self.hook_manager = FunctionHookManager()

        self.wrapped_function = self.hook_manager.register_hook(
            target_function,
            self._save_feats
        )

    def _save_feats(self, module, input, output):
        print("进入保存特征")
        if isinstance(output, tuple) and len(output) >= 10:
            # 假设输出是(rgb_mask, ni_mask, ti_mask)
            self.feats = {
                'RGB_ref': output[1].detach(),
                'RGB_pos': output[2].detach(),
                'NIR_ref': output[3].detach(),
                'NIR_pos': output[4].detach(),
                'TIR_ref': output[5].detach(),
                'TIR_pos': output[6].detach(),
                'RGB_patch': output[7].detach(),
                'NI_patch': output[8].detach(),
                'TI_patch': output[9].detach()

            }

    def visualize_sampling_with_offset(self, feature_maps, reference_pointss, sampled_pointss, img_paths, pattern=0,
                                       writer=None,
                                       epoch=0, title='Sampling Points with Offset', patch_size=(16, 16)):
        
        modality = ['RGB', 'NI', 'TI']
        if pattern == 0:
            prefix = '/media/zpp2/Datamy/lyy/512/data/RGBNT201/test/RGB/'
        elif pattern == 1:
            prefix = '/media/zpp2/Datamy/lyy/512/data/RGBNT201/test/NI/'
        elif pattern == 2:
            prefix = '/media/zpp2/Datamy/lyy/512/data/RGBNT201/test/TI/'
        
        img_path = prefix + img_paths

        
        original_image = Image.open(img_path)
        original_image = np.array(original_image)

        
        feature_map = torch.mean(feature_maps, dim=0, keepdim=True)
        feature_map = feature_map.detach().cpu().numpy()
        feature_map = (feature_map - np.min(feature_map)) / np.ptp(feature_map)

        
        sampled_points = sampled_pointss.detach().cpu().numpy()
        reference_points = reference_pointss.detach().cpu().numpy()

        reference_points = reference_points.reshape(-1, 2)  
        sampled_points = sampled_points.reshape(-1, 2)     
        H_feat, W_feat = feature_map.shape[1:]
        H_orig, W_orig = original_image.shape[:2]


        scale_x = W_orig / W_feat
        scale_y = H_orig / H_feat
        

        sampled_points[:, 1] = (sampled_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
        sampled_points[:, 0] = (sampled_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y
        reference_points[:, 1] = (reference_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
        reference_points[:, 0] = (reference_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(original_image)
        ax.set_title(title)

        for ref, samp in zip(reference_points, sampled_points):
            ref_y, ref_x = ref
            samp_y, samp_x = samp

            ax.scatter(ref_x, ref_y, c='black', s=70, marker='o', edgecolor='black', linewidth=4)  
            ax.scatter(samp_x, samp_y, c='orange', s=70, marker='x', linewidth=4)  

            ax.arrow(ref_x, ref_y, samp_x - ref_x, samp_y - ref_y, color='limegreen', alpha=0.7,
                    head_width=3, head_length=4, linewidth=4, length_includes_head=True)

        patch_height, patch_width = patch_size
        for y in range(0, H_orig, patch_height):
            ax.plot([0, W_orig], [y, y], color='white', linewidth=1.5, linestyle='--')  
        for x in range(0, W_orig, patch_width):
            ax.plot([x, x], [0, H_orig], color='white', linewidth=1.5, linestyle='--')  

        ax.set_xlim(-1, W_orig)
        ax.set_ylim(H_orig, -1)  

        if writer is not None:
            writer.add_figure(f"{title}", fig, global_step=epoch)
        
        output_dir = f'zoff_vis/{modality[pattern]}'
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(
            f'zoff_vis/{modality[pattern]}/{img_path.split("/")[-1].split(".")[0]}.png')
        # plt.show()
        plt.close(fig)

    def __del__(self):
        
        original_function = self.hook_manager.remove_hooks()
        
        for name, obj in vars(self.model.AlignmentM).items():
            if obj is self.wrapped_function:
                setattr(self.model.AlignmentM, name, original_function)
                break


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
    parser.add_argument("--pts_path", default="your_path/Signal_50.pth",
                        help="Path to pth file", type=str)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.pts_path)
    model.eval()
    model.to(device)

    target_function = model.AlignM.patch_Align 

    visualizer = PointsVisualizer(model, target_function)
    wrapped_function = visualizer.wrapped_function

    setattr(model.AlignM, 'patch_Align', wrapped_function)


    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = Newdict({
            'RGB': img['RGB'].to(device),
            'NI': img['NI'].to(device),
            'TI': img['TI'].to(device),
            'cam_label': camids.to(device),
            'view_label': target_view.to(device)
        })
        
        
        with torch.no_grad():
            _ = model(img)

        
        for i in range(img['RGB'].size(0)): 
            for modality in ['RGB_ref', 'RGB_pos', 'NIR_ref', 'NIR_pos', 'TIR_ref', 'TIR_pos']:
                
                visualizer.visualize_sampling_with_offset( visualizer.feats['RGB_patch'][i],  visualizer.feats['RGB_ref'][i],  visualizer.feats['RGB_pos'][i], imgpath[i],0)
                visualizer.visualize_sampling_with_offset( visualizer.feats['NI_patch'][i],  visualizer.feats['NIR_ref'][i],  visualizer.feats['NIR_pos'][i], imgpath[i],1)
                visualizer.visualize_sampling_with_offset( visualizer.feats['TI_patch'][i],  visualizer.feats['TIR_ref'][i],  visualizer.feats['TIR_pos'][i], imgpath[i],2)

        
        if n_iter == 0:
            break

    print("Mask visualization completed!")
    print(f"Results saved in: zoffset_vis/{cfg.DATASETS.NAMES}/")