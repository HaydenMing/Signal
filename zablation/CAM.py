import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from config import cfg
from data import make_dataloader
from modeling import make_frame
from pytorch_grad_cam.utils.image import show_cam_on_image

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

def reshape_transform(tensor, height=16, width=8):
    #result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

class SimpleCAM:
    def __init__(self, model, target_layer, modality_index, total_modalities=3):
        self.model = model
        self.target_layer = target_layer
        self.modality_index = modality_index  # 0: RGB, 1: NI, 2: TI
        self.total_modalities = total_modalities
        self.feature_maps = None
        
        self.hook = target_layer.register_forward_hook(self._save_feature_maps)
        
    # def _save_feature_maps(self, module, input, output):
    #     self.feature_maps = output.detach()  # [bs,n,dim]

    def _save_feature_maps(self, module, input, output):
        if self.modality_index == 0:
            self.feature_maps = output[0].detach()  # [bs,128,dim]
            print(self.feature_maps.shape)
        elif self.modality_index == 1:
            self.feature_maps = output[1].detach()
        else:
            self.feature_maps = output[2].detach()
        
    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)[1]  # [batch_size, num_classes]
            
        if class_idx is None:
            _, class_idx = torch.max(outputs, dim=1)
            
        self.feature_maps = reshape_transform(self.feature_maps) 
        batch_size, num_channels, h, w = self.feature_maps.size()
        cams = []
        
        if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'weight'):
            class_weights = self.model.classifier.weight  # [num_classes, 1536]
        else:
            raise ValueError("none")
        
        modality_channels = class_weights.size(1) // self.total_modalities  # 1536 / 3 = 512
        start_idx = self.modality_index * modality_channels
        end_idx = start_idx + modality_channels
        
        for i in range(batch_size):
            target_class = class_idx[i].item()
            
            class_weight = class_weights[target_class, start_idx:end_idx].unsqueeze(1)  # [512, 1]
            
            feature_map = self.feature_maps[i].view(num_channels, h * w)  # [num_channels, h*w]  [512,128]
            cam = torch.matmul(class_weight.t(), feature_map)  # [1,512]*[512,128] = [1, h*w]
            cam = cam.view(h, w)  # [h, w]
            
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)

            cam = cam.detach().cpu().numpy()
            
            cams.append(cam)
            
        return cams
    
    def __del__(self):
        self.hook.remove()




def show_cam(index, imgpath, grayscale_cam, modality, cfg):

    index = int(index)
    img_path = imgpath[index] # '000288_cam1_0_09.jpg'
    
    base_filename, file_extension = os.path.splitext(img_path)

    # Load image based on dataset and modality
    if cfg.DATASETS.NAMES == 'RGBNT201':
        img_path = f'/media/zpp2/Datamy/lyy/512/data/RGBNT201/test/{modality}/{img_path}'
    elif cfg.DATASETS.NAMES == 'RGBNT100':
        img_path = f'../RGBNT100/rgbir/query/{img_path}'

    grayscale_cam = grayscale_cam[index]
    if cfg.DATASETS.NAMES == 'RGBNT100':
        img = Image.open(img_path).convert('RGB')
        if modality == "RGB":
            cropped_image = img.crop((0, 0, 256, 128))
        elif modality == "NI":
            cropped_image = img.crop((256, 0, 512, 128))
        else:
            cropped_image = img.crop((512, 0, 768, 128))
        rgb_image = np.float32(cropped_image) / 255
    else:
        img = cv2.imread(img_path, 1)
        rgb_image = cv2.resize(img, (128, 256))
        rgb_image = np.float32(rgb_image) / 255

    grayscale_cam = cv2.resize(grayscale_cam, (rgb_image.shape[1], rgb_image.shape[0]))
    
    visualization = show_cam_on_image(rgb_image, grayscale_cam)
    output_dir = f'zcam_vis/base+SIM+cls+pat/{cfg.DATASETS.NAMES}/{modality}'  
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{base_filename}.jpg'
    cv2.imwrite(save_path, visualization)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Testing")
    parser.add_argument("--config_file", default="configs/RGBNT201/Signal.yml", help="Path to config file", type=str)
    parser.add_argument("opts", help="Modify config options via command line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--pts_path", default="your_path/Signal_50.pth", help="Path to pth file", type=str)


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

    target_layer = model.SIM.token_selection

    cam_rgb = SimpleCAM(model=model, target_layer=target_layer, modality_index=0)
    cam_ni = SimpleCAM(model=model, target_layer=target_layer, modality_index=1)
    cam_ti = SimpleCAM(model=model, target_layer=target_layer, modality_index=2)
    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = Newdict({'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device),
                       'cam_label': camids.to(device),
                       'view_label': target_view.to(device)})

        modalities = {
            "RGB": cam_rgb,
            "NI": cam_ni,
            "TI": cam_ti
        }
        
        for modality, cam in modalities.items():
            print(f"Generating CAM for {modality} modality...")
            grayscale_cam = cam(input_tensor=img)
            
            print(f"Processing {modality} Images -> ",img['RGB'].size(0))
            for i in range(img['RGB'].size(0)): 
                show_cam(i, imgpath, grayscale_cam, modality, cfg)

        if n_iter == 0:
            break
    
    print("CAM generation completed for all modalities!")
    print("Results saved in: zcam_vis/")

