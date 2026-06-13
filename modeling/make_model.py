
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
import torch.nn.functional as F


from modeling.AddModule.useA import Select_Interactive_Module
from modeling.AddModule.useB import AlignmentM
    
class Signal(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(Signal, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        elif 'dinov3' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.direct = cfg.MODEL.DIRECT
        self.camera = camera_num
        self.view = view_num
        self.use_A = cfg.MODEL.USE_A  
        self.use_B = cfg.MODEL.USE_B  

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.h, self.w = self.image_size[0]//16,self.image_size[1]//16
      
       
        self.clip_vision_encoder = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        if self.direct:
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
           

           
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

        
       
        if self.use_A:
            num = 3
            self.SIM = Select_Interactive_Module(self.feat_dim,k=int(cfg.MODEL.TOPK))
            self.classifier_var = nn.Linear(num*self.feat_dim, self.num_classes, bias=False)
            self.classifier_var.apply(weights_init_classifier)

            self.bottleneck_var = nn.BatchNorm1d(num*self.feat_dim)
            self.bottleneck_var.bias.requires_grad_(False)
            self.bottleneck_var.apply(weights_init_kaiming)

        if self.use_B:
            
            self.AlignM = AlignmentM(self.feat_dim,self.h, self.w)
            

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=1, training=True, sge="CLS",
                get_multimodal_feat=False):
        """
        Args:
            get_multimodal_feat: if True in inference mode, return (RGB_global, NI_global, TI_global)
                                 instead of the usual combined feature.
        """
        if training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            rgb_patch, RGB_global = self.clip_vision_encoder(RGB, cam_label=cam_label, view_label=view_label)
            ni_patch,  NI_global  = self.clip_vision_encoder(NI,  cam_label=cam_label, view_label=view_label)
            ti_patch,  TI_global  = self.clip_vision_encoder(TI,  cam_label=cam_label, view_label=view_label)

            # cache per-modality features for external access (memory bank)
            self.rgb_global_feat = RGB_global
            self.ni_global_feat  = NI_global
            self.ti_global_feat  = TI_global



            if self.use_A:
                vars_total  = self.SIM(rgb_patch,ni_patch,ti_patch,RGB_global, NI_global, TI_global)
                vars_global = self.bottleneck_var(vars_total)
                vars_score = self.classifier_var(vars_global) 


            if self.use_B:
                if sge == "CLS":
                    loss_area = self.AlignM(rgb_patch,ni_patch,ti_patch ,stage="CLS")

                else:
                    loss_area,patch_loss = self.AlignM(rgb_patch,ni_patch,ti_patch,stage="together_CLS_Patch")

                
                   

            if self.direct:
                
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
                    
                    
            else:
                
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))
              


            if self.direct:
                if not self.use_A:
                    sign = 1
                    return sign,ori_score, ori
            
                elif self.use_A and self.use_B == False:
                    
                    sign = 2
                    return sign,ori_score, ori,vars_score,vars_total
                
                elif self.use_A and self.use_B:
                    sign = 3

                    if sge == "CLS":
                        return sign, ori_score, ori,vars_score,vars_total,loss_area   
                    else:
                        return sign, ori_score, ori,vars_score,vars_total,loss_area,patch_loss  
                    
                    

                   
            else:
                if not self.use_A:
                    
                    sign = 1
                    return sign,RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global
            
                elif self.use_A and self.use_B == False:

                    
                    sign = 2
                    return sign,RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global,vars_score,vars_total
                     
                
                elif self.use_A and self.use_B:
                   
                    sign = 3
                    if sge == "CLS":
                        return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global,vars_score,vars_total,loss_area  
                    else:
                        return sign, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global,vars_score,vars_total,loss_area,patch_loss
                    


        else:

            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            if 'cam_label' in x:
                cam_label = x['cam_label']

            rgb_patch,RGB_global = self.clip_vision_encoder(RGB, cam_label=cam_label, view_label=view_label)
            ni_patch,NI_global = self.clip_vision_encoder(NI, cam_label=cam_label, view_label=view_label)
            ti_patch,TI_global = self.clip_vision_encoder(TI, cam_label=cam_label, view_label=view_label)

            # cache per-modality features
            self.rgb_global_feat = RGB_global
            self.ni_global_feat  = NI_global
            self.ti_global_feat  = TI_global

            if get_multimodal_feat:
                return RGB_global, NI_global, TI_global

            if self.use_A:
                vars_total = self.SIM(rgb_patch,ni_patch,ti_patch,RGB_global, NI_global, TI_global)
                

            if self.use_B:
                if sge == "CLS":
                    _ = self.AlignM(rgb_patch,ni_patch,ti_patch,stage="CLS")

                else:
                    _,_ = self.AlignM(rgb_patch,ni_patch,ti_patch,stage="together_CLS_Patch")
                


            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            
            if not self.use_A:
                return ori
            else:
                return torch.cat([ori,vars_total],dim=-1)
            

    
        
__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}





def make_frame(cfg, num_class, camera_num, view_num=0):
    model = Signal(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building Signal===========')
    return model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    """before and after attention,use it"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



    