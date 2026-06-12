import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.volume import volume_computation3
from modeling.AddModule.DAS import DA_sample as DAS
import math

# class Contra_head(nn.Module):
#     def __init__(self, input_dim, contra_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, contra_dim, bias=False)
#     def forward(self, cls_token):
#         return self.linear(cls_token)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class AlignmentM(nn.Module):
    def __init__(self, feat_dim,H,W):
        super(AlignmentM, self).__init__()
        self.feat_dim = feat_dim
        # cls_align 
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        # self.rgb_linear = Contra_head(feat_dim,feat_dim)
        # self.ni_linear = Contra_head(feat_dim,feat_dim)
        # self.ti_linear = Contra_head(feat_dim,feat_dim)

        # pat_align
        self.h,self.w = H,W
        self.mse = nn.MSELoss()
        n_heads = 1  
        n_head_channels = 512 
        n_groups = 1 
        stride = 4  
        offset_range_factor = 2  
        ksize = 4  


        self.DAS_r = DAS(
                n_heads, n_head_channels, n_groups,
                stride,offset_range_factor, ksize)
        
        self.DAS_n = DAS(
                n_heads, n_head_channels, n_groups,
                stride,offset_range_factor, ksize)
        
        self.DAS_t = DAS(
                n_heads, n_head_channels, n_groups,
                stride,offset_range_factor, ksize)
        
        


    def Cls_Align(self,RGB_patch,NI_patch,TI_patch):
    
        rgb_feature = torch.mean(RGB_patch, dim=1)
        ni_feature = torch.mean(NI_patch, dim=1)
        ti_feature = torch.mean(TI_patch, dim=1)
        
        # rgb_feature = self.rgb_linear(rgb_feature)
        # ni_feature = self.ni_linear(ni_feature)
        # ti_feature = self.ti_linear(ti_feature)


        feat_ro = F.normalize(rgb_feature,dim=-1)
        feat_no = F.normalize(ni_feature,dim=-1)
        feat_to = F.normalize(ti_feature,dim=-1)
        

        V = volume_computation3(feat_ro,feat_no,feat_to)
        volume = V / self.contra_temp

        VT = volume_computation3(feat_ro,feat_no,feat_to).T
        volumeT = VT / self.contra_temp

        bs = feat_ro.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(feat_ro.device)
        loss_area = (
                    F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                    + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
            ) / 2
        
        return loss_area
    
    def patch_Align(self,RGB_patch,NI_patch,TI_patch):
        bs,n,dim = RGB_patch.shape
        # reshape  [bs,dim,16,8]
        RGB_patch = RGB_patch.reshape(bs,self.h,self.w,-1).permute(0,3,1,2)
        NI_patch = NI_patch.reshape(bs,self.h,self.w,-1).permute(0,3,1,2)
        TI_patch = TI_patch.reshape(bs,self.h,self.w,-1).permute(0,3,1,2)

        rgb_sampled = self.DAS_r(RGB_patch)
        nir_sampled  = self.DAS_n(NI_patch)
        tir_sampled = self.DAS_t(TI_patch)

        loss1 = self.mse(nir_sampled,rgb_sampled)
        loss2 = self.mse(tir_sampled,rgb_sampled)
        loss3 = self.mse(tir_sampled,nir_sampled)
        
               

        pat_loss = ( loss1 + loss2 + loss3 ) /3

           
        return pat_loss



    def forward(self,RGB_patch,NI_patch,TI_patch,stage):
        if stage == "CLS":
            cls_loss = self.Cls_Align(RGB_patch,NI_patch,TI_patch)
            return cls_loss
        
        loss_area  = self.Cls_Align(RGB_patch,NI_patch,TI_patch)
        patch_loss = self.patch_Align(RGB_patch,NI_patch,TI_patch) 

        return loss_area,patch_loss

