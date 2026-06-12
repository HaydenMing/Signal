
import torch.nn as nn
import einops
import torch.nn.functional as F
import torch


class DA_sample(nn.Module):

    def __init__(
            self, n_heads, n_head_channels, n_groups,
            stride,offset_range_factor, ksize
    ):

        super().__init__()
        self.n_head_channels = n_head_channels 
        self.nc = n_head_channels * n_heads 
        self.n_groups = n_groups 
        self.n_group_channels = self.nc // self.n_groups 
        self. kk = ksize 
        self.stride = stride 
        self.offset_range_factor = offset_range_factor
        pad_size = 0
        
        
 
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, pad_size,
                        groups=self.n_group_channels),
            nn.GELU(),
            
            nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
        )
         
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )


    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref


  


    def forward(self,x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        
        Hk, Wk = offset.size(2), offset.size(3)
   
        if self.offset_range_factor > 0:

            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        reference = self._get_ref_points(Hk, Wk, B,dtype, device)
        pos = (offset + reference).clamp(-1., +1.)

        x_sampled = F.grid_sample(
        input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
        grid=pos[..., (1, 0)],  # y, x -> x, y
        mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        return x_sampled

                
