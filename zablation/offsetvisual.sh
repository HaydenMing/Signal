1.  DAS 修改 
    return ,reference,pos

2. make_model.py

sge="together_CLS_Patch"
    cam_label = x['cam_label']
    view_label = x['view_label']

3. useB.py
 rgb_sampled,reference1,pos1= self.DAS_r(RGB_patch)
        nir_sampled,reference2,pos2 = self.DAS_n(NI_patch)
        tir_sampled,reference3,pos3= self.DAS_t(TI_patch)

        loss1 = self.mse(nir_sampled,rgb_sampled)
        loss2 = self.mse(tir_sampled,rgb_sampled)
        loss3 = self.mse(tir_sampled,nir_sampled)
        
               

        pat_loss = ( loss1 + loss2 + loss3 ) /3
        print("patch alignment finish")
           
        return pat_loss,reference1,pos1,reference2,pos2,reference3,pos3,RGB_patch,NI_patch,TI_patch



    def forward(self,RGB_patch,NI_patch,TI_patch,stage):
        if stage == "CLS":
            cls_loss = self.Cls_Align(RGB_patch,NI_patch,TI_patch)
            return cls_loss
        
        loss_area  = self.Cls_Align(RGB_patch,NI_patch,TI_patch)
        patch_loss,reference1,pos1,reference2,pos2,reference3,pos3,RGB_patch,NI_patch,TI_patch = self.patch_Align(RGB_patch,NI_patch,TI_patch) 

        return loss_area,patch_loss


conda activate envs
python offsetvisual.py