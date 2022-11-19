import torch
from torch import nn
from kornia.filters import filter2d
from einops import reduce, rearrange, repeat
from datasets import ray_utils


def shiftscale_invariant_depthloss(depth, disp):
    """
        Should check the supp why we do this
        disp should change to neg
    Inputs:
        depth: (N) depth in NDC space.
        disp: (N) disparity in Euclidean space, produced by image-based method.
    Outputs:
        loss: (N)
    """
    t_pred = torch.median(depth)
    s_pred = torch.mean(torch.abs(depth-t_pred))
    t_gt = torch.median(-disp)
    s_gt = torch.mean(torch.abs(-disp-t_gt))

    pred_depth_n = (depth-t_pred)/s_pred
    gt_depth_n = (-disp-t_gt)/s_gt
    loss = (pred_depth_n-gt_depth_n)**2
    return loss

class NeRFWLoss(nn.Module):
    """
    If any interest, please refer to my notes for the loss
    There are EIGHT individual loss in the original paper

    # Eq.15 / Notes Loss-5 : render color loss
        col_l   without lambda
    # Eq.11 / Notes Loss-4 : monodepth loss
        disp_l  with lambda      lambda_geo_d
    # Eq.8 /  Notes Loss-1 : temporal pho consistency, forward+backward color loss
        pho_l   without lambda
    # Eq.9 /  Notes Loss-2 : scene flow priors, cycle forward+backward flow loss
        cyc_l   with lambda      cyc_w (=1)
    # Eq.10,Supp Eq.4-7 / Notes Loss-3 : geometric consistency, optical flow guided loss
        flow_fw_l: consistent forward 2D-3D flow loss   with lambda lambda_geo_f
        flow_bw_l: consistent backward 2D-3D flow loss  with lambda lambda_geo_f
    # Supp Eq.1-3 / Notes Loss-6-8 : regularization with lambda_reg (=0.1)
        reg_sp_sm_l: spatial smoothness flow loss
        reg_temp_sm_l: linear flow loss
        reg_min_l: small flow loss
    
    TWO additional loss
    # entropy loss to encourage weights to be concentrated
        entropy_l       lambda_ent
    # encourage static to have different peaks than dynamic
      @thickness specifies how many intervals to separate the peaks
        cross_entropy_l     cross_entropy_w
    """
    def __init__(self,
                 lambda_geo=0.04,   # for geo and depth loss
                 lambda_reg=0.1,    # as paper original
                 thickness=1):       # for cross_entropy_l
        super().__init__()
        self.lambda_geo_d = lambda_geo  
        self.lambda_geo_f = lambda_geo    # half not here
        self.lambda_reg = lambda_reg
        self.lambda_ent = 1e-3
        self.z_far = 0.95
        self.thickness_filter = torch.ones(1, 1, max(thickness, 1))

        # buffered items:
        # self.Ks (1,3,3)           camera intrinsics
        # self.Ps (1,#frames,3,4)   w2c_mats
        # self.max_t    #frames-1   

    def forward(self, inputs, targets, **kwargs):
        """
        Inputs:
            inputs: results from rendering, a dict having everything
            targets:results from batched dataset, a dict having training items
            kwargs: 'epoch' : cur epoch
                    'output_transient_flow': True for training, check the rendering short summary for what it has now
        Outputs:
            ret: a dcit having everything
        """
        ret = {}

        # Eq.15 / Notes Loss-5 : render color loss
        ret['col_l'] = reduce((inputs['rgb_fine']-targets['rgbs'])**2, 'B c -> B', 'mean')  # MSE mean reduc.
        # Eq.11 / Notes Loss-4 : monodepth loss
        ret['disp_l'] = self.lambda_geo_d * shiftscale_invariant_depthloss(inputs['depth_fine'], targets['disps'])


        if kwargs['output_transient_flow']: # this is true for good and [fw,bw,disocc]
        
            # BONUS ENTROPY LOSS----------------------------------------------------------------------
            # entropy loss to encourage weights to be concentrated
            # transient_weights_fine (C,N) dyn weights (transient_alphas * overall_transmittance )
            ret['entropy_l'] = self.lambda_ent * \
                reduce(-inputs['transient_weights_fine']*torch.log(inputs['transient_weights_fine']+1e-8), 'C N -> C', 'sum')
            # dilate transient_weight with @thickness window
            tr_w = inputs['transient_weights_fine'].detach() # (C,N)
            tr_w = rearrange(tr_w, 'C N -> 1 1 C N')
            tr_w = filter2d(tr_w, self.thickness_filter, 'constant') # 0-pad
            tr_w = rearrange(tr_w, '1 1 C N -> C N')
            # cross entropy loss encourage static to have different peaks than dynamic
            # linearly increase the weight from 0 to lambda_ent/5 in 10 epochs
            cross_entropy_w = self.lambda_ent/5 * min(kwargs['epoch']/10, 1.0)
            ret['cross_entropy_l'] = cross_entropy_w * reduce(tr_w*torch.log(inputs['static_weights_fine']+1e-8), 'C N -> C', 'sum')
            # BONUS ENTROPY LOSS----------------------------------------------------------------------

            # targets['cam_ids'] (B)   self.Ks  (1,3,3)   -->  (B,3,3)
            Ks = self.Ks[targets['cam_ids']] # (B,3,3)
            # xyz_fw (C,3)   overall weights*(xyz+fw_flow) weighted next pos
            # xyz_bw (C,3)   overall weights*(xyz+bw_flow) weighted prev pos
            xyz_fw_w = ray_utils.ndc2world(inputs['xyz_fw'], Ks) # (B,3)    weighted fw world position 
            xyz_bw_w = ray_utils.ndc2world(inputs['xyz_bw'], Ks) # (B,3)    weighted bw world position 

            # fw 
            # self.Ps (1, #frames, 3, 4)      w2c_mats
            ts_fw = torch.clamp(targets['ts']+1, max=self.max_t) # (B)
            Ps_fw = self.Ps[targets['cam_ids'], ts_fw] # (B,3,4)  fw camera pos
            # (B,3,3)*(B,3,1)+(B,3,1) = (B,3,1)
            # this could be done by a (4,4)*(4,1), or the way below, gives points loc in camera world
            uvd_fw = Ps_fw[:, :3, :3] @ xyz_fw_w.unsqueeze(-1) + Ps_fw[:, :3, 3:] # (B,3,1)
            # (B,2)/(B,1) = (B,2)  camera projection
            uv_fw = uvd_fw[:, :2, 0] / (torch.abs(uvd_fw[:, 2:, 0])+1e-8)

            # bw
            ts_bw = torch.clamp(targets['ts']-1, min=0)
            Ps_bw = self.Ps[targets['cam_ids'], ts_bw] # (N_rays, 3, 4)
            # (B,3,3)*(B,3,1)+(B,3,1) = (B,3,1)
            # this could be done by a (4,4)*(4,1), or the way below, gives points loc in camera world
            uvd_bw = Ps_bw[:, :3, :3] @ xyz_bw_w.unsqueeze(-1) + Ps_bw[:, :3, 3:]
            # (B,2,1)/(B,1,1) = (B,2,1)  camera projection
            uv_bw = uvd_bw[:, :2, 0] / (torch.abs(uvd_bw[:, 2:, 0])+1e-8)

            # disable geo loss for the first and last frames (no gt for fw/bw)
            # also projected depth must > 0 (must be in front of the camera)
            valid_geo_fw = (uvd_fw[:, 2, 0]>0)&(targets['ts']<self.max_t)   # (B) boolean
            valid_geo_bw = (uvd_bw[:, 2, 0]>0)&(targets['ts']>0)            # (B) boolean
            # Eq.10,Supp Eq.4-7 / Notes Loss-3 : geometric consistency, optical flow guided loss
            # considering all valid
            # (B,2)-(B,2)=(B,2)
            if valid_geo_fw.any():
                ret['flow_fw_l'] = self.lambda_geo_f/2 * torch.abs(uv_fw[valid_geo_fw]-targets['uv_fw'][valid_geo_fw])
                ret['flow_fw_l'] = reduce(ret['flow_fw_l'], 'B c -> B', 'mean')
            if valid_geo_bw.any():
                ret['flow_bw_l'] = self.lambda_geo_f/2 * torch.abs(uv_bw[valid_geo_bw]-targets['uv_bw'][valid_geo_bw])
                ret['flow_bw_l'] = reduce(ret['flow_bw_l'], 'B c -> B', 'mean')

            # pho_w = cyc_w = 1.0
            # Eq.8 /  Notes Loss-1 : temporal pho consistency, forward+backward color loss
            # (B,1)*(B,3) = (B,3)
            ret['pho_l'] = inputs['disocc_fw']*(inputs['rgb_fw']-targets['rgbs'])**2 / inputs['disocc_fw'].mean()
            ret['pho_l']+= inputs['disocc_bw']*(inputs['rgb_bw']-targets['rgbs'])**2 / inputs['disocc_bw'].mean()
            ret['pho_l'] = reduce(ret['pho_l'], 'B c -> B', 'mean')     

            # Eq.9 /  Notes Loss-2 : scene flow priors, cycle forward+backward flow loss
            # (B,N,1)*(B,N,3) = (B,N,3)
            ret['cyc_l'] = inputs['disoccs_fw']*torch.abs(inputs['xyzs_fw_bw']-inputs['xyzs_fine']) / inputs['disoccs_fw'].mean()
            ret['cyc_l']+= inputs['disoccs_bw']*torch.abs(inputs['xyzs_bw_fw']-inputs['xyzs_fine']) / inputs['disoccs_bw'].mean()
            ret['cyc_l'] = reduce(ret['cyc_l'], 'B N c -> B', 'mean')

            # Supp Eq.1-3 / Notes Loss-6-8 : regularization with lambda_reg (=0.1)
            N = inputs['xyzs_fine'].shape[1]
            # original sample places xyz (B,N_cutFar,3) Not weighted!
            xyzs_w = ray_utils.ndc2world(inputs['xyzs_fine'][:, :int(N*self.z_far)], Ks)
            # predicted fw sample places xyz (B,N_cutFar,3) Not weighted!
            xyzs_fw_w = ray_utils.ndc2world(inputs['xyzs_fw'][:, :int(N*self.z_far)], Ks)
            # predicted bw sample places xyz (B,N_cutFar,3) Not weighted!
            xyzs_bw_w = ray_utils.ndc2world(inputs['xyzs_bw'][:, :int(N*self.z_far)], Ks)

            # encourage linear flow loss -> bw + fw == 0
            # as we dont directly get flow, we use points for computation as :
            # bw + fw + 2*cur - 2*cur == 0  -> bw_pts + fw_pts - 2*cur = 0
            ret['reg_temp_sm_l'] = self.lambda_reg * torch.abs(xyzs_fw_w+xyzs_bw_w-2*xyzs_w)
            ret['reg_temp_sm_l'] = reduce(ret['reg_temp_sm_l'], 'B n2 c -> B', 'mean')

            # encourage small flow loss
            ret['reg_min_l'] = self.lambda_reg * (torch.abs(xyzs_fw_w-xyzs_w)+torch.abs(xyzs_bw_w-xyzs_w))
            ret['reg_min_l'] = reduce(ret['reg_min_l'], 'B n2 c -> B', 'mean')

            # encourage spatial smoothness flow (pts along ray should neighborly similar)
            d = torch.norm(xyzs_w[:, 1:]-xyzs_w[:, :-1], dim=-1, keepdim=True)  # (B,N_cutFar-1,1)
            sp_w = torch.exp(-2*d)      # dist part, authentic paper implementation # (B,N_cutFar-1,1)
            sf_fw_w = xyzs_fw_w-xyzs_w  # forward scene flow in world coordinate    # (B,N_cutFar,3)
            sf_bw_w = xyzs_bw_w-xyzs_w  # backward scene flow in world coordinate   # (B,N_cutFar,3)
            ret['reg_sp_sm_l'] = self.lambda_reg * (torch.abs(sf_fw_w[:, 1:]-sf_fw_w[:, :-1])*sp_w+
                                                    torch.abs(sf_bw_w[:, 1:]-sf_bw_w[:, :-1])*sp_w)
            ret['reg_sp_sm_l'] = reduce(ret['reg_sp_sm_l'], 'B n2 c -> B', 'mean')

        for k, loss in ret.items(): # 11
            ret[k] = loss.mean()

        return ret