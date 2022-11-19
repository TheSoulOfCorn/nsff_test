import torch
from einops import rearrange, reduce, repeat
from datasets import ray_utils

# for frame interpolation
from kornia import create_meshgrid
from .softsplat import FunctionSoftsplat

def render_rays(models,         # fine model for nsff, get it as models['fine']
                embeddings,     # 'xyz','dir','t'
                rays,           # (chunk,6)
                ts,             # (chunk) ray time, always exist bcz output_transient is always true
                max_t,          # time max, default as self.N_frames-1
                N_samples=128,  # samples for nerf per ray
                perturb=0,      # factor to perturb the sampling position on the ray
                noise_std=0,    # factor to perturb the model's prediction of sigma
                chunk=1024*32,  # chunk size
                test_time=False,# True for val, False for train. whether it is test (inference only) or not. 
                **kwargs):      # 'output_transient': as always True
                                # 'output_transient_flow': ['fw', 'bw', 'disocc'] or [] (for val)

    def inference(results, model, xyz, zs, test_time=False, **kwargs):
        """
        Model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF and NSFF model
            xyz: (C,N,3) sampled positions
            zs: (C,N) depths of the sampled positions
            test_time: test time or not
        """

        def render_transient_warping(xyz, t_embedded, flow):
            """
            Helper function that performs forward or backward warping for dynamic scenes.
            static sigma and rgbs of the CURRENT time are used to composite the result.
            Inputs:
                xyz: warped xyz
                t_embedded: embedded time for the warping time instance (t+i)
                flow: 'fw' or 'bw', the flow for the warped xyz
            Outputs:
                rgb_map_warped: (C, 3) warped rendering
                transient_flows_: (C, N, 3) warped points' fw/bw flow
                transient_weights_w: (C, N) warped transient weights, used to infer occlusion
            """
            out_chunks = []
            for i in range(0, B, chunk):
                inputs = [embedding_xyz(xyz[i:i+chunk]),    # xyz warped
                          dir_embedded_[i:i+chunk],         # dir not warped
                          t_embedded[i:i+chunk]]            # t warped for sure
                out_chunks += [model(torch.cat(inputs, 1),
                                     output_static=False,           # not full dim! not using static model
                                     output_transient=True,         
                                     output_transient_flow=[flow])]
                # the out_chunks shape: (B,7)
            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(C N) c -> C N c', C=N_rays, N=N_samples_) # (C,N,7)
            transient_rgbs_w = out[..., :3]     # (C,N,3) warped dyn rgb
            transient_sigmas_w = out[..., 3]    # (C,N) warped dyn sigmas
            transient_flows_w = out[..., 4:7]   # (C,N,3) warped scene's flows
            # bw for the fw warped, fw for the bw warped.
            transient_flows_w[zs>z_far] = 0     # far scene doesnt warp

            # warped scene alpha

            # warped dyn alpha compute
            transient_sigmas_w = act(transient_sigmas_w+torch.randn_like(transient_sigmas_w)*noise_std) # add noise
            transient_alphas_w = 1-torch.exp(-transient_deltas*transient_sigmas_w)
            # whole scene alpha and transmit
            alphas_w = 1-(1-static_alphas)*(1-transient_alphas_w)  # (C,N)
            alphas_w_sh = torch.cat([torch.ones_like(alphas_w[:, :1]), 1-alphas_w], -1)  # (C,N+1)
            transmittance_w = torch.cumprod(alphas_w_sh[:, :-1], -1) # (C,N)  used for both dyn and static scenes

            static_weights_w = rearrange(static_alphas*transmittance_w, 'C N -> C N 1')
            transient_weights_w = rearrange(transient_alphas_w*transmittance_w, 'C N -> C N 1')
            # (C,N,1)*(C,N,3) -> (C,N,3) and reduce as volume rendering -> (C,3)
            static_rgb_map_w = reduce(static_weights_w*static_rgbs, 'C N c -> C c', 'sum')      # (C,3)
            transient_rgb_map_w = reduce(transient_weights_w*transient_rgbs_w, 'C N c -> C c', 'sum')   # (C,3)
            rgb_map_w = static_rgb_map_w + transient_rgb_map_w  # (C,3)
            # returning:
            # rgb_map_w         (C,3) rgb for whole color
            # transient_flows_w (C,N,3) bw for the fw warped, fw for the bw warped.
            # transient_weights_w[..., 0] (C,N) weights for each points
            return rgb_map_w, transient_flows_w, transient_weights_w[..., 0]

        typ = model.typ
        results[f'zs_{typ}'] = zs       # output zs_fine
        results[f'xyzs_{typ}'] = xyz    # output xyzs_fine
        N_samples_ = xyz.shape[1]       # N
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3) # (CN,3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0] # B = CN
        out_chunks = []

        # (C,#dir_eb) -> (CN,#dir_eb)
        dir_embedded_ = repeat(dir_embedded, 'C dir_eb -> (C n2) dir_eb', n2=N_samples_)
        # (C,#t_eb) -> (CN,#t_eb)
        t_embedded_ = repeat(t_embedded, 'C t_eb -> (C n2) t_eb', n2=N_samples_)

        for i in range(0, B, chunk): # B is CN, chunk should be low now
            inputs = [embedding_xyz(xyz_[i:i+chunk]),   # (newC,#xyz_eb)
                      dir_embedded_[i:i+chunk],         # (newC,#dir_eb)
                      t_embedded_[i:i+chunk]]           # (newC,#t_eb)
            chunk_result = model(torch.cat(inputs, 1),  # (newC,#xyz_eb+#dir_eb+#t_eb)
                                 output_transient=True,
                                 output_transient_flow=output_transient_flow)
            # (newC,14) static 4 + dyn 4 + fw 3 + bw 3
            out_chunks += [chunk_result]

        out = torch.cat(out_chunks, 0)  # (B,14)
        out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_) # (C,N,14)

        # rsults after model
        static_rgbs = out[..., :3]      # (C,N,3)
        static_sigmas = out[..., 3]     # (C,N)
        results[f'static_rgbs_{typ}'] = static_rgbs         # output static_rgbs_fine
        transient_rgbs = out[..., 4:7]  # (C,N,3)
        transient_sigmas = out[..., 7]  # (C,N)
        results[f'transient_rgbs_{typ}'] = transient_rgbs   # output transient_rgbs_fine

        if output_transient_flow: # dim 14 here for sure ( [] then nothing)
            # output_transient is anyway true, dim depends on output_transient_flow
            transient_flows_fw = out[..., 8:11]     # (C,N,3)
            results['transient_flows_fw'] = transient_flows_fw  # output transient_flows_fw
            transient_flows_bw = out[..., 11:14]    # (C,N,3)
            results['transient_flows_bw'] = transient_flows_bw  # output transient_flows_bw
            # zs: (C,N)  far away than z_far (0.95) has no flow (far scene freeze)
            transient_flows_fw[zs>z_far] = 0
            transient_flows_bw[zs>z_far] = 0

        # TODO part of the eval
        # set invisible transient_sigmas to a very negative value
        if test_time and 'dataset' in kwargs:
            dataset = kwargs['dataset']
            K = dataset.Ks[0].to(xyz.device)
            visibilities = torch.zeros(len(xyz_), device=xyz.device)
            xyz_w = ray_utils.ndc2world(xyz_, K)
            for i in range(len(dataset.cam_train)):
                ray_utils.compute_world_visiblility(visibilities,
                    xyz_w, K, dataset.img_wh[1], dataset.img_wh[0],
                    torch.FloatTensor(dataset.poses[i*dataset.N_frames+ts[0]]).to(xyz.device))
            transient_sigmas[visibilities.view_as(transient_sigmas)==0] = -10

        # deltas
        deltas = zs[:, 1:] - zs[:, :-1]     # (C, N-1)
        static_deltas = torch.cat([deltas, 100*torch.ones_like(deltas[:, :1])], -1) # (C,N) tail 100
        transient_deltas = torch.cat([deltas, 1e-3*torch.ones_like(deltas[:, :1])], -1) # (C,N) tail 1e-3

        # static alpha 
        static_sigmas = act(static_sigmas+torch.randn_like(static_sigmas)*noise_std) # add noise
        results[f'static_sigmas_{typ}'] = static_sigmas     # output static_sigmas_fine
        static_alphas = 1-torch.exp(-static_deltas*static_sigmas)   # (C, N)

        # dyn alpha
        transient_sigmas = act(transient_sigmas+torch.randn_like(transient_sigmas)*noise_std) # add noise
        results[f'transient_sigmas_{typ}'] = transient_sigmas  # output transient_sigmas_fine
        transient_alphas = 1-torch.exp(-transient_deltas*transient_sigmas)  # (C, N)

        # sum alpha
        alphas = 1-(1-static_alphas)*(1-transient_alphas)   # (C, N)

        # training, bw and fw flow render
        if (not test_time) and output_transient_flow: # render with flowed-xyzs

            # forward
            xyz_fw = xyz + transient_flows_fw  # (C,N,3) + (C,N,3), model predicted xyz forward position
            results['xyzs_fw'] = xyz_fw     # output xyzs_fw
            xyz_fw_ = rearrange(xyz_fw, 'C N c -> (C N) c', c=3)    # (CN,3)    for computation
            tp1_embedded = embeddings['t'](torch.clamp(ts+1, max=max_t)) # (C,#t_eb) for next t 
            tp1_embedded_ = repeat(tp1_embedded, 'C c -> (C n2) c', n2=N_samples_)  # (CN,#t_eb)  for computation
            results['rgb_fw'], transient_flows_fw_bw, transient_weights_fw = \
                render_transient_warping(xyz_fw_, tp1_embedded_, 'bw')
            # results['rgb_fw']         (C,3)       fw rgb for whole
            # transient_flows_fw_bw     (C,N,3)     fw scene's bw flow
            # transient_weights_fw      (C,N)       weights for each points in fw scene

            # backward
            xyz_bw = xyz + transient_flows_bw # (C,N,3) + (C,N,3), model predicted xyz backward position
            results['xyzs_bw'] = xyz_bw     # output xyzs_bw
            xyz_bw_ = rearrange(xyz_bw, 'C N c -> (C N) c', c=3)    # (CN,3)    for computation
            tm1_embedded = embeddings['t'](torch.clamp(ts-1, min=0)) # (C,#t_eb) for prev t 
            tm1_embedded_ = repeat(tm1_embedded, 'C c -> (C n2) c', n2=N_samples_) # (CN,#t_eb)  for computation
            results['rgb_bw'], transient_flows_bw_fw, transient_weights_bw = \
                render_transient_warping(xyz_bw_, tm1_embedded_, 'fw')
            # results['rgb_bw']         (C,3)       bw rgb for whole
            # transient_flows_bw_fw     (C,N,3)     bw scene's fw flow
            # transient_weights_bw      (C,N)       weights for each points in bw scene

            # to compute fw-bw consistency
            results['xyzs_fw_bw'] = xyz_fw + transient_flows_fw_bw
            results['xyzs_bw_fw'] = xyz_bw + transient_flows_bw_fw

        # volume render the cur scene now..
        alphas_sh = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1)
        transmittance = torch.cumprod(alphas_sh[:, :-1], -1)    # (C, N) overall transmittance

        static_weights = static_alphas * transmittance          # (C, N) static
        transient_weights = transient_alphas * transmittance    # (C, N) dyn

        weights = alphas * transmittance                # (C, N) overall weights
        weights_ = rearrange(weights, 'C N -> C N 1')   # (C,N,1)  for computation

        results[f'static_weights_{typ}'] = static_weights       # output static_weights_fine
        results[f'transient_weights_{typ}'] = transient_weights # output transient_weights_fine
        results[f'weights_{typ}'] = weights                     # output weights_fine

        if test_time:   # val and eval  
            results[f'static_alphas_{typ}'] = static_alphas        # output static_alphas_fine
            results[f'transient_alphas_{typ}'] = transient_alphas  # output transient_alphas_fine

        results[f'depth_{typ}'] = reduce(weights*zs, 'C N -> C', 'sum') # (C) output depth_fine

        # static_rgb_map computation
        static_weights_ = rearrange(static_weights, 'C N -> C N 1')
        static_rgb_map = reduce(static_weights_*static_rgbs,'C N c -> C c', 'sum')

        # transient_rgb_map computation
        transient_weights_ = rearrange(transient_weights, 'C N -> C N 1')
        transient_rgb_map = reduce(transient_weights_*transient_rgbs, 'C N c -> C c', 'sum')

        results[f'rgb_{typ}'] = static_rgb_map + transient_rgb_map  # output rgb_fine (C,3)

        # not quite sure whats this for
        results[f'transient_alpha_{typ}'] = reduce(transient_weights, 'C N -> C', 'sum') # output what ???
        # results[f'transient_rgb_{typ}'] = transient_rgb_map + 0.8*(1-rearrange(results[f'transient_alpha_{typ}'], 'n1 -> n1 1')) # gray bg


        # Compute also depth and rgb when only one field exists. (only static alpha)
        static_alphas_sh = torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)  # (C,N+1)
        static_transmittance = torch.cumprod(static_alphas_sh[:, :-1], -1)  # (C,N+1)
        _static_weights = static_alphas * static_transmittance  # (C,N) transmittance is static only
        _static_weights_ = rearrange(_static_weights, 'C N -> C N 1') # (C,N,1) for computation
        results[f'_static_rgb_{typ}'] = reduce(_static_weights_*static_rgbs, 'C N c -> C c', 'sum') # (C,3)
        results[f'_static_depth_{typ}'] = reduce(_static_weights*zs, 'C n2 -> C', 'sum')  # (C)

        if output_transient_flow:
            # xyz: sampled (C,N,3)  weights_ : (C,N,1)
            # you dont have to understand what is the pos*weights for the first line..
            results['xyz_fine'] = reduce(weights_*xyz, 'C N c-> C c', 'sum')
            results['transient_flow_fw'] = reduce(weights_*transient_flows_fw, 'C N c -> C c', 'sum')
            # because it's actually weights*(xyz+flow)
            results['xyz_fw'] = results['xyz_fine']+results['transient_flow_fw']    # (C,3) weighted fw pos
            results['transient_flow_bw'] = reduce(weights_*transient_flows_bw, 'C N c -> C c', 'sum')
            results['xyz_bw'] = results['xyz_fine']+results['transient_flow_bw']    # (C,3) weighted bw pos

            if (not test_time) and 'disocc' in output_transient_flow:   # only in train
                # weights in fw scene - weights in cur scene, both use same static scene base
                occ_fw = (transient_weights_fw-transient_weights).detach() # (C,N)
                occ_bw = (transient_weights_bw-transient_weights).detach() # (C,N)
                # what's going on here for the non-trained occ?
                # we use dyn weights from the MLP to get the disocc
                # notice the weights come from : [alphas,fw,bw] namely sigmas and fw,bw
                # for a pos
                #   occ small means THE DYN in fw/bw and cur almost same seen in this pos
                #       high disocc may lead to NO CHANGE
                #   occ big means THE DYN in fw/bw and cur changed in this pos
                #       low disocc may lead to CHANGE HUGE
                results['disoccs_fw'] = 1-torch.abs(rearrange(occ_fw, 'C N -> C N 1'))
                results['disoccs_bw'] = 1-torch.abs(rearrange(occ_bw, 'C N -> C N 1'))

                # for a ray 
                #   same case
                results['disocc_fw'] = 1-torch.abs(reduce(occ_fw, 'C N -> C 1', 'sum'))
                results['disocc_bw'] = 1-torch.abs(reduce(occ_bw, 'C N -> C 1', 'sum'))


        # a short summary what we have in the results ..
        # zs_fine               (C,N)   sampled depth, range [0,1]
        # xyzs_fine             (C,N,3) sample position
        # static_rgbs_fine      (C,N,3) MLP output static rgb
        # transient_rgbs_fine   (C,N,3) MLP output dyn rgb
        # IF output_transient_flow:
        #   transient_flows_fw  (C,N,3) MLP output fw
        #   transient_flows_bw  (C,N,3) MLP output bw
        # static_sigmas_fine    (C,N)   (noised) MLP output static sigma
        # transient_sigmas_fine (C,N)   (noised) MLP output dyn sigma
        # IF training and output_transient_flow:
        #   xyzs_fw             (C,N,3) model predicted xyz forward position
        #   rgb_fw              (C,3)   fw warped volome render rgb, with cur static scene
        #   xyzs_bw             (C,N,3) model predicted xyz backward position
        #   rgb_bw              (C,3)   bw warped volome render rgb, with cur static scene
        #   xyzs_fw_bw          (C,N,3) model predicted xyz postion fw->bw
        #   xyzs_bw_fw          (C,N,3) model predicted xyz postion bw->fw
        # static_weights_fine   (C,N)   static weights (static_alphas * overall_transmittance )
        # transient_weights_fine(C,N)   dyn weights (transient_alphas * overall_transmittance )
        # weights_fine          (C,N)   overall weights (overall_alphas * overall_transmittance )
        # IF testtime(val/eval):
        #   static_alphas_fine  (C,N)   static_alphas
        #   transient_alphas_fine(C,N)  transient_alphas
        # depth_fine            (C)     overall weights * sampled depth and sum
        # rgb_fine              (C,3)   volume rendering sum of both static and dyn
        # transient_alpha_fine  (C)     transient_weights_fine sumed, TODO what's this for
        # _static_rgb_fine      (C,3)   using all static compute color, nothing dyn
        # _static_depth_fine    (C)     using all static compute depth, nothing dyn
        # IF output_transient_flow:
        #   xyz_fine            (C,3)   overall weights*(xyz)  summed, will not real use!
        #   transient_flow_fw   (C,3)   overall weights*(fw_flow)   summed
        #   xyz_fw              (C,3)   overall weights*(xyz+fw_flow) weighted next pos
        #   transient_flow_bw   (C,3)   overall weights*(bw_flow)   summed
        #   xyz_bw              (C,3)   overall weights*(xyz+bw_flow) weighted prev pos
        # IF training and wants 'disocc':
        #   disoccs_fw          (C,N,1) fw disocclusion in each pos
        #   disoccs_bw          (C,N,1) bw disocclusion in each pos
        #   disocc_fw           (C,1)   summed version
        #   disocc_bw           (C,1)   summed version

        return

    # begins here...
    results = {}
    act = torch.nn.Softplus()   # GLOBAL sigma activation function
    z_far = 0.95                # GLOBAL explicitly zero the flow if z exceeds this value

    N_rays = rays.shape[0]  # chunk (C)
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # (C,3) for both
    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']  # a nnModule class for both
    dir_embedded = embedding_dir(rays_d)    # (C,#dir_eb)
    t_embedded = embeddings['t'](ts)    # ts: (C) -> (C,#t_eb)
    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')    # (C,1,3)
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')    # (C,1,3)

    # sample depths
    zs = torch.linspace(0, 1, N_samples, device=rays.device)    # (N)
    zs = zs.expand(N_rays, N_samples)   # (C,N)
    
    if perturb > 0: # perturb sample depths
        # get intervals between samples
        zs_mid = 0.5 * (zs[: ,:-1]+zs[: ,1:])   # (C, N-1) interval mid points
        upper = torch.cat([zs_mid, zs[: ,-1:]], -1)
        lower = torch.cat([zs[: ,:1], zs_mid], -1)
        perturb_rand = perturb * torch.rand_like(zs)
        zs = lower + (upper - lower) * perturb_rand

    # (C,1,3) + (C,1,3)*(C,N,1) = (C,N,3)
    xyz_fine = rays_o + rays_d * rearrange(zs, 'C N -> C N 1')


    model = models['fine']
    output_transient_flow = kwargs.get('output_transient_flow', [])  # GLOBAL what to output
    # TODO this is so confusing when we have global var.. just organize it a little ..
    inference(results,      # empty dict
              model,        # check nerf
              xyz_fine,     # (C,N,3)
              zs,           # (C,N)
              test_time,    # train or val
              **kwargs)     # and three more global vars ..

    return results

def interpolate(results_t, results_tp1, dt, K, c2w, img_wh):
    """
    Interpolate between two results t and t+1 to produce t+dt, dt in (0, 1).
    For each sample on the ray (the sample points lie on the same distances, so they
    actually form planes), compute the optical flow on this plane, then use softsplat
    to splat the flows. Finally use MPI technique to compute the composite image.
    Used in test time only.

    Inputs:
        results_t:    cur frame render result dict
        results_tp1:  next frame render result dict  
        dt:           float in (0, 1)
        K:            (3,3) intrinsics matrix
        c2w:          (3,4) current pose    # IT DOESNT MATTER !!!! IT'S ALL SAME POS!!
                      # because interpolate here is for fix_view, the c2w never changes.
        img_wh:       image width and height

    Outputs:
        (img_wh[1], img_wh[0], 3) rgb interpolation result
        (img_wh[1], img_wh[0]) depth of the interpolation (in NDC)
    """
    device = results_t['xyzs_fine'].device
    N_rays, N_samples = results_t['xyzs_fine'].shape[:2] # (C),(N)
    w, h = img_wh
    rgba = torch.zeros((h, w, 4), device=device)
    depth = torch.zeros((h, w), device=device)

    # cam -----------------------------------------------------------------------------------------
    c2w_ = torch.eye(4)
    c2w_[:3] = c2w
    w2c = torch.inverse(c2w_)[:3]
    w2c[1:] *= -1 # "right up back" to "right down forward" for cam projection
    P = K @ w2c # (3, 4) projection matrix

    # base ----------------------------------------------------------------------------------------
    # (1,H,W,2) grids
    grid = create_meshgrid(h, w, False, device)
    # (C,N,3) sample position, equals results_tp1['xyzs_fine']
    xyzs = results_t['xyzs_fine']
    # (C,N)->(H,W,N) depth(0,1)
    zs = rearrange(results_t['zs_fine'], '(h w) N -> h w N', w=w, h=h)

    # static buffers ------------------------------------------------------------------------------
    # (C,N,3) -> (H,W,N,3) MLP output static rgb
    static_rgb = rearrange(results_t['static_rgbs_fine'],'(h w) N c -> h w N c', w=w, h=h, c=3)
    # (C,N) -> (H,W,N,1) static_alphas
    static_a = rearrange(results_t['static_alphas_fine'],'(h w) N -> h w N 1', w=w, h=h)

    # forward buffers -----------------------------------------------------------------------------
    # from cur frame to the delta middle frame, that's why it's forward
    # (CN,3) sample position in world
    xyzs_w = ray_utils.ndc2world(rearrange(xyzs, 'C N c -> (C N) c'), K)
    # (CN,3) MLP predicted full fw sample position in world
    xyzs_fw_w = ray_utils.ndc2world(rearrange(xyzs+results_t['transient_flows_fw'], 'C N c -> (C N) c'), K)
    # (CN,3) scale with dt
    xyzs_fw_w = xyzs_w + dt*(xyzs_fw_w-xyzs_w)
    # (3,CN) projected screen loc with depth
    uvds_fw = P[:3, :3] @ rearrange(xyzs_fw_w, 'n c -> c n') + P[:3, 3:]
    # (2,CN) projected screen loc 
    uvs_fw = uvds_fw[:2] / uvds_fw[2]
    # (2,CN) -> (2,C,N)
    uvs_fw = rearrange(uvs_fw, 'c (C N) -> c C N', C=N_rays, N=N_samples)
    # (2,C,N) -> (N,H,W,2)
    uvs_fw = rearrange(uvs_fw, 'c (h w) N -> N h w c', w=w, h=h)
    # (N,H,W,2) -> (N,2,H,W) optical flow forward
    of_fw = rearrange(uvs_fw-grid, 'N h w c -> N c h w', c=2)

    # (C,N,3) -> (N,3,H,W) MLP output dyn rgb
    transient_rgb_t = rearrange(results_t['transient_rgbs_fine'],'(h w) N c -> N c h w', w=w, h=h, c=3)
    # (C,N) -> (N,1,H,W) MLP output dyn alpha
    transient_a_t = rearrange(results_t['transient_alphas_fine'],'(h w) N -> N 1 h w', w=w, h=h)
    # (N,4,H,W) dyn rgb+alpha
    transient_rgba_t = torch.cat([transient_rgb_t, transient_a_t], 1)

    # backward buffers ----------------------------------------------------------------------------
    # from next frame to the delta middle frame, that's why it's backward
    # (CN,3) MLP predicted full bw sample position in world
    xyzs_bw_w = ray_utils.ndc2world(rearrange(xyzs+results_tp1['transient_flows_bw'],'C N c -> (C N) c'), K)
    # (CN,3) scale with 1-dt
    xyzs_bw_w = xyzs_w + (1-dt)*(xyzs_bw_w-xyzs_w)
    # (3,CN) projected screen loc with depth
    uvds_bw = P[:3, :3] @ rearrange(xyzs_bw_w, 'n c -> c n') + P[:3, 3:]
    # (2,CN) projected screen loc 
    uvs_bw = uvds_bw[:2] / uvds_bw[2]
    # (2,CN) -> (2,C,N)
    uvs_bw = rearrange(uvs_bw, 'c (C N) -> c C N', n1=N_rays, n2=N_samples)
    # (2,C,N) -> (N,H,W,2)
    uvs_bw = rearrange(uvs_bw, 'c (h w) N -> N h w c', w=w, h=h)
    # (N,H,W,2) -> (N,2,H,W) optical flow backward
    of_bw = rearrange(uvs_bw-grid, 'N h w c -> N c h w', c=2)

    # (C,N,3) -> (N,3,H,W) MLP output dyn rgb
    transient_rgb_tp1 = rearrange(results_tp1['transient_rgbs_fine'],'(h w) N c -> N c h w', w=w, h=h, c=3)
    # (C,N) -> (N,1,H,W) MLP output dyn alpha
    transient_a_tp1 = rearrange(results_tp1['transient_alphas_fine'],'(h w) N -> N 1 h w', w=w, h=h)
    # (N,4,H,W) dyn rgb+alpha
    transient_rgba_tp1 = torch.cat([transient_rgb_tp1, transient_a_tp1], 1)

    # directly borrowed from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/main/nsff_exp/softsplat.py
    for s in range(N_samples): # compute MPI planes (front to back composition)
        transient_rgba_fw = FunctionSoftsplat(tenInput=transient_rgba_t[s:s+1].cuda(), 
                                              tenFlow=of_fw[s:s+1].cuda(), 
                                              tenMetric=None, 
                                              strType='average').cpu()
        transient_rgba_fw = rearrange(transient_rgba_fw, '1 c h w -> h w c')
        transient_rgba_bw = FunctionSoftsplat(tenInput=transient_rgba_tp1[s:s+1].cuda(), 
                                              tenFlow=of_bw[s:s+1].cuda(), 
                                              tenMetric=None, 
                                              strType='average').cpu()
        transient_rgba_bw = rearrange(transient_rgba_bw, '1 c h w -> h w c')
        # transient_rgba_fw (H,W,4)
        # transient_rgba_bw (H,W,4)
        composed_rgb = transient_rgba_fw[..., :3]*transient_rgba_fw[..., 3:]*(1-dt) + \
                       transient_rgba_bw[..., :3]*transient_rgba_bw[..., 3:]*dt + \
                       static_rgb[:, :, s]*static_a[:, :, s]
        temp_dyn_a = (transient_rgba_fw[..., 3:]*(1-dt)+transient_rgba_bw[..., 3:]*dt)
        composed_a = 1 - (1-temp_dyn_a)*(1-static_a[:, :, s])
        rgba[..., :3] += (1-rgba[..., 3:])*composed_rgb
        depth += (1-rgba[..., 3])*composed_a[..., 0]*zs[..., s]
        rgba[..., 3:] += (1-rgba[..., 3:])*composed_a

    return rgba[..., :3], depth # (H,W,3) (H,W)