import cv2
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from collections import defaultdict
from scipy.stats import linregress
from PIL import Image
from torchvision import transforms as T

from . import ray_utils, colmap_utils, pos_utils, flowlib


class MonocularDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(512, 288), start_end=(0, 30)):
        """
        split options:
            train               training mode, rays are from all images
            val                 validation mode, validate on the middle frame
            test                test on the training poses and times
            test_spiral         create spiral poses around the whole trajectory,
                                time is gradually advanced (only integers for now)
            test_spiralX        create spiral poses (fixed time) around training pose X
            test_fixviewX_interpY   fix view to training pose X and interpolate Y frames
                                    between each integer timestamps, from start to end
                                    must start from interp1 (save cur)
                                    interp Y-1 frames in between
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh                # training resizing
        self.start_frame = start_end[0]     # your data may have bonus imgs, we only take sorted (start,)
        self.end_frame = start_end[1]       # (,end) as our training imgs
        self.transform = T.ToTensor()
        self.cam_train = [0]                # # for val and eval
        self.read_meta()

    def read_meta(self):

        # [STEP 1] READ DATA ###########################################################################
        # read inputs
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))[self.start_frame:self.end_frame]
        self.disp_paths = sorted(glob.glob(os.path.join(self.root_dir, 'disps/*')))[self.start_frame:self.end_frame]
        self.mask_paths = sorted(glob.glob(os.path.join(self.root_dir, 'masks/*')))[self.start_frame:self.end_frame]
        # fw and bw flow is off head/tail, ADD dummy for full len
        self.flow_fw_paths = sorted(glob.glob(os.path.join(self.root_dir, 'flow_fw/*.flo')))[self.start_frame:self.end_frame] + ['dummy']
        self.flow_bw_paths = ['dummy'] + sorted(glob.glob(os.path.join(self.root_dir, 'flow_bw/*.flo')))[self.start_frame:self.end_frame]
        self.N_frames = len(self.image_paths)

        # read intrinsic -------------------------------------------------------------------------------
        camdata = colmap_utils.read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        # camdata[1] is a tuple having:
        #   ["id", "model", "width", "height", "params"]
        H = camdata[1].height       # real cam H
        W = camdata[1].width        # real cam W
        f = camdata[1].params[0]    # params: [f cx,cy,k]   here just need the focal

        self.K = np.array([[f, 0, W/2],
                           [0, f, H/2],
                           [0,  0,  1]], dtype=np.float32)
        self.K[0] *= self.img_wh[0]/W
        self.K[1] *= self.img_wh[1]/H
        # self.K : camera intrinsic matrix shared by all frames
        # TODO refer to my notes PAGE3 for a quick grasp of camera intrinsic and extrinsic

        # read extrinsics ------------------------------------------------------------------------------
        imdata = colmap_utils.read_images_binary(os.path.join(self.root_dir,'sparse/0/images.bin'))
        # imdata is dict with keys numed, and the tuple having:
        # ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
        # id: id in colmap ordered
        # name: original file name
        # q,t: Rt
        perm = np.argsort([imdata[k].name for k in imdata])
        #print(perm)
        # basically perm is just a list range(#frames)
        w2c_mats = []
        fill_line = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), fill_line], 0)]  #(4,4)
        print(len(w2c_mats))
        w2c_mats = np.stack(w2c_mats, 0)[perm]  
        w2c_mats = w2c_mats[self.start_frame:self.end_frame] # (N_frames, 4, 4)
        c2w_mats = np.linalg.inv(w2c_mats)  # (N_images,4,4)
        poses = c2w_mats[:, :3] # (N_frames, 3, 4) 
        # to turn the images data from camera specific to world space
        # to correct poses, change "right down front" of COLMAP to "right up back"
        self.poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses = pos_utils.center_poses(self.poses)

        # read bounds for pose correction --------------------------------------------------------------
        pts3d = colmap_utils.read_points3d_binary(os.path.join(self.root_dir,'sparse/0/points3D.bin'))
        # pts3d is dict of points reconstructed
        # ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
        # image_ids: who saw it
        pts_w = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)     location of pts
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_frames, N_points)
        for i, k in enumerate(pts3d):       # enum order in i and keys in k
            pts_w[0, :, i] = pts3d[k].xyz   # point loc
            for j in pts3d[k].image_ids:    # images saw it
                if self.start_frame <= j-1 < self.end_frame:
                    visibilities[j-1-self.start_frame, i] = 1  # frame j-1 saw point i

        min_depth = 1e8
        for i in range(self.N_frames):
            # for each image, compute the nearest depth according to real depth from COLMAP
            # and the disparity estimated by monodepth.
            # (using linear regression to find the best scale and shift)

            # read disparity estimated by monodepth
            disp = cv2.imread(self.disp_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)

            # read points
            pts_w_ = np.concatenate([pts_w[0], np.ones((1, len(pts3d)))], 0)    # all points (4, N_points)
            visibility_i = visibilities[i]          # for this frame, (N_points) 1 if visible
            pts_w_v = pts_w_[:, visibility_i==1]    # (4, N_points_v) only visible points
            pts_c_v = (w2c_mats[i] @ pts_w_v)[:3]   # (3, N_points_v) in camera space points loc
            pts_uvd_v = self.K @ pts_c_v            # (3, N_points_v) to the img by intrinsic
            pts_uv_v = (pts_uvd_v[:2]/pts_uvd_v[2:]).T # (N_points_v, 2)  x,y
            pts_uv_v = pts_uv_v.astype(int) # to integer pixel coordinates
            pts_uv_v[:, 0] = np.clip(pts_uv_v[:, 0], 0, self.img_wh[0]-1)
            pts_uv_v[:, 1] = np.clip(pts_uv_v[:, 1], 0, self.img_wh[1]-1)   # (2, N_points_v) xy for pts
            pts_d_v = pts_uvd_v[2]                                          # (1, N_points_v) depth for pts

            reg = linregress(1/pts_d_v, disp[pts_uv_v[:, 1], pts_uv_v[:, 0]]) # further depth, less disp
            if reg.rvalue**2 > 0.9: # coefficient of determination > 0.9, if the regression is authentic
                # the equation:  disp = slope*1/depth +intercept
                # and then depth = slope/(disp-intercept)
                # use 95% as a large disp/less depth to closer the bound
                min_depth = min(min_depth, reg.slope/(np.percentile(disp, 95)-reg.intercept))
            else:
                # if this is not trusted then just go 5% of the COLMAP ..
                min_depth = min(min_depth, np.percentile(pts_d_v, 5))
            # basically we want to use disp as much as an info here
        # correct scale using min depth
        self.poses[..., 3] /= (min_depth * 0.75)

        # [STEP 2] GENERATE MATRIX #####################################################################
        # create projection matrix, used to compute optical flow
        # this part is buffered for loss, not used directly here
        bottom_line = np.zeros((self.N_frames, 1, 4))    # (#frames,1,4)
        bottom_line[..., -1] = 1  # as a point affine
        Rt = np.linalg.inv(np.concatenate([self.poses, bottom_line], 1))[:, :3] # w2c_mats (#frames,3,4)
        # TODO cam projection
        Rt[:, 1:] *= -1     # "right up back" to "right down front" for cam projection
        self.Ps = self.K @ Rt
        self.Ps = torch.FloatTensor(self.Ps).unsqueeze(0) # (1, N_frames, 3, 4)      w2c_mats
        self.Ks = torch.FloatTensor(self.K).unsqueeze(0) # (1, 3, 3)        camera intrinsics

        # [STEP 3] GENERATE RAY ########################################################################
        # for the training, this step already gives everything you need for a data load
        # for the validation, this step does nothing
        # for the testing, this step gives only poses
        # TODO more ways of generating rays for testing
        if self.split == 'train':
            self.last_t = -1
            directions, uv = ray_utils.get_ray_directions(self.img_wh[1], self.img_wh[0], self.K, return_uv=True)
            # directions: (HW,3) uv: (HW,2); uv starts from center, scanline left->down
            # directions starts from left top to right bot
            self.rays_dict = {}
            for t in range(self.N_frames):
                # load img
                img = Image.open(self.image_paths[t]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img).view(3, -1).T # (HW,3)

                # load origin and rays
                c2w = torch.FloatTensor(self.poses[t])
                rays_o, rays_d = ray_utils.get_rays(directions, c2w) # (HW, 3) for both
                # to NDC
                shift_near = -min(-1.0, self.poses[t, 2, 3]) # "right up back" c2w (3,4)
                # for this img
                    # if camera is moving APPROACHING near to the scene, where shift_near > 1, set it to new val
                    # if camera is not crossing z = -1 , set it to 1
                # reason for this: dynamic scene is not guaranteeing the camera will not shake back&forth a lot..
                rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, shift_near, rays_o, rays_d)
                # time tag for ray
                rays_t = t * torch.ones(len(rays_o), 1) # (HW, 1)
                # in total, loading the origin and rays give:
                # rays_o # (HW, 3)
                # rays_d # (HW, 3)
                # rays_t # (HW, 1)

                # load disparity 
                disp = cv2.imread(self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                disp = torch.FloatTensor(disp).reshape(-1, 1) # (HW,1)

                # load mask
                mask = Image.open(self.mask_paths[t]).convert('L')
                mask = mask.resize(self.img_wh, Image.NEAREST)
                mask = self.transform(mask).flatten() # (h*w)
                rays_mask = mask.unsqueeze(-1) # (HW,1)
                # 0:static, 1:dynamic as preprocess defined here

                # load optical flow
                if t < self.N_frames-1:
                    flow_fw = flowlib.read_flow(self.flow_fw_paths[t])
                    flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                    flow_fw = torch.FloatTensor(flow_fw.reshape(-1, 2))
                    # check the flowlib for a quick visualization and save
                else:
                    flow_fw = torch.zeros(len(rays_o), 2)

                if t >= 1:
                    flow_bw = flowlib.read_flow(self.flow_bw_paths[t])
                    flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                    flow_bw = torch.FloatTensor(flow_bw.reshape(-1, 2))
                else:
                    flow_bw = torch.zeros(len(rays_o), 2)
                # in total, loading the optical flow give:
                # flow_fw # (HW, 2)
                # flow_bw # (HW, 2)               

                rays = [rays_o,     # 3
                        rays_d,     # 3
                        img,        # 3
                        rays_t,     # 1
                        disp,       # 1
                        rays_mask,  # 1
                        uv+flow_fw, # 2  optical flow predicted next pixel loc
                        uv+flow_bw] # 2  optical flow predicted prev pixel loc
                self.rays_dict[t] = torch.cat(rays, 1) # (HW,16)


        # generate for testing
        # TEST_SCENE1 test       test on the training poses and times
        elif self.split == 'test':
            self.poses_test = self.poses.copy()     # (#frames,3,4)

        # TEST_SCENE2 test_fixviewX_interpY       test on the training poses and times
        elif self.split.startswith('test_fixview'):
            # fix to target view and change time
            target_idx = int(self.split.split('_')[1][7:])
            self.poses_test = np.tile(self.poses[target_idx], (self.N_frames, 1, 1))  # (#frames,3,4)

        elif self.split.startswith('test_spiral'):
            # TEST_SCENE3 test_spiral      spiral on the whole sequence
            if self.split == 'test_spiral': 
                # create spiral poses around the whole trajectory
                shaking = 0.05 # tune the shaking for your own
                x_r = np.sum(np.abs(poses[:, 0, 3]))/self.N_frames*shaking
                y_r = np.sum(np.abs(poses[:, 1, 3]))/self.N_frames*shaking
                # spiral it
                radii = np.array([x_r, y_r, 0]) # not moving z
                # no spiral
                #radii = np.array([0, 0, 0])
                self.poses_test = pos_utils.create_spiral_poses(self.poses, radii, n_poses=6*self.N_frames)

            # TEST_SCENE4 test_spiralX      create spiral poses (fixed time) around training pose X                        
            else:
                target_idx = int(self.split.split('_')[1][6:])
                r = np.abs(self.poses[target_idx, 0, 3]-self.poses[-1, 0, 3])*0.15  # tune this if too large
                self.poses_test = pos_utils.create_spiral_poses_single(self.poses[target_idx], max_trans=r, n_poses=60)


    def __len__(self):
        if self.split == 'train':
            # epoch size 
            # so actually there are 1000 epochs to go over all of them..
            # in each step in one epoch, train size given by batch_size
            return self.img_wh[0]*self.img_wh[1]*self.N_frames//1000
        if self.split == 'val': return 1
        # test
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train':
            # SELECT t / which img to train
            if self.last_t == -1:
                # choose any time if first one
                t = np.random.choice(self.N_frames)
            else:
                # do repeat on neighbors (neighbor-3)
                valid_t = list(set(range(self.N_frames))- set(range(self.last_t-3, self.last_t+4)))
                t = np.random.choice(valid_t)
            self.last_t = t

            # SELECT rays
            # rays_dict[t] (HW,16)
            rand_idx = np.random.choice(len(self.rays_dict[t]), self.batch_size) # batch_size out of HW
            rays = self.rays_dict[t][rand_idx]  # (B,16)
            sample = {'rays':       rays[:, :6],
                      'rgbs':       rays[:, 6:9],
                      'ts':         rays[:, 9].long(),     # has no second dim
                      'cam_ids':    0*rays[:, 9].long(),   # for the loss, it's always 0 because mono
                      'disps':      rays[:, 10],           # has no second dim
                      'rays_mask':  rays[:, 11],           # has no second dim
                      'uv_fw':      rays[:, 12:14],
                      'uv_bw':      rays[:, 14:16]}
        else:
            if self.split == 'val':
                # use center frame for validation
                t = self.N_frames//2
                c2w = torch.FloatTensor(self.poses[t])
            else:
                # for the idx-th testing, get c2w and t
                # the t is totally depends on you!
                # TODO this could be hardcode or optioned for different render results
                c2w = torch.FloatTensor(self.poses_test[idx])
                if self.split == 'test':    # test on the training poses and times
                    t = idx
                elif self.split.startswith('test_spiral'):
                    if self.split == 'test_spiral':     # all pos, time roaming
                        # fix time
                        t = 8
                        # roaming time
                        #t = int(idx/len(self.poses_test)*self.N_frames)
                    else:   # single pos, fix time to spiral
                        t = int(self.split.split('_')[1][6:])
                elif self.split.startswith('test_fixview'): # fixview has #frame time
                    t = idx 
                else: t = 0
                
            # get t and c2w

            directions = ray_utils.get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)
            rays_o, rays_d = ray_utils.get_rays(directions, c2w)
            shift_near = -min(-1.0, c2w[2, 3])
            rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, shift_near, rays_o, rays_d)
            rays_t = t * torch.ones(len(rays_o), dtype=torch.long) # (HW)

            rays = torch.cat([rays_o, rays_d], 1) # (HW,6)

            sample = {'rays': rays, 'ts': rays_t, 'c2w': c2w}

            sample['cam_ids'] = 0
            img = Image.open(self.image_paths[t]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3,H,W)
            img = img.view(3, -1).T # (HW, 3)
            sample['rgbs'] = img

            disp = cv2.imread(self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
            sample['disp'] = torch.FloatTensor(disp.flatten()) # (HW)

            mask = Image.open(self.mask_paths[t]).convert('L')
            mask = mask.resize(self.img_wh, Image.NEAREST)
            sample['mask'] = self.transform(mask).flatten() # (HW)

            if t < self.N_frames-1:
                flow_fw = flowlib.read_flow(self.flow_fw_paths[t])
                flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                sample['flow_fw'] = flow_fw
            else:
                sample['flow_fw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

            if t >= 1:
                flow_bw = flowlib.read_flow(self.flow_bw_paths[t])
                flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                sample['flow_bw'] = flow_bw
            else:
                sample['flow_bw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

            # val/eval:
            # rays          (HW,6)
            # ts            (HW)
            # c2w           (3,4)
            # cam_ids       0
            # rgbs  (@t)    (HW,3)
            # disp  (@t)    (HW)
            # mask  (@t)    (HW)
            # flow_fw (@t)  (HW,2)
            # flow_bw (@t)  (HW,2)

        # train sample: dict with items (B,..)
        return sample   