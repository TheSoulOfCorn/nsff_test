from argparse import ArgumentParser
import cv2
import copy
from collections import defaultdict
import imageio
import numpy as np
import os
import torch
from tqdm import tqdm

from models.rendering import render_rays, interpolate
from models.nerf import PosEmbedding, NeRF

from utils import load_ckpt, visualize_depth
import metrics
import third_party.lpips.lpips.lpips as lpips

from datasets.monocular import MonocularDataset

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_opts():
    parser = ArgumentParser()
    # basic 
    parser.add_argument('--root_dir', type=str, required=True, help='root directory of dataset')
    parser.add_argument('--ckpt_path', type=str, required=True, help='pretrained checkpoint path to load')
    parser.add_argument('--scene_name', type=str, default='test', help='scene name, used as output folder name')

    # PLEASE keep uniform setting as training, only the split option at will
    parser.add_argument('--split', type=str, default='test',help='refer to the dataset py for a detailed intro')

    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 288], help='resolution of the image')
    parser.add_argument('--start_end', nargs="+", type=int, default=[0, 30], help='start frame and end frame')
    parser.add_argument('--N_emb_xyz', type=int, default=10, help='number of features in xyz embedding')
    parser.add_argument('--N_emb_dir', type=int, default=4, help='number of features in dir embedding')
    parser.add_argument('--N_samples', type=int, default=128, help='number of nerf samples')
    parser.add_argument('--N_tau', type=int, default=48, help='number of embeddings for transient objects')
    parser.add_argument('--flow_scale', type=float, default=0.2, help='flow scale to multiply to flow network output')
    parser.add_argument('--chunk', type=int, default=32*1024, help='chunk size to split the input to avoid OOM')

    # save option
    parser.add_argument('--fps', type=int, default=10, help='video frame per second')
    parser.add_argument('--save_depth', default=True, action="store_true", help='whether to save depth prediction')
    parser.add_argument('--cal_test', default=False, action="store_true", help='calculation for split test')

    return parser.parse_args()

@torch.no_grad()
def f(models,       # {'fine': nerf_fine}
      embeddings,   # {'xyz','dir','t'}
      rays,         # (HW,6)
      ts,           # (HW)
      max_t,        # #frame-1 
      N_samples,    # 128 as training or other
      chunk,        # chunk size
      **kwargs):    # {'K': dataset.K, 'dataset': dataset, 'output_transient': True,'output_transient_flow'} 
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    kwargs_ = copy.deepcopy(kwargs)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk],
                        max_t,
                        N_samples,
                        0,  # dont perturb
                        0,  # dont noise
                        chunk,
                        test_time=True,
                        **kwargs_)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

# depths += [save_depth(depth_pred, h, w, dir_name, f'depth_{i}_{int(dt*100):03d}.png')]
def save_depth(depth, h, w, dir_name, filename):
    depth_pred = np.nan_to_num(depth.view(h, w).numpy()) # (H,W)
    depth_pred_img = visualize_depth(torch.from_numpy(depth_pred)).permute(1, 2, 0).numpy() # (H,W,3)
    depth_pred_img = (depth_pred_img*255).astype(np.uint8)
    imageio.imwrite(os.path.join(dir_name, filename), depth_pred_img)
    return depth_pred_img

if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    # load dataset and inference kwargs ------------------------------------------------------
    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': (w, h),
              'start_end': tuple(args.start_end)}
    dataset = MonocularDataset(**kwargs)
    kwargs = {'K': dataset.K, 'dataset': dataset}  # K (3,3) intrinsic

    # fix view, changing time, only case here requires fw,bw output
    # we also may not need interp only go without interpY
    if args.split.startswith('test_fixview') and int(args.split.split('_')[-1][6:])>0:
        # this takes too much memo
        #kwargs['output_transient_flow'] = ['fw', 'bw']
        kwargs['output_transient_flow'] = []
    else:
        kwargs['output_transient_flow'] = []

    # load model and embeddings --------------------------------------------------------------
    embedding_t = torch.nn.Embedding(dataset.N_frames, args.N_tau).to(device)
    embeddings = {'xyz': PosEmbedding(args.N_emb_xyz),
                  'dir': PosEmbedding(args.N_emb_dir),
                  't': embedding_t}
    load_ckpt(embedding_t, args.ckpt_path, 'embedding_t')

    nerf_fine = NeRF(typ='fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     in_channels_t=args.N_tau,
                     output_flow=len(kwargs['output_transient_flow'])>0,
                     flow_scale=args.flow_scale).to(device)
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    models = {'fine': nerf_fine}

    # setting output dir ---------------------------------------------------------------------
    dir_name = f'results/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    # rendering ------------------------------------------------------------------------------
    # get clear with different split:
    # test: ALL pos, no pos interp, no time interp.  will be evaluation there if test
    # test_spiral:  ALL pos, pos interp, no time interp (hard time pos)
    # test_spiralX: Single pos, pos interp, no time interp (fix time)
    # test_fixviewX_interpY: Single pos, no pos interp, time interp

    # TODO create a ALL pos fix time version
    # the time interp is ok that only for single pos, it's too large memo assumption
    # for both time-pos interp, that's why test_spiral choose hard time for whole
    # For a more detailed discussion, pls refer to my github page.


    imgs, depths = [], []

    # metrics are only for test
    if args.split == 'test' and args.cal_test:
        psnrs = np.zeros((dataset.N_frames, 2))
        ssims = np.zeros((dataset.N_frames, 2))
        lpipss = np.zeros((dataset.N_frames, 2))
        lpips_model = lpips.LPIPS(net='alex', spatial=True)

    # last_results is only for test_fixview
    if args.split.startswith('test_fixview'):
        last_results = None
        interp = int(args.split.split('_')[-1][6:]) # interp num

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        # sample:
            # rays          (HW,6)
            # ts            (HW)
            # c2w           (3,4)
            # cam_ids       0
            # rgbs  (@t)    (HW,3)
            # disp  (@t)    (HW)
            # mask  (@t)    (HW)
            # flow_fw (@t)  (HW,2)
            # flow_bw (@t)  (HW,2)

        ts = sample['ts'].to(device)
        rays = sample['rays'].to(device)

        # test_fixview case, only one needs to interp
        if args.split.startswith('test_fixview'):

            if i==len(dataset)-1: # last frame, cant interp with next one.
                img_pred = torch.clip(last_results['rgb_fine'].view(h, w, 3), 0, 1)
                img_pred_ = (255*img_pred.numpy()).astype(np.uint8)
                imgs += [img_pred_]
                imageio.imwrite(os.path.join(dir_name, f'{i:03d}_{int(0):03d}.png'), img_pred_)
                if args.save_depth:
                    depths += [save_depth(last_results['depth_fine'], h, w, dir_name, f'depth_{i:03d}_{int(0):03d}.png')]
                break

            if last_results is None: # first frame
                results = f(models, embeddings, rays, ts, dataset.N_frames-1, args.N_samples, args.chunk, **kwargs)
            else: results = last_results # middle frame

            results_tp1 = f(models, embeddings, rays, ts+1, dataset.N_frames-1, args.N_samples, args.chunk, **kwargs)
            last_results = results_tp1
            # if first frame (last_results is None), render this one and render next one, also take next one as last_results
            # if middle frame, directly take last_results as cur one, no need to render again, render next one as last_results

            # interpolation
            for dt in np.linspace(0, 1, interp+1)[:-1]: # interp images, interp must bigger than 0
                if dt == 0:     # cur frame
                    img_pred = results['rgb_fine'].view(h, w, 3)    # (H,W,3)
                    depth_pred = results['depth_fine'] # (HW)
                else:
                    # the K, c2w never changes!  depth output actually  (H,W)
                    img_pred, depth_pred = interpolate(results, results_tp1, dt, dataset.Ks[sample['cam_ids']], sample['c2w'], (w, h))
                    # depth (H,W)

                img_pred = torch.clip(img_pred, 0, 1)
                img_pred_ = (255*img_pred.numpy()).astype(np.uint8)

                # save
                imgs += [img_pred_]
                imageio.imwrite(os.path.join(dir_name, f'{i}_{int(dt*100):03d}.png'), img_pred_)
                if args.save_depth:
                    depths += [save_depth(depth_pred, h, w, dir_name, f'depth_{i}_{int(dt*100):03d}.png')]

        # all other case, no need to interp case
        else:
            results = f(models, embeddings, rays, ts, dataset.N_frames-1, args.N_samples, args.chunk, **kwargs)

            img_pred = torch.clip(results['rgb_fine'].view(h, w, 3), 0, 1)
            img_pred_ = (255*img_pred.numpy()).astype(np.uint8)
            
            # save
            imgs += [img_pred_]
            imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
            if args.save_depth:
                depths += [save_depth(results['depth_fine'], h, w, dir_name, f'depth_{i:03d}.png')]

            # additional save
            # only static
            temp = torch.clip(results['_static_rgb_fine'].view(h, w, 3), 0, 1)
            temp = (255*temp.numpy()).astype(np.uint8)
            imageio.imwrite(os.path.join(dir_name, f'static_rgb_{i:03d}.png'), temp)
            save_depth(results['_static_depth_fine'], h, w, dir_name, f'static_depth_{i:03d}.png')


        if args.split == 'test' and args.cal_test:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs[i, 0] = metrics.psnr(img_gt, img_pred).item()
            ssims[i, 0] = metrics.ssim(img_gt, img_pred).item()
            lpipss[i, 0] = metrics.lpips(lpips_model, img_gt, img_pred).item()
            if 'mask' in sample:
                mask = sample['mask'].view(h, w)
                psnrs[i, 1] = metrics.psnr(img_gt, img_pred, mask==0).item()
                ssims[i, 1] = metrics.ssim(img_gt, img_pred, mask==0).item()
                lpipss[i, 1] = metrics.lpips(lpips_model, img_gt, img_pred, mask==0).item()

    if args.split == 'test' and args.cal_test:
        mean_psnr = np.nanmean(psnrs, 0)
        mean_ssim = np.nanmean(ssims, 0)
        mean_lpips = np.nanmean(lpipss, 0)

        np.save(os.path.join(dir_name, 'psnr.npy'), psnrs)
        np.save(os.path.join(dir_name, 'ssim.npy'), ssims)
        np.save(os.path.join(dir_name, 'lpips.npy'), lpipss)

        print(f'Score \t Whole image  \t Dynamic only')
        print(f'-------------------------------------')
        print(f'PSNR  \t {mean_psnr[0]:.4f} \t {mean_psnr[1]:.4f}')
        print(f'SSIM  \t {mean_ssim[0]:.4f} \t {mean_ssim[1]:.4f}')
        print(f'LPIPS \t {mean_lpips[0]:.4f} \t {mean_lpips[1]:.4f}')

    # save gif
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=args.fps)
    if args.save_depth:
        imageio.mimsave(os.path.join(dir_name, f'depth_{args.scene_name}.gif'), depths, fps=args.fps)