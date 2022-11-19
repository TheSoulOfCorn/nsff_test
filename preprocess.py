import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2

def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data for nsff training')

    parser.add_argument('--root_dir', type=str, help='a root dir, contains everything wrt. the data preprocessing', required=True)
    parser.add_argument('--cuda_device',type=str,default='0',help='cuda device to use')
    parser.add_argument('--overwrite', default=False,action='store_true', help='overwrite cache / Do preprocessing anyway')
    parser.add_argument('--max_width', type=int, default=1280, help='max image width of resizing frames')
    parser.add_argument('--max_height', type=int, default=720, help='max image height of resizing frames')

    # please hardcode your input folder name under root as 'frames'
    # the resized/renamed images will be under 'images_resized'
    # the undistorted images will be under 'images'

    args = parser.parse_args()
    return args

def resize_frames(args):
    vid_name = os.path.basename(args.root_dir)  # basename is the most right dir    # '/foo/bar/item' -> 'item'
    frames_dir = os.path.join(args.root_dir, 'images_resized')
    os.makedirs(frames_dir, exist_ok=True)

    files = sorted( # taking path of any jpg/png, careful about things like jpeg or upper case
        glob.glob(os.path.join(args.root_dir, 'frames', '*.jpg')) +
        glob.glob(os.path.join(args.root_dir, 'frames', '*.png')))

    print('[0/5] Resizing images ...')

    for file_ind, file in enumerate(tqdm(files, desc=f'imresize: {vid_name}')):
        out_frame_fn = f'{frames_dir}/{file_ind:05}.png'  # output file name is len=5 with stacking 0 left

        # skip if both the output frame and the mask exist
        if os.path.exists(out_frame_fn) and not args.overwrite:
            continue

        im = cv2.imread(file) # (H,W,3)

        # resize if too big
        # W <= maxW and H <= maxH
        if im.shape[1] > args.max_width or im.shape[0] > args.max_height:
            factor = max(im.shape[1] / args.max_width, im.shape[0] / args.max_height)
            dsize = (int(im.shape[1] / factor), int(im.shape[0] / factor))
            im = cv2.resize(src=im, dsize=dsize, interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_frame_fn, im)

    print('[1/5] Resizing images done.')
    
def generate_masks(args):
    # ugly hack, masks expects images in images, but undistorted ones are going there later
    # generate a 'masks' folder contains white background black motion obj mask images.
    undist_dir = os.path.join(args.root_dir, 'images')
    print("[1/5] Predicting motion masks ...")
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)

        os.system(f'cp -r {args.root_dir}/images_resized/*.png {args.root_dir}/images')
        # you need to modify the categories in the predict_make.py
        # TODO modify it as an input instead of directly modification..
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python third_party/predict_mask.py --root_dir {args.root_dir}')
        os.system(f'rm {args.root_dir}/images')     # deleted

    print("[2/5] Predicting motion masks done.")

def run_colmap(args):
    max_num_matches = 132768  # colmap setting
    print("[2/5] Running colmap ...")

    # colmap steps.. matching under masks
    if not os.path.exists(f'{args.root_dir}/database.db') or args.overwrite:
        os.system(f'''
                CUDA_VISIBLE_DEVICES={args.cuda_device} \
                colmap feature_extractor \
                --database_path={args.root_dir}/database.db \
                --image_path={args.root_dir}/images_resized\
                --ImageReader.mask_path={args.root_dir}/masks \
                --ImageReader.camera_model=SIMPLE_RADIAL \
                --ImageReader.single_camera=1 \
                --ImageReader.default_focal_length_factor=0.95 \
                --SiftExtraction.peak_threshold=0.004 \
                --SiftExtraction.max_num_features=8192 \
                --SiftExtraction.edge_threshold=16
                ''')

        os.system(f'''
                CUDA_VISIBLE_DEVICES={args.cuda_device} \
                colmap exhaustive_matcher \
                --database_path={args.root_dir}/database.db \
                --SiftMatching.multiple_models=1 \
                --SiftMatching.max_ratio=0.8 \
                --SiftMatching.max_error=4.0 \
                --SiftMatching.max_distance=0.7 \
                --SiftMatching.max_num_matches={max_num_matches}
                ''')

    if not os.path.exists(f'{args.root_dir}/sparse') or args.overwrite:
        os.makedirs(os.path.join(args.root_dir, 'sparse'), exist_ok=True)
        os.system(f'''
                CUDA_VISIBLE_DEVICES={args.cuda_device} \
                colmap mapper \
                    --database_path={args.root_dir}/database.db \
                    --image_path={args.root_dir}/images_resized \
                    --output_path={args.root_dir}/sparse 
                ''')

    undist_dir = os.path.join(args.root_dir, 'images') # this func gives dir 'images' itself
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)
        os.system(f'''          
                CUDA_VISIBLE_DEVICES={args.cuda_device} \
                colmap image_undistorter \
                    --input_path={args.root_dir}/sparse/0 \
                    --image_path={args.root_dir}/images_resized \
                    --output_path={args.root_dir} \
                    --output_type=COLMAP
                ''')

    print("[3/5] Running colmap done.")

def generate_depth(args):

    print("[3/5] Generating depth ...")

    disp_dir = os.path.join(args.root_dir, 'disps')
    if not os.path.exists(disp_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/depth')   # go to this dir
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run_monodepth.py -i {args.root_dir}/images -o {args.root_dir}/disps -t dpt_large')
        os.chdir(f'{str(cur_dir)}')     # back to current dir

    print("[4/5] Generating depth done.")

def generate_flow(args):
    # outputs .flo files
    print("[4/5] Generating optical flow ...")

    flow_fw_dir = os.path.join(args.root_dir, 'flow_fw')
    flow_bw_dir = os.path.join(args.root_dir, 'flow_bw')
    if not os.path.exists(flow_fw_dir) or not os.path.exists(flow_bw_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/flow')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python demo.py --model models/raft-things.pth --path {args.root_dir}')
        os.chdir(f'{str(cur_dir)}') 

    print("[5/5] Generating optical flow done.")

if __name__ == '__main__':
    args = parse_args()

    resize_frames(args)
    # for COLMAP recon
    generate_masks(args)
    run_colmap(args)
    # for training loss
    generate_depth(args)
    generate_flow(args)

    print('All finished.')