# NSFF IMPLEMENTATION for UCSD_CSE274_PROJ
##### By Mohan Li, EMAIL: mol002@ucsd.edu   

## FOR MILESTONE2
This part is for the requirements of project milestone for CSE274.   
- Currently only some preliminary results as below, sorry for that code is not organized yet.
- For next step, I would like to:   
Try to figure out the dynamic distortion (shown in the video below)   
More training on different scene   
More testing on different settings (fix time/view/neither, calibrate camera path etc.)     
- others   
This will be my final project presentation. Nerf/mipnerf will be part of my paper presentation instead.


#### before start
> This is a neural scene flow fields implementation based on torch/torchlightning. This implementation could be seen as a fork of [nsff_pl](https://github.com/kwea123/nsff_pl) and [nerf_mipnerf](https://github.com/TheSoulOfCorn/nerf_mipnerf_test). Updates/optimizations will be detailed as follow.   
> Hardware: WSL2 on Windows 11, RTX 3070. Memory consumption is less than 10 GB using my default setting.

## RUNNING NSFF
### 1. create environment
> run   
`conda create -n nsff python=3.7` then `conda activate nsff`   
`pip install -r requirements.txt`   
better check the sanity of the GPU status before going next step, as I developed in WSL2 so the environment setting may be not as universal.

### 2. prepare your data   
This repo follows the recommended steps of [nsff_pl](https://github.com/kwea123/nsff_pl). Three borrowed tools are as following:   

Detectron2: we use [COLMAP](https://github.com/colmap/colmap) to generate camera intrinsic and extrinsic, assuming the scene is static at most. This is not true for dynamic scenes, and we may want to filter out the dynamic parts first and then do the [COLMAP](https://github.com/colmap/colmap) job. We use maskrcnn from [detectron2](https://github.com/facebookresearch/detectron2), which already has pretrained model for our task. I have notes in the code illustrating each step.   
> install detectron2 `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html`.   
> you also need to install [COLMAP](https://github.com/colmap/colmap).

RAFT: NSFF training requires guidance of 2D optical flows, we use [RAFT](https://github.com/princeton-vl/RAFT) here.   
> Download [raft-things.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and put it in `third_party/flow/models/`.   

DPT: NSFF training requires guidance of predicted depth, we use [DPT](https://github.com/intel-isl/DPT) here.   
> Download the model [weights](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and put it in `third_party/depth/weights/`.   

Following the original repo, all the data preparation steps are summarized to the `preprocess.py`. user-friendly notes are attached to this version. Please prepare your data as following steps. Refer to my notations if any confusion.   

> create your training dataset root directory (`test`), then create a folder named `frames`, put your original training images in `frames`. (the images should be named  time-orderly)   

> according to your own dataset, modify the `DYNAMIC_CATEGORIES` variable in `third_party/predict_mask.py` to the dynamic classes in your data.

> `python preprocess.py --root_dir <path/to/your_root_dir>`   

Warning in this step:   

I do not know why sometimes it's not creating full training directory, the final dir should have several folders for your data, you may doublecheck the `preprocess.py` as I noted, or `README.md` from [nsff_pl](https://github.com/kwea123/nsff_pl) to ensure your training set is fully built. After masking, the COLMAP sometimes cannot build all cameras. This is lethal and a problem I ran into..

Based on my own implementation, I may have you some recommendations about the dataset preparation in the discussion part. I HIGHLY RECOMMEND you to read them first to avoid wasting time on training nonsense.

### 3. train   

Check the `opt.py` for options for training, modify the `quick_train.sh`, and `bash quick_train.sh`.   
the checkpoints will be saved in `./ckpts`, tfevents file in `./logs`.

### 4. evaluation

Check the `eval.py` for options for evaluation, modify the `quick_eval.sh`, and `bash quick_eval.sh`.   
please make sure your evaluation settings are in harmony with your training (e.g. Don't have a 60 pos_embedding for training while trying to recover a 30 pos_embedding model). See the discussion part for options in the evaluation.


## DISCUSSION
### 1. WHAT'S NEW TO THE PAPER?   
Refer to [nsff_pl](https://github.com/kwea123/nsff_pl) for a first round optimization based on the original paper implementation. I followed a few of that and it's working pretty well.

### 2. WHAT'S NEW TO THE [NSFF_PL](https://github.com/kwea123/nsff_pl) REPO?   
A second round optimization. several bug fixed, full notation for everything. Much more readable. I also optimized some unnecessary settings with basically same result. BTW, A hand-writing notes for nsff paper is also updated just for your interest..

### 3. RECOMMENDATIONS FOR YOUR DATASET
I didn't get much information for how to select your original data, and I did run into problems with my custom datasets.. Here I pave the way:   

1. At least 30 frames, as recommended anywhere. This is not necessarily leading to a higher-frame-number-better-result, but longer training time of course.

2. Question - Can I try intricate dynamic/movements? My answer: If your dataset contains very complicated movements from objects, I recommend you to increase the frame number, because the flow captures the dynamic anyway. But this actually leads to a trade-off given a budget frame number: Can I add more dynamic parts for my scene, or just increase the continuity for my current course of time?

3. Question - What's the requirement for continuity between frames? My answer: If you export frames from a video with a budget training frame number, you may be in a dilemma that: should I have a long-duration-low-frame-rate training, or short-duration-high-frame-rate training? The extreme of the former is the model captures nothing at all, it sees all frames as no continuity. The extreme of the latter is the model degrades to a pure nerf model, or worse (single-image train). From my experience, I usually limit the scene duration first as: don't contain too many movements for reconstructions if budgets are low, and shorten the duration even more if the static background is not 'static' (camera roaming). Then evenly distribute the frames with the duration. This procedure works just fine for my implementation. I didn't dig into any other details about this. If someone knows any systematic discussions/publications of it please let me know! thanks.   

### 4. A SUMMARY OF RENDERING WITH NSFF   
Nsff provides additional dimensions for neural radiance rendering. The original project page did not provide a thorough discussion of extension of renderings. Though trivial, a summary table may help you better understand what’s going on about this model with a better visualization results.

![Picture1](https://user-images.githubusercontent.com/68495667/202873024-fcc04e7e-454b-4d36-8ade-b20e2dc9e571.png)

This figure includes almost all cases of rendering (not discussing visualization of other (middle) outputs like model predicted flow).   
Focusing on the position number/position interp/time, we discuss all different situations here.   

##### __【dead case】__ Single-pose-no-interp with fixed-time, there is nothing changing!   
##### __【case 1】__ Single-pose-no-interp with varying-time and rendering full dyn+static, this is working in the repo with data split method `test_fixviewX_interpY`, basically we fix the view as input number X, and interpolate Y-1 frames in between of two. Noticing that Y=0 outputs nothing, Y=1 outputs no interpolate results. A discussion of interpolation is below.   
##### __【dead case】__ Single-pose-no-interp with varying-time and rendering only static, the static is barely changing with time when position fixed.
##### __【dead case】__ Single-pose-smooth-interp, we cannot give a single position smooth interpolation.
##### __【case 2】__ Single-pose-spiral-interp with fixed-time and rendering full dyn+static, this is working in the repo with data split method `test_spiralX`, where X is the view you gonna fix
##### __【case 3】__ Single-pose-spiral-interp with fixed-time and rendering only static, this is working in the repo with data split method `test_spiralX`, where X is the view you gonna fix. This is sometimes called _background reconstruction_ in other papers or projects.
##### __【dead case】__ Single-pose-spiral-interp with varying-time, I believe some repos have tried this. The original video from the nsff website is doing something like this. But I don't consider this is a good demonstration/ this is covered by a full path rendering, we don't have to do this for a single view.
##### __【case 4】__ All-poses-no-interp, this is a pure testing with your original taining set, and it could be static (or combining dyn), this is working in the repo with data split method `test`.
##### __【case 5】__ All-poses-smooth-interp with fixed-time and rendering full dyn+static, this is working in the repo with data split method `test_spiral`. Noticing that you need to go into the `monocular.py` to decide which frame's time you want to fix on. This is very much as original nerf results. Just make the camera moving offset to 0.
##### __【case 6】__ All-poses-smooth-interp with fixed-time and rendering only static, this is working in the repo with data split method `test_spiral`. Noticing that you need to go into the `monocular.py` to decide which frame's time you want to fix on. This is very much as original nerf results, just make the camera moving offset to 0, but only for background. Again, background does not change as time if correct, so it's the same for varying-time case. (check the figure)
##### __【case 7】__ All-poses-smooth-interp with varying-time and rendering full dyn+static, this is working in the repo with data split method `test_spiral`. You need to go into the `monocular.py` to choose not fixing the time, and make the camera moving offset to 0. A discussion of interpolation related to this case is below.
##### __【case 8】__ All-poses-spiral-interp with fixed-time and rendering full dyn+static, this is working in the repo with data split method `test_spiral`. Noticing that you need to go into the `monocular.py` to decide which frame's time you want to fix on. This is very much as original nerf results.
##### __【case 9】__ All-poses-spiral-interp with fixed-time and rendering only static, this is working in the repo with data split method `test_spiral`. Noticing that you need to go into the `monocular.py` to decide which frame's time you want to fix on. This is very much as original nerf results, but only for background. Again, background does not change as time if correct, so it's the same for varying-time case. (check the figure)
##### __【case 10】__ All-poses-spiral-interp with varying-time and rendering full dyn+static, this is working in the repo with data split method `test_spiral`. You need to go into the `monocular.py` to choose not fixing the time. A discussion of interpolation related to this case is below.
##### __【case \*】__ As discussed, `interpY` decides if you really needs interpolation for the time, we borrow from the original repo for the splat rendering for this.
##### __【case \#】__ We currently have no time interpolations for this case, but only space-interp. A visually non-smooth-time training set will manifest itself regarding the rendering result: the camera is smoothly roaming, but the dynamic objects acutally not. This is soothed if your training set is essentially visually smooth. All in all, just for your information that the splat rendering is not arriving here.

We do not discuss rendering only dynamic here though we may. It gives a black/white or whatever background image output which is not realistic. You could definitely try! In the `rendering.py` file I already prepared all you need to know, an easy implementation to try by yourself.
