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
A second round optimization. several bug fixed, full notation for everything. Much more readable. I also optimized some unnecessary settings with basically same result. BTW, A hand-writing notes for nsff paper is also updated just for your interest.. (though this is not an authentic implementation!)

### 3. RECOMMENDATIONS FOR YOUR DATASET
I didn't get much information for how to select your original data, and I did run into problems with my custom datasets.. Here I pave the way:   

1. At least 30 frames, as recommended anywhere. This is not necessarily leading to a higher-frame-number-better-result, but longer training time of course.

2. Question - Can I try intricate dynamic/movements? My answer: If your dataset contains very complicated movements from objects, I recommend you to increase the frame number, because the flow captures the dynamic anyway. But this actually leads to a trade-off given a budget frame number: Can I add more dynamic parts for my scene, or just increase the continuity for my current course of time?

3. Question - What's the requirement for continuity between frames? My answer: If you export frames from a video with a budget training frame number, you may be in a dilemma that: should I have a long-duration-low-frame-rate training, or short-duration-high-frame-rate training? The extreme of the former is the model captures nothing at all, it sees all frames as no continuity. The extreme of the latter is the model degrades to a pure nerf model, or worse (single-image train). From my experience, I usually limit the scene duration first as: don't contain too many movements for reconstructions if budgets are low, and shorten the duration even more if the static background is not 'static' (camera roaming). Then evenly distribute the frames with the duration.   

4. UPDATED 11/29 : more about the dataset. Training data is decisive. If you are extracting frames from videos (instead of taking pictures), please take care of the camera position and dynamic objects. If you have continuous frames with __relatively unmoved camera__ with __dynamic objects do not move a lot__, this over smoothness will result in that the model considers that certain part as background. While dramatic dynamics in your dataset should be avoided for sure, do not take frames having ill camera & object relation as described either. As my result suffers from this problem.

### 4. A SUMMARY OF RENDERING WITH NSFF   
Nsff provides additional dimensions for neural radiance rendering. The original project page did not provide a thorough discussion of extension of renderings. Though trivial, a summary table may help you better understand what’s going on about this model with a better visualization results.

![Picture2](https://user-images.githubusercontent.com/68495667/202875722-47436ef0-a4fa-455e-8d23-fbb6f2de7048.png)

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
The discussion above just covers every case of rendering, though with all kinds of different names elsewhere. Basically you always match that somewhere on this figure.   
My apologies that some evaluation settings are implicit still, but I make it all clear in the `.py` files. 

## RESULTS_1 (rendering for kid_run)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202946233-e6bcef93-3630-4c4a-a5c5-68f0f6a40609.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202946269-cc399870-fc43-4449-937e-4bede993803c.gif", width="40%">
  <br>
  <sup>test on training frames (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202981065-702f8d3c-0365-46a2-8a6e-3168193b337f.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202980843-96e5f32c-cda3-474c-918b-2ef883e07426.gif", width="40%">
  <br>
  <sup>test on training frames, background (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202981301-10923a91-2185-4df8-802d-eda29627f75c.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202981357-f8f70154-20d6-444d-98fb-e9fa2fd2f81c.gif", width="40%">
  <br>
  <sup>spiral interp on one position (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202981602-1fdebca4-76db-4b3a-8cbd-36d2a8ed572e.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202981521-840340ac-d1d4-48cb-8155-1a0288fa29a5.gif", width="40%">
  <br>
  <sup>spiral interp on one position, background (right: depth)</sup>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202981750-12eb5742-a6f7-4b5b-b726-488d0bb47d81.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202981805-e8e6828d-db33-4e67-965e-41d150e00a85.gif", width="40%">
  <br>
  <sup>fix view no interp (right: depth)</sup>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202982325-84feb57e-24ce-43c9-84b5-8b6eaa8343d8.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202982525-582a74b5-1a96-4cae-bf15-1b024d7eaee0.gif", width="40%">
  <br>
  <sup>fix view with 4 frames interp in each between (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202982912-eb78d512-1c02-4f7b-b219-c697eb30de5f.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202983033-e48eb916-f37b-4d75-b781-c2e8b09a6b0f.gif", width="40%">
  <br>
  <sup>smooth pos interp with roaming time on whole set (right: depth)</sup>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202983279-498b8547-4eb2-4e0a-92b8-738076f2580c.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202983287-ab1df55f-edf2-4ac2-952d-9e0fd6c612d5.gif", width="40%">
  <br>
  <sup>smooth pos interp with fix time on whole set (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202983705-f6866ec0-8ef1-48c6-a448-358df4bd3664.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202983710-6b9c6535-8c99-4ef7-a3f4-41ba89deaa41.gif", width="40%">
  <br>
  <sup>spiral pos interp with roaming time on whole set (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202983969-33876b00-6218-4693-b1b7-55ac599e46be.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202983980-452c3944-b19e-4bfd-a706-a0e7a07460c2.gif", width="40%">
  <br>
  <sup>spiral pos interp with fix time on whole set (right: depth)</sup>
</p>

## RESULTS_1 (tf board for kid_run)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202994672-51d00a87-c7c3-40ef-b5fc-1770cf4c32f4.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202994685-1585e117-2733-4ffd-9a7d-9a434d3b7111.png", width="40%">
  <br>
  <sup>train</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202995045-509a9f38-2d17-470a-b4c8-345bfc5b8be3.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202995060-f62f19fb-06ea-49c1-8faa-48200f371032.png", width="40%">
  <br>
  <sup>validation</sup>
</p>


## RESULTS_2 (rendering for girl_dog)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202984781-9f23e8f4-4ddd-423a-8b32-ec920c389dcd.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202984792-fb4e5a5b-0464-4612-b92a-2524e5732072.gif", width="40%">
  <br>
  <sup>test on training frames (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202985245-765ceceb-2ef1-4fb0-8e82-1ae3b9b72f73.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202985267-21f0ad18-cc90-46a5-930e-58ff905f4c27.gif", width="40%">
  <br>
  <sup>test on training frames, background (right: depth)</sup>
</p>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202985533-620ac809-2342-4d93-bfec-9f3f7eba4df6.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202985542-10ba82a3-2a63-42ea-9b8a-fbd05698827c.gif", width="40%">
  <br>
  <sup>spiral interp on one position (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202985975-3806a008-4377-4f39-9798-325d645959b1.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202985959-da07e852-5931-43b2-8fba-c15a3bb97511.gif", width="40%">
  <br>
  <sup>spiral interp on one position, background (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202988403-b94ac96b-26b0-4bf1-88ea-efcdef050920.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202988374-baa265a1-8a67-4388-886c-f89c9cb02b9f.gif", width="40%">
  <br>
  <sup>fix view with 4 frames interp in each between (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202988947-7853e57f-7e49-412b-8956-c7f2481ead53.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202988953-670a8fcc-422b-47ba-8bcd-09e98590c859.gif", width="40%">
  <br>
  <sup>smooth pos interp with fix time on whole set (right: depth)</sup>
</p>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202989413-cc3b69d1-630a-4c71-b2d0-0982548e307a.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202989439-ec3dc039-fb89-4742-b847-44cb6f1a8489.gif", width="40%">
  <br>
  <sup>smooth pos interp with fix time on whole set, background (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202989821-d8285d68-20b8-40bf-b3f3-3fa13a308299.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202989826-d6f8a0de-1104-4f84-8ea6-01c08e0cc938.gif", width="40%">
  <br>
  <sup>smooth pos interp with roaming time on whole set (right: depth)</sup>
</p>

## RESULTS_2 (tf board for girl_dog)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202996210-15dd9496-e6cf-4ffa-b6ac-c0ca6250b66e.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202996219-6d04bc93-abbc-40dd-9106-b3d3c3ff8452.png", width="40%">
  <br>
  <sup>train</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202996276-afdd9e33-ebac-4dd0-be54-b91dbe0e19f1.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202996288-d15ebf69-cb69-4b43-901c-01db37dc4233.png", width="40%">
  <br>
  <sup>validation</sup>
</p>

## RESULTS_3 (rendering for woman_dog)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202990546-bc4870c6-9625-4822-a39c-4f8796a99b54.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202990550-76dd26d8-727b-4be5-9c7f-a86ba0190419.gif", width="40%">
  <br>
  <sup>test on training frames (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202991096-8ef46d90-34fc-49f7-a2f2-e5163abd19aa.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202991077-622cbf45-20e2-43a2-bdf3-1e678154a3a2.gif", width="40%">
  <br>
  <sup>test on training frames, background (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202991361-326d90d4-5127-48fa-892a-44199d75f9a3.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202991397-ad5b32d9-c6a3-4e03-8ecf-56b8cd49cb0e.gif", width="40%">
  <br>
  <sup>spiral interp on one position (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202991795-26e1d302-5e7d-4f5f-8f18-9ee880bbe801.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202991814-6f9ae774-1e6d-4ae6-b0a1-08736f28b90f.gif", width="40%">
  <br>
  <sup>spiral interp on one position, background (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202992289-559e5f7c-c215-4b5f-9a98-da8d58b4fb87.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202992302-aa6cac15-b52e-492a-beb5-b5a26431989f.gif", width="40%">
  <br>
  <sup>fix view with 4 frames interp in each between (right: depth)</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202992818-119b35bc-6eb4-4d86-822e-f188b43422b1.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202992824-e3563f13-c21d-4034-aac7-2280092b1a4d.gif", width="40%">
  <br>
  <sup>smooth pos interp with roaming time on whole set (right: depth)</sup>
</p>

## RESULTS_3 (tf board for woman_dog)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202996746-c5065d39-c2ad-48d7-8197-8e41cc422eeb.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202996754-8681f21b-7161-44da-89a5-e25d4a6d2e31.png", width="40%">
  <br>
  <sup>train</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202996802-cbc33b05-12f9-4b7f-97c8-b34f7982db30.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202996810-28060052-170d-4aa3-b1d4-5b5813b05641.png", width="40%">
  <br>
  <sup>validation</sup>
</p>

## RESULTS_4 (rendering for car) (failed case)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202993234-381256f2-82fe-4f83-b983-e303214395bf.gif", width="40%">
  <br>
  <sup>spiral (right: depth)</sup>
</p>

## RESULTS_4 (tf board for car) (failed case)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202997231-f402dae2-e72a-48ba-afcb-58d44ae12f7e.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202997240-596c4016-40de-4595-ad98-f2928f78edd7.png", width="40%">
  <br>
  <sup>train</sup>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68495667/202997298-66cf9ac1-2f0a-4b1f-a455-fab9ca19df4e.png", width="40%">
  <img src="https://user-images.githubusercontent.com/68495667/202997311-fd3f2a81-494b-472d-8d5e-3c04734838bf.png", width="40%">
  <br>
  <sup>validation</sup>
</p>
