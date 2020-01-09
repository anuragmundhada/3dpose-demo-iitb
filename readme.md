# Learning 3D Human Pose from Structure and Motion

This is the demo code for the paper [Learning 3D Human Pose from Structure and Motion](https://www.cse.iitb.ac.in/~rdabral/3DPose/)

Parts of this code have been taken from [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train) and [Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://github.com/xingyizhou/pose-hg-3d).

## Requirements
- cudnn
- [Torch7](https://github.com/torch/torch7) with hdf5, csvigo and image packages
- Python with h5py and opencv2

## Running the code
- Download the [models](https://drive.google.com/open?id=1eCdwb5lrakmHo79YLaKbXOV39tTkJSzy) and place them in 'models/' and the test image sequence in 'data/'
- Run `th main.lua'

## Options
 - Skeleton fitting is turned on be default. You can turn that off using 'skelFit' option
 - Results (csv file) will be stored in 'results/{exp}'. 
 - Every 20th frame is displayed by default. You can change that using the 'display' option.
 

We provide an example sequence of images in 'data/' from Human3.6M dataset. For testing your own video, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Reference
```
@InProceedings{Dabral_2018_ECCV,
author = {Dabral, Rishabh and Mundhada, Anurag and Kusupati, Uday and Afaque, Safeer and Sharma, Abhishek and Jain, Arjun},
title = {Learning 3D Human Pose from Structure and Motion},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
} 
```
