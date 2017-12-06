# Structure-Aware and Temporally Coherent 3D Human Pose Estimation

This is the demo code for the paper Structure-Aware and Temporally Coherent 3D Human Pose Estimation
Project page: tinyurl.com/pose-iitb
arXiv link: https://arxiv.org/abs/1711.09250v1

Parts of this code have been taken from [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train) and [Towards 3D pose in the wild](https://github.com/xingyizhou/pose-hg-3d)

## Requirements
- cudnn
- [Torch7](https://github.com/torch/torch7) with hdf5 and image
- Python with h5py and opencv2

## Running the code
- Place the unit-pose-net.t7 and time-pose-net.t7 models in 'models/' and the image sequence in 'data/'
- Run `th main.lua'

## Options
 - Skeleton fitting is turned on be default. You can turn that off using 'skelFit' option
 - Results (csv file) will be stored in 'results/{exp}'. 
 - Every 20th frame is displayed by default. You can change that using the 'display' option.
 

We provide an example sequence of images in `data` from Human3.6M dataset. For testing your own video, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

Please drop an email to rdabral@cse.iitb.ac.in to request access to the models.

