# Practical-RIFE
**[V4.0 Promotional Video (宣传视频）](https://www.bilibili.com/video/BV1J3411t7qT?p=1&share_medium=iphone&share_plat=ios&share_session_id=7AE3DA72-D05C-43A0-9838-E2A80885BD4E&share_source=QQ&share_tag=s_i&timestamp=1639643780&unique_k=rjqO0EK)**

This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and aims to make RIFE more practical for users by adding various features and design new models. Because improving the PSNR index is not compatible with subjective effects, we hope this part of work and our academic research are independent of each other. To reduce development difficulty, this project is for engineers and developers. For users, we recommend the following softwares:

**[RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes) | [SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation)**

For business cooperation, please [contact my email](mailto:huangzhewei@megvii.com).

## Usage
### Model List
v4.0 - 2021.12.6 | [Google Drive](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view?usp=sharing) | [百度网盘，密码:mocg](https://pan.baidu.com/s/1NEqyqTNS4uj2TbTultZ3_g) || v3.9beta - 2021.11.23 | [Google Drive](https://drive.google.com/file/d/1iosmPTt2ayAdSMqnI1cxO_R1-Qhrranp/view?usp=sharing) | [百度网盘, 密码:4nrl](https://pan.baidu.com/s/1dO-Yh9KY9gypa7baABADCA) 

v3.8 - 2021.6.17 | [Google Drive](https://drive.google.com/file/d/1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj/view?usp=sharing) | [百度网盘, 密码:kxr3](https://pan.baidu.com/s/1X-jpWBZWe-IQBoNAsxo2mA) || v3.5 - 2021.6.12 | [Google Drive](https://drive.google.com/file/d/1YEi5KAdo0e6XnCTcbzOGCNtU33Lc2yO2/view?usp=sharing) | [百度网盘, 密码:1rb7](https://pan.baidu.com/s/1FqMcoIbYDV-Oq_ogcuuHjQ)

v3.1 - 2021.5.17 | [Google Drive](https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view?usp=sharing) | [百度网盘, 密码:64bz](https://pan.baidu.com/s/1W4p_Ni04HLI_jTy45sVodA) || v3.0 - 2021.5.15 | [Google Drive](https://drive.google.com/file/d/1JmwH8L3pdy49NroCVwracDW5UM43AAqd/view?usp=sharing) | [百度网盘, 密码:tgmd](https://pan.baidu.com/s/1_cC0sUdi6qx9tOkagddslw)

### Installation

```
git clone git@github.com:hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
```
Download a model from the model list and put *.py and flownet.pkl on train_log/
### Run

**Video Frame Interpolation**

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or your video. 
```
python3 inference_video.py --multi=2 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --multi=4 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --multi=2 --video=video.mp4 --scale=0.5
```
(If your video has high resolution, such as 4K, we recommend set --scale=0.5 (default 1.0))
```
python3 inference_video.py ---multi=4 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --multi=3 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)

## Report Bad Cases
Please share your original video clip with us via Github issue and Google Drive. We may add it to our test set so that it is likely to be improved in later versions. It will be beneficial to attach a screenshot of the model's effect on the issue.

## Model training
Since we are in the research stage of engineering tricks, and our work and paper have not been authorized for patents nor published, we are sorry that we cannot provide users with training scripts. If you are interested in academic exploration, please refer to our academic research project [RIFE](https://github.com/hzwer/arXiv2020-RIFE). 

## To-do List
Multi-frame input of the model

Frame interpolation at any time location (Done)

Eliminate artifacts as much as possible

Make the model applicable under any resolution input

Provide models with lower calculation consumption

## Citation

```
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```

## Reference

Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI) [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
