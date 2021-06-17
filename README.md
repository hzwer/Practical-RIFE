# Practical-RIFE
This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and aims to make RIFE more practical for users by adding various features and design new models. Because improving the PSNR index is not compatible with subjective effects, we hope this part of work and our academic research are independent of each other. To reduce development difficulty, this project is for engineers and developers. For users, we recommend the following softwares:
[Squirrel-RIFE(中文软件)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | [Flowframes](https://nmkd.itch.io/flowframes) | [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | [RIFE-App(Paid)](https://grisk.itch.io/rife-app) | [Autodesk Flame](https://vimeo.com/505942142) | [SVP](https://www.svp-team.com/wiki/RIFE_AI_interpolation) |

For business cooperation, please [contact my email](mailto:huangzhewei@megvii.com).

16X interpolation results from two input images: 

![Demo](./demo/I0_slomo_clipped.gif)
![Demo](./demo/I2_slomo_clipped.gif)

## Model training
Since we are in the research stage of engineering tricks, and our work and paper have not been authorized for patents nor published, we are sorry that we cannot provide users with training scripts. If you are interested in academic exploration, please refer to our academic research project [RIFE](https://github.com/hzwer/arXiv2020-RIFE). 

## To-do List
Multi-frame input of the model

Frame interpolation at any time location

Eliminate artifacts as much as possible

Make the model applicable under any resolution input

Provide models with lower calculation consumption

## Report Bad Cases
Please share your original video clip with us via Github issue and Google Drive. We may add it to our test set so that it is likely to be improved in later versions. It will be beneficial to attach a screenshot of the model's effect on the issue.

## Usage

### Model List
v3.6 | [Google Drive](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing) | [百度网盘, 密码:75nd](https://pan.baidu.com/s/1L3LugphpmWIHE1EVkZm-1Q)

v3.5 | [Google Drive](https://drive.google.com/file/d/1YEi5KAdo0e6XnCTcbzOGCNtU33Lc2yO2/view?usp=sharing) | [百度网盘, 密码:1rb7](https://pan.baidu.com/s/1FqMcoIbYDV-Oq_ogcuuHjQ)

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
python3 inference_video.py --exp=1 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --exp=2 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5
```
(If your video has high resolution, such as 4K, we recommend set --scale=0.5 (default 1.0))
```
python3 inference_video.py --exp=2 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)

The warning info, 'Warning: Your video has *** static frames, it may change the duration of the generated video.' means that your video has changed the frame rate by adding static frames. It is common if you have processed 25FPS video to 30FPS.


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
