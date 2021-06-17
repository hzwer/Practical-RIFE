# Practical-RIFE
This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and aims to make RIFE more practical for users by adding various features and design new models. Because improving the PSNR index is not compatible with subjective effects, we hope this part of work and our academic research are independent of each other. To reduce development difficulty, this project is for engineers and developers. For users, we recommend the following softwares:
[Squirrel-RIFE(中文软件)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | [Flowframes](https://nmkd.itch.io/flowframes) | [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | [RIFE-App(Paid)](https://grisk.itch.io/rife-app) | [Autodesk Flame](https://vimeo.com/505942142) | [SVP](https://www.svp-team.com/wiki/RIFE_AI_interpolation) |

For business cooperation, please [contact my email](mailto:huangzhewei@megvii.com).

16X interpolation results from two input images: 

![Demo](./demo/I0_slomo_clipped.gif)
![Demo](./demo/I2_slomo_clipped.gif)

## Usage

### Model List
v3.6 | [Google Drive](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing) | [百度网盘, 密码:hfk3](https://pan.baidu.com/s/1u6Q7-i4Hu4Vx9_5BJibPPA)

v3.5 | [Google Drive](https://drive.google.com/file/d/1YEi5KAdo0e6XnCTcbzOGCNtU33Lc2yO2/view?usp=sharing) | [百度网盘, 密码:1rb7](https://pan.baidu.com/s/1FqMcoIbYDV-Oq_ogcuuHjQ)

[Update log](https://github.com/hzwer/arXiv2020-RIFE/issues/164)

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

## Collection
**2d Animation**
[DAIN-App vs RIFE-App](https://www.youtube.com/watch?v=0OXzVGLRhK0) | [Chika Dance](https://www.youtube.com/watch?v=yjMYefRXikI) | [御坂大哥想让我表白 - 魔女之旅](https://www.bilibili.com/video/BV1r54y1Y7fn) | [ablyh - 超电磁炮](https://www.bilibili.com/video/BV1gK4y1Q7d9?from=search&seid=16584204362417247463) [超电磁炮.b](https://www.bilibili.com/video/BV1sp4y1p73K?from=search&seid=7774996509988438677) | [赫萝与罗伦斯的旅途 - 绫波丽](https://www.bilibili.com/video/BV1yz4y1m7iF) | [花儿不哭 - 乐正绫](https://www.bilibili.com/video/BV1cs411b7qL?from=search&seid=11977379973861203602) |

[没有鼠鼠的雏子Official - 千恋万花](https://www.bilibili.com/video/BV1AT4y1P7kY?from=search&seid=15458655842150253738) | [晨曦光晖 - 从零开始的异世界生活](https://www.bilibili.com/video/BV1QV411i7B4?from=search&seid=151780224584608151) | [琴乃乃 - 天才麻将少女](https://www.bilibili.com/video/BV1Qz4y1y7tH) |

**3d Animation**
[没有鼠鼠的雏子Official - 原神](https://www.bilibili.com/video/BV1iU4y1s7Lk) | [今天我练出腹肌了吗 - 最终幻想](https://www.bilibili.com/video/BV1R541177qr) [仙剑奇侠传](https://www.bilibili.com/video/BV14p4y1s7na) | [娜不列颠 - 冰雪奇缘](https://www.bilibili.com/video/BV1fy4y1J7Mu) | [索尼克释放：刺猬之夜](https://www.bilibili.com/video/BV1fo4y127hG?from=search&seid=14012843036741636664)

**MV and Film**
[Navetek - 邓丽君](https://www.bilibili.com/video/BV1ZK411u7CM) | [生米阿怪 - 周深](https://www.bilibili.com/video/BV1cp4y1W717) | [EzioAuditoreDFirenZe - 中森明菜](https://www.bilibili.com/video/BV18K4y1T7L6?from=search&seid=14012843036741636664) | [Dragostea Din Tei](https://www.bilibili.com/video/BV14B4y1A7Gr?from=search&seid=14012843036741636664) | [Life in a Day 2020](https://www.youtube.com/user/lifeinaday) |

**MMD**
[深邃黑暗の银鳕鱼 - 镜音铃](https://www.bilibili.com/video/BV1nU4y1W7RF?from=search&seid=151780224584608151) [fufu](https://www.bilibili.com/video/BV16K4y1Q7CM) [fufu.b](https://www.bilibili.com/video/BV1Xb4y1R7iT) | [Abism0 - 弱音](https://www.bilibili.com/video/BV1Wf4y147cP?from=search&seid=7774996509988438677) |

## Report Bad Cases
Please share your original video clip with us via Github issue and Google Drive. We may add it to our test set so that it is likely to be improved in later versions. It will be beneficial to attach a screenshot of the model's effect on the issue.

## Model training
Since we are in the research stage of engineering tricks, and our work and paper have not been authorized for patents nor published, we are sorry that we cannot provide users with training scripts. If you are interested in academic exploration, please refer to our academic research project [RIFE](https://github.com/hzwer/arXiv2020-RIFE). 

## To-do List
Multi-frame input of the model

Frame interpolation at any time location

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
