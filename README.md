# Practical-RIFE 
**[V4.0 Promotional Video (宣传视频）](https://www.bilibili.com/video/BV1J3411t7qT?p=1&share_medium=iphone&share_plat=ios&share_session_id=7AE3DA72-D05C-43A0-9838-E2A80885BD4E&share_source=QQ&share_tag=s_i&timestamp=1639643780&unique_k=rjqO0EK)**  

  **Use this project on Google Colab for free! Check out the [Practical-RIFE Colab Notebook](https://colab.research.google.com/drive/1BZmGSq15O4ZU5vPfzkv7jFNYahTm6qwT?usp=sharing).** 

2024.01 - We recently release new v4.7-4.14 models. In our tests, 4.14 makes a great improvement for animation scenes. 🎉

<img width="710" alt="image" src="https://github.com/hzwer/Practical-RIFE/assets/10103856/e692b6f0-ff1a-45f7-8cd7-2bd3e6919ec5">

This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and [SAFA](https://github.com/megvii-research/WACV2024-SAFA). We aim to make them more practical for users by adding various features and designing new models. Because improving the PSNR index is not compatible with subjective effects, we hope this part of work and our academic research are independent of each other. To reduce development pressure, this project is for engineers and developers. For common users, we recommend the following softwares:

**[SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes) | [Drop frame fixer and FPS converter](https://github.com/may-son/RIFE-FixDropFrames-and-ConvertFPS)**

Thanks to [SVFI team](https://github.com/Justin62628/Squirrel-RIFE) to support model testing on Animation. 

## Frame Interpolation
### Model List
The content of these links is under the same MIT license as this project. **lite** means using similar training framework, but lower computational cost model.

4.18 - 2024.07.03 | [Google Drive](https://drive.google.com/file/d/1octn-UVuEjXa_HlsIUbNeLTTvYCKbC_s/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1fqtxJyXSgUx-gE3rieuKxg?pwd=udr1)

4.17 - 2024.05.24 | [Google Drive](https://drive.google.com/file/d/1962p_lEWo_kLTEynarNaRYRNVdaiQG2k/view?usp=share_link) | [百度网盘](https://pan.baidu.com/s/1bMzTYoJKZXsoxuSBmzj6VQ?pwd=as37) : Add gram loss from [FILM](https://github.com/google-research/frame-interpolation/blob/69f8708f08e62c2edf46a27616a4bfcf083e2076/losses/vgg19_loss.py) || [4.17.lite](https://drive.google.com/file/d/1e9Qb4rm20UAsO7h9VILDwrpvTSHWWW8b/view?usp=share_link)

4.15 - 2024.03.11 | [Google Drive](https://drive.google.com/file/d/1xlem7cfKoMaiLzjoeum8KIQTYO-9iqG5/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1IGNIX7JXGUwI_tfoafYHqA?pwd=bg0b) || [4.15.lite](https://drive.google.com/file/d/1BoOF-qSEnTPDjpKG1sBTa6k7Sv5_-k7z/view?usp=sharing) || 4.14 - 2024.01.08 | [Google Drive](https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view?usp=share_link) | [百度网盘](https://pan.baidu.com/s/1d-W64lRsJTqNsgWoXYiaWQ?pwd=xawa) || [4.14.lite](https://drive.google.com/file/d/1eULia_onOtRXHMAW9VeDL8N2_7z8J1ba/view?usp=share_link)

4.13.1 - 2023.12.05 | [Google Drive](https://drive.google.com/file/d/1mj9lH6Be7ztYtHAr1xUUGT3hRtWJBy_5/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1e0I-ERSYQThANP7BQmz3Vw?pwd=e2h8) || [4.13.lite](https://drive.google.com/file/d/1l3lH9QxQQeZVWtBpdB22jgJ-0kmGvXra/view?usp=sharing) || v4.12.2 - 2023.11.13 | [Google Drive](https://drive.google.com/file/d/1ZHrOBL217ItwdpUBcBtRE3XBD-yy-g2S/view?usp=share_link) | [百度网盘](https://pan.baidu.com/s/1zyAw-qZJsIsAyFOIZKumYQ?pwd=gwij) 

v4.11.1 - 2023.11.11 | [Google Drive](https://drive.google.com/file/d/1Dwbp4qAeDVONPz2a10aC2a7-awD6TZvL/view?usp=share_link) | [百度网盘](https://pan.baidu.com/s/1TZiZuCHaG4SKmxKrqbbBDQ?pwd=pw2i) || v4.10.1 - 2023.11.09 [Google Drive](https://drive.google.com/file/d/1WNot1qYBt05LUyY1O9Uwwv5_K8U6t8_x/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/19niopL-Qwu2qOydIB1pEBg?pwd=88kv)

v4.9.2 - 2023.11.01 | [Google Drive](https://drive.google.com/file/d/1UssCvbL8N-ty0xIKM5G5ZTEgp9o4w3hp/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/18cbx3EP4HWgSa1vkcXvvyw?pwd=swr9) || v4.8.1 - 2023.10.23 | [Google Drive](https://drive.google.com/file/d/1wZa3SyegLPUwBQWmoDLM0MumWd2-ii63/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1hq-O3QY5OVzLHIQ90jKkyg?pwd=vhsq)

v4.7.1 - 2023.9.25 | [Google Drive](https://drive.google.com/file/d/1s2zMMIJrUAFLexktm1rWNhlIyOYJ3_ju/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1Mc7bvkVWMUG9F0Gqb_Vcpg?pwd=848u) || v4.6 - 2022.9.26 | [Google Drive](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1Oc1enSD7kGnoQda2MdPYsw?pwd=gtkf)

v4.3 - 2022.8.17 | [Google Drive](https://drive.google.com/file/d/1xrNofTGMHdt9sQv7-EOG0EChl8hZW_cU/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/12AUAeZLZf5E1_Zx6WkS3xw?pwd=q83a) || v4.2 - 2022.8.10 | [Google Drive](https://drive.google.com/file/d/1JpDAJPrtRJcrOZMMlvEJJ8MUanAkA-99/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1Io4Z_QUaBv-O7dYERqQAPw?pwd=y3ad) 

v3.8 - 2021.6.17 | [Google Drive](https://drive.google.com/file/d/1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1X-jpWBZWe-IQBoNAsxo2mA?pwd=kxr3) || v3.1 - 2021.5.17 | [Google Drive](https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1W4p_Ni04HLI_jTy45sVodA?pwd=64bz) 

[More Older Version](https://github.com/megvii-research/ECCV2022-RIFE/issues/41)

### Installation

```
git clone git@github.com:hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
```
Download a model from the model list and put *.py and flownet.pkl on train_log/
### Run

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
python3 inference_video.py --multi=4 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)

Parameter descriptions:

--img / --video: The input file address

--output: Output video name 'xxx.mp4'

--model: Directory with trained model files

--UHD: It is equivalent to setting scale=0.5

--montage: Splice the generated video with the original video, like [this demo](https://www.youtube.com/watch?v=kUQ7KK6MhHw)

--fps: Set output FPS manually

--ext: Set output video format, default: mp4

--multi: Interpolation frame rate multiplier

--exp: Set --multi to 2^(--exp)

--skip: It's no longer useful refer to [issue 207](https://github.com/hzwer/ECCV2022-RIFE/issues/207)


### Model training
The whole repo can be downloaded from [v4.0](https://drive.google.com/file/d/1zoSz7b8c6kUsnd4gYZ_6TrKxa7ghHJWW/view?usp=sharing), [v4.12](https://drive.google.com/file/d/1IHB35zhO4rr-JSMnpRvHhU9U65Z4giWv/view?usp=sharing), [v4.15](https://drive.google.com/file/d/19sUMZ-6H7g_hYDjTcqxYu9kE7TqnfS3k/view?usp=sharing). However, we currently do not have the time to organize it well, it is for reference only.

## Video Enhancement

<img width="710" alt="image" src="https://github.com/hzwer/Practical-RIFE/assets/10103856/5bae134c-0747-4084-bbab-37b1595352f1">

We are developing a practical model of [SAFA](https://github.com/megvii-research/WACV2024-SAFA). Welcome to check its [demo](https://www.youtube.com/watch?v=QII2KQSBBwk) ([BiliBili](https://www.bilibili.com/video/BV1Up4y1d7kF/)) and provide advice.

v0.5 - 2023.12.26 | [Google Drive](https://drive.google.com/file/d/1OLO9hLV97ZQ4uRV2-aQqgnwhbKMMt6TX/view?usp=sharing)

```
python3 inference_video_enhance.py --video=demo.mp4
```

## Citation

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
```
@inproceedings{huang2024safa,
  title={Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution},
  author={Huang, Zhewei and Huang, Ailin and Hu, Xiaotao and Hu, Chen and Xu, Jun and Zhou, Shuchang},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## Reference

Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI) [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
