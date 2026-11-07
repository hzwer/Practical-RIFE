import subprocess

import numpy as np
np.float = np.float64
np.int = np.int32
import os
import cv2
import torch
import argparse
from tqdm import tqdm
import _thread
import skvideo.io
from queue import Queue
from practical_rife_a6i.interpolation import (
    load_rife_model,
    make_inference,
    numpy_frame_to_tensor,
    ssim_score,
)

def transfer_audio(source_video, target_video: str):
    import shutil
    temp_audio_file_name = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        # os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(source_video, temp_audio_file_name))
        subprocess.run(
            ["ffmpeg", "-y", "-i", source_video, "-c:a", "copy", "-vn", temp_audio_file_name],
            check=True,
        )
    target_no_audio = os.path.splitext(target_video)[0] + "_noaudio" + os.path.splitext(target_video)[1]
    os.rename(target_video, target_no_audio)
    # combine audio file and new video file
    # os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(target_no_audio, temp_audio_file_name, target_video))
    subprocess.run(
        ["ffmpeg", "-y", "-i", target_no_audio, "-i", temp_audio_file_name, "-c", "copy", target_video],
        check=True,
    )
    if os.path.getsize(target_video) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        temp_audio_file_name = "./temp/audio.m4a"
        # os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(source_video, temp_audio_file_name))
        # os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(target_no_audio, temp_audio_file_name, target_video))

        # 1. 从原视频提取音频，转 aac 160k
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", source_video,
                "-c:a", "aac", "-b:a", "160k",
                "-vn",
                temp_audio_file_name,
            ],
            check=True,
        )
        # 2. 把无声视频和提取出的音频合并
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", target_no_audio,
                "-i", temp_audio_file_name,
                "-c", "copy",
                target_video,
            ],
            check=True,
        )
        if os.path.getsize(target_video) == 0: # if aac is not supported by selected format
            os.rename(target_no_audio, target_video)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(target_no_audio)
    else:
        os.remove(target_no_audio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default=None, help='directory with trained model files (default: bundled train_log)')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # fp16 is applied to the model and padded inputs by load_rife_model /
    # pad_image, so the process-wide default dtype/device stays untouched.

model, device = load_rife_model(args.modelDir, fp16=args.fp16, device=device)
print("Loaded 3.x/4.x HD model.")

fpsNotAssigned = False
fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
video_path_wo_ext = ''
if args.video is not None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    # fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    assert args.img is not None
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape
vid_out_name = ''
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

left = w // 4 if args.montage else 0
if args.montage:
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = numpy_frame_to_tensor(lastframe, device, padding, args.fp16)
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = numpy_frame_to_tensor(frame, device, padding, args.fp16)
    ssim = ssim_score(I0, I1)

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = numpy_frame_to_tensor(frame, device, padding, args.fp16)
        I1 = model.inference(I0, I1, scale=args.scale)
        ssim = ssim_score(I0, I1)
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
    if ssim < 0.2:
        output = []
        for i in range(args.multi - 1):
            output.append(I0)
        '''
        output = []
        step = 1 / args.multi
        alpha = 0
        for i in range(args.multi - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(model, I0, I1, args.multi - 1, args.scale)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transfer_audio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
