import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from practical_rife.pytorch_msssim import ssim_matlab
from practical_rife.train_log.RIFE_HDv3 import Model
from practical_rife.utils import transferAudio
from pathlib import Path

warnings.filterwarnings("ignore")


def run_upsample(video_path: str, output_path: str, interpolate_multiplier: int, device="cuda"):

    model = Model(device)
    if not hasattr(model, "version"):
        model.version = 0

    dir_path = os.path.dirname(__file__)
    model_path = Path(dir_path) / "train_log"

    model.load_model(model_path, -1)
    print("Loaded 3.x/4.x HD model.")
    model.eval()
    model.device(device)

    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()

    fpsNotAssigned = True
    target_fps = fps * interpolate_multiplier

    videogen = skvideo.io.vreader(video_path)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_path_wo_ext, ext = os.path.splitext(video_path)
    print(
        "{}.{}, {} frames in total, {}FPS to {}FPS".format(
            video_path_wo_ext, "mp4", tot_frame, fps, target_fps
        )
    )

    h, w, _ = lastframe.shape
    vid_out = None

    scale = 1.0

    vid_out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    tmp = 128
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)

    def clear_write_buffer(write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            vid_out.write(item[:, :, ::-1])

    def build_read_buffer(read_buffer, videogen):
        try:
            for frame in videogen:
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(I0, I1, n):
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), scale))
        return res

    def pad_image(img):
        return F.pad(img, padding)

    write_buffer = Queue(maxsize=500)

    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (write_buffer,))

    I1 = (
        torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    I1 = pad_image(I1)
    temp = None  # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get()  # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = (
                torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                .to(device, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, 0.5, scale)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range(interpolate_multiplier - 1):
                output.append(I0)
            """
            output = []
            step = 1 / args.multi
            alpha = 0
            for i in range(args.multi - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            """
        else:
            output = make_inference(I0, I1, interpolate_multiplier - 1)

        write_buffer.put(lastframe)
        for mid in output:
            mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    write_buffer.put(lastframe)
    import time

    while not write_buffer.empty():
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()

    transferAudio(video_path, output_path)


if __name__ == "__main__":
    run_upsample(
        "/home/terrance/Practical-RIFE/test_vids/test2.mp4",
        "/home/terrance/Practical-RIFE/test_vids/test2_OUTPUTX4.mp4",
        4,
    )
