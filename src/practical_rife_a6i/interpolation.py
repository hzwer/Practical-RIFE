"""Tensor-in / tensor-out video frame interpolation for Practical-RIFE.

This module turns the inference loop of ``inference_video.py`` into an
importable API:

>>> from practical_rife_a6i.interpolation import interpolate_video
>>> out = interpolate_video(video, exp=1)      # torch.Tensor in, torch.Tensor out

For repeated calls, keep the model loaded with :class:`RifeVideoInterpolator`:

>>> engine = RifeVideoInterpolator(scale=1.0)
>>> out = engine(video, exp=2)

Input
-----
A 4-D video tensor (or numpy array) in one of two layouts:

* ``(T, C, H, W)`` -- torch convention (default when ``layout="auto"``)
* ``(T, H, W, C)`` -- numpy/decoded-video convention

``C`` must be 3 (RGB). ``uint8`` tensors are interpreted as 0..255, floating
point tensors as 0..1. ``exp`` is the number of extra frames per input frame:
``multi = 2 ** exp``, so ``exp=1`` doubles the frame rate, ``exp=2``
quadruples it, and ``exp=0`` returns the input unchanged.

Output
------
The interpolated video, ``(T - 1) * 2 ** exp + 1`` frames long, in the same
layout as the input by default. ``uint8`` input yields ``uint8`` output;
floating point input yields ``float32`` in 0..1. The result is returned on
CPU unless ``output_device`` is given.

Inference details are copied from ``inference_video.py``: zero padding to
``max(128, 128 / scale)``, static-frame collapsing through a look-ahead frame
(SSIM > 0.996), duplication instead of hallucination on hard cuts
(SSIM < 0.2), and per-timestep inference for v3.9+ models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ``model`` and ``train_log`` live at the repository root, while this module
# is installed from ``src``. Make the checkout importable when the package is
# used as an editable install from an arbitrary working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.pytorch_msssim import create_window_3d, ssim_matlab  # noqa: E402
from train_log.RIFE_HDv3 import Model  # noqa: E402

__all__ = [
    "RifeVideoInterpolator",
    "interpolate_video",
    "load_rife_model",
    "make_inference",
    "numpy_frame_to_tensor",
    "pad_image",
    "ssim_score",
]

_DEFAULT_MODEL_DIR = _REPO_ROOT / "train_log"
_VALID_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
_STATIC_SSIM = 0.996
_CUT_SSIM = 0.2
_VIDEO_LAYOUTS = ("TCHW", "THWC")


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_model_dir(model_dir) -> Path:
    return Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR


def _normalize_layout(layout: str) -> str:
    layout = layout.upper()
    if layout not in _VIDEO_LAYOUTS:
        raise ValueError(f"layout must be one of {_VIDEO_LAYOUTS}, got {layout!r}")
    return layout


def _resolve_layout(shape, layout: str) -> str:
    if layout is not None and layout.upper() != "AUTO":
        return _normalize_layout(layout)
    if shape[1] == 3 and shape[3] != 3:
        return "TCHW"
    if shape[3] == 3:
        return "THWC"
    if shape[1] == 3:
        return "TCHW"
    raise ValueError(
        "could not infer the channel dimension; pass layout='TCHW' or "
        f"layout='THWC' explicitly (got shape {tuple(shape)})"
    )


def _compute_padding(height: int, width: int, scale: float) -> tuple[int, int, int, int]:
    """Padding used by ``inference_video.py`` to make dimensions model-friendly."""
    tmp = max(128, int(128 / scale))
    padded_height = ((height - 1) // tmp + 1) * tmp
    padded_width = ((width - 1) // tmp + 1) * tmp
    return (0, padded_width - width, 0, padded_height - height)


def load_rife_model(model_dir=None, fp16: bool = False, device=None):
    """Load the RIFE HDv3 model the same way ``inference_video.py`` does.

    Returns ``(model, device)``. ``model_dir`` defaults to the checkout's
    ``train_log`` directory.
    """
    target_device = _resolve_device(device)
    if fp16 and target_device.type != "cuda":
        raise ValueError("fp16 inference requires a CUDA device")

    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model(str(_resolve_model_dir(model_dir)), -1)
    model.eval()
    model.flownet.to(target_device)
    if fp16:
        model.flownet.half()
    return model, target_device


def pad_image(img: torch.Tensor, padding, fp16: bool = False) -> torch.Tensor:
    """Pad a BCHW tensor; casts to half when ``fp16`` is enabled."""
    padded = F.pad(img, padding)
    return padded.half() if fp16 else padded


def numpy_frame_to_tensor(frame, device, padding, fp16: bool = False) -> torch.Tensor:
    """Convert an HWC RGB ``uint8`` frame to a padded BCHW float tensor."""
    img = (
        torch.tensor(np.transpose(frame, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    return pad_image(img, padding, fp16)


_SSIM_WINDOWS: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _ssim_window(size: int, channel: int, device: torch.device) -> torch.Tensor:
    key = (size, channel, device)
    window = _SSIM_WINDOWS.get(key)
    if window is None:
        # Build the window in float32 explicitly: the caller may have changed
        # the global default dtype (inference_video.py's fp16 mode used to).
        window = create_window_3d(size, channel).float().to(device)
        _SSIM_WINDOWS[key] = window
    return window


def ssim_score(I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
    """SSIM on 32x32 RGB thumbnails, matching ``inference_video.py``."""
    I0_small = F.interpolate(I0.float(), (32, 32), mode="bilinear", align_corners=False)
    I1_small = F.interpolate(I1.float(), (32, 32), mode="bilinear", align_corners=False)
    height, width = I0_small.shape[-2:]
    size = min(11, height, width)
    window = _ssim_window(size, 1, I0_small.device)
    return ssim_matlab(
        I0_small[:, :3], I1_small[:, :3], window_size=size, window=window
    )


def make_inference(model, I0: torch.Tensor, I1: torch.Tensor, n: int, scale: float = 1.0):
    """Return ``n`` intermediate BCHW frames between ``I0`` and ``I1``.

    v3.9+ models infer every timestep directly; older models use the recursive
    midpoint strategy from ``inference_video.py``.
    """
    if getattr(model, "version", 0) >= 3.9:
        return [
            model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), scale)
            for i in range(n)
        ]

    middle = model.inference(I0, I1, scale)
    if n == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, n // 2, scale)
    second_half = make_inference(model, middle, I1, n // 2, scale)
    if n % 2:
        return [*first_half, middle, *second_half]
    return [*first_half, *second_half]


class RifeVideoInterpolator:
    """Load once, interpolate many videos.

    Parameters
    ----------
    model_dir:
        Directory containing ``flownet.pkl``. Defaults to ``train_log`` in the
        repository checkout.
    scale:
        Inference scale, one of 0.25, 0.5, 1.0, 2.0, 4.0. Use 0.5 for 4K input.
    fp16:
        Half precision inference (CUDA only).
    device:
        Torch device, e.g. ``"cuda"`` or ``"cpu"``. Auto-detected by default.
    """

    def __init__(self, model_dir=None, scale: float = 1.0, fp16: bool = False, device=None):
        if scale not in _VALID_SCALES:
            raise ValueError(f"scale must be one of {_VALID_SCALES}, got {scale!r}")
        self.scale = float(scale)
        self.fp16 = bool(fp16)
        self.model, self.device = load_rife_model(
            model_dir, fp16=self.fp16, device=device
        )

    def __call__(self, video: Tensor, exp: int, **kwargs) -> torch.Tensor:
        return self.interpolate(video, exp, **kwargs)

    def interpolate(
        self,
        video: Tensor,
        exp: int,
        *,
        layout: str = "auto",
        output_layout: str | None = None,
        output_device=None,
    ) -> torch.Tensor:
        """Interpolate ``video`` with ``2 ** exp`` frames per input frame.

        Parameters
        ----------
        video:
            4-D ``(T, C, H, W)`` or ``(T, H, W, C)`` tensor/ndarray. ``uint8``
            values are 0..255, float values are 0..1.
        exp:
            Non-negative integer; the frame-rate multiplier is ``2 ** exp``.
        layout:
            ``"auto"`` (default), ``"TCHW"`` or ``"THWC"``.
        output_layout:
            Layout of the returned tensor. ``None``/``"same"`` (default) keeps
            the input layout.
        output_device:
            Device for the returned tensor. Defaults to CPU.

        Returns
        -------
        torch.Tensor
            ``(T - 1) * 2 ** exp + 1`` frames in the requested layout/device.
        """
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        if not isinstance(video, torch.Tensor):
            raise TypeError("video must be a torch.Tensor or numpy.ndarray")
        if video.ndim != 4:
            raise ValueError(
                "video must be 4-D (T,C,H,W) or (T,H,W,C), "
                f"got shape {tuple(video.shape)}"
            )
        if isinstance(exp, bool) or not isinstance(exp, int) or exp < 0:
            raise ValueError(f"exp must be a non-negative integer, got {exp!r}")
        if video.dtype != torch.uint8 and not video.is_floating_point():
            raise TypeError(
                f"video must be uint8 or floating point, got {video.dtype}"
            )

        layout = _resolve_layout(video.shape, layout)
        if layout == "TCHW":
            total, channels, height, width = video.shape
        else:
            total, height, width, channels = video.shape
        if total < 1:
            raise ValueError("video must contain at least one frame")
        if channels != 3:
            raise ValueError(f"video must have 3 (RGB) channels, got {channels}")

        if output_layout is None or str(output_layout).lower() == "same":
            resolved_output_layout = layout
        else:
            resolved_output_layout = _normalize_layout(output_layout)

        multi = 2 ** exp
        padding = _compute_padding(height, width, self.scale)
        to_uint8 = video.dtype == torch.uint8

        def frame_to_chw(index: int) -> torch.Tensor:
            frame = video[index]
            if layout == "THWC":
                frame = frame.permute(2, 0, 1)
            if to_uint8:
                frame = frame.float() / 255.0
            else:
                frame = frame.float()
            return frame.to(self.device, non_blocking=True)

        def to_model_input(frame_chw: torch.Tensor) -> torch.Tensor:
            return pad_image(frame_chw.unsqueeze(0), padding, self.fp16)

        def crop(frame: torch.Tensor) -> torch.Tensor:
            frame = frame[0] if frame.ndim == 4 else frame
            return frame[:, :height, :width]

        def export(frame: torch.Tensor) -> torch.Tensor:
            frame = frame.detach().to("cpu", torch.float32)
            if to_uint8:
                frame = frame.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
            return frame

        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            lastframe = frame_to_chw(0)
            I1 = to_model_input(lastframe)
            index = 1
            temp = None

            while True:
                if temp is not None:
                    frame = temp
                    temp = None
                else:
                    if index >= total:
                        break
                    frame = frame_to_chw(index)
                    index += 1

                I0 = I1
                I1 = to_model_input(frame)
                ssim = ssim_score(I0, I1)

                break_flag = False
                if ssim > _STATIC_SSIM:
                    # Nearly identical frames: look one frame ahead and skip
                    # the redundant one, exactly like inference_video.py.
                    if index >= total:
                        break_flag = True
                        frame = lastframe
                    else:
                        frame = frame_to_chw(index)
                        index += 1
                        temp = frame
                    I1 = to_model_input(frame)
                    I1 = self.model.inference(I0, I1, scale=self.scale)
                    ssim = ssim_score(I0, I1)
                    frame = crop(I1)

                if ssim < _CUT_SSIM:
                    # Scene cut: repeat the previous frame rather than blending
                    # unrelated content.
                    output = [I0] * (multi - 1)
                else:
                    output = make_inference(self.model, I0, I1, multi - 1, self.scale)

                outputs.append(export(lastframe))
                for mid in output:
                    outputs.append(export(crop(mid)))
                lastframe = frame

                if break_flag:
                    break

            outputs.append(export(lastframe))

        result = torch.stack(outputs, 0)
        if resolved_output_layout == "THWC":
            result = result.permute(0, 2, 3, 1).contiguous()

        output_device = torch.device(output_device) if output_device is not None else torch.device("cpu")
        if result.device != output_device:
            result = result.to(output_device)
        return result


def interpolate_video(
    video: Tensor,
    exp: int,
    *,
    model_dir=None,
    scale: float = 1.0,
    fp16: bool = False,
    device=None,
    layout: str = "auto",
    output_layout: str | None = None,
    output_device=None,
) -> torch.Tensor:
    """One-shot convenience wrapper around :class:`RifeVideoInterpolator`.

    Loads the model, interpolates ``video`` and returns the result tensor. If
    you process several videos, construct :class:`RifeVideoInterpolator` once
    instead to avoid reloading the weights.
    """
    engine = RifeVideoInterpolator(
        model_dir=model_dir, scale=scale, fp16=fp16, device=device
    )
    return engine.interpolate(
        video,
        exp,
        layout=layout,
        output_layout=output_layout,
        output_device=output_device,
    )
