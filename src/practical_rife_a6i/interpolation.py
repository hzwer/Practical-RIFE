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

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from practical_rife_a6i.model.pytorch_msssim import create_window_3d, ssim_matlab
from practical_rife_a6i.train_log.RIFE_HDv3 import Model

__all__ = [
	"RifeVideoInterpolator",
	"interpolate_video",
	"load_rife_model",
	"make_inference",
	"numpy_frame_to_tensor",
	"pad_image",
	"ssim_score",
]

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _PACKAGE_DIR / "train_log"
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

	Returns ``(model, device)``. ``model_dir`` defaults to the ``train_log``
	directory bundled inside the installed package.
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
	if n <= 0:
		return []
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


def _load_skvideo(ffmpeg: Path):
	"""Import scikit-video configured to use ``ffmpeg``.

	scikit-video 1.1.11 still relies on the ``np.int`` / ``np.float`` aliases
	that NumPy 2 removed, and it locates the binaries through a directory
	rather than an executable path. It also copies ``_FFMPEG_PATH`` and
	``_HAS_FFMPEG`` into its submodules at import time, so those bound copies
	are refreshed after ``setFFmpegPath`` to make the explicitly provided
	executable the one that actually runs.
	"""
	if not hasattr(np, "float"):
		np.float = np.float64  # type: ignore[attr-defined]
	if not hasattr(np, "int"):
		np.int = np.int32  # type: ignore[attr-defined]
	import importlib

	import skvideo
	import skvideo.io

	ffmpeg_module = importlib.import_module("skvideo.io.ffmpeg")
	ffprobe_module = importlib.import_module("skvideo.io.ffprobe")
	io_module = importlib.import_module("skvideo.io.io")

	directory = Path(ffmpeg).parent
	ffprobe_path = directory / skvideo._FFPROBE_APPLICATION
	if not ffprobe_path.is_file():
		raise RuntimeError(
			f"ffprobe must be installed next to ffmpeg: expected {ffprobe_path}"
		)

	skvideo.setFFmpegPath(str(directory))
	for module in (skvideo, ffmpeg_module, ffprobe_module, io_module):
		if hasattr(module, "_FFMPEG_PATH"):
			module._FFMPEG_PATH = str(directory)
		if hasattr(module, "_HAS_FFMPEG"):
			module._HAS_FFMPEG = 1
	return skvideo.io


def _transfer_audio(source: Path, target: Path, temp_directory: Path, ffmpeg: Path) -> None:
	"""Mux the audio track of ``source`` into ``target`` in place.

	``target`` is left as a valid silent video if muxing fails; a
	``RuntimeWarning`` is emitted in that case.
	"""
	import subprocess
	import uuid

	ffmpeg = str(ffmpeg)
	source = Path(source)
	target = Path(target)
	temp_directory = Path(temp_directory)
	created_directory = not temp_directory.exists()
	temp_directory.mkdir(parents=True, exist_ok=True)

	token = uuid.uuid4().hex[:8]
	audio_path = temp_directory / f"rife_audio_{token}.mkv"
	aac_path = temp_directory / f"rife_audio_{token}.m4a"
	no_audio_path = target.with_name(f"{target.stem}_noaudio{target.suffix}")
	target_moved = False
	merged = False

	try:
		subprocess.run(
			[ffmpeg, "-y", "-i", str(source), "-c:a", "copy", "-vn", str(audio_path)],
			check=True,
		)
		target.replace(no_audio_path)
		target_moved = True
		subprocess.run(
			[
				ffmpeg, "-y",
				"-i", str(no_audio_path),
				"-i", str(audio_path),
				"-c", "copy",
				str(target),
			],
			check=True,
		)
		merged = target.exists() and target.stat().st_size > 0

		if not merged:
			# The container does not support the source codec: transcode to AAC.
			subprocess.run(
				[
					ffmpeg, "-y",
					"-i", str(source),
					"-c:a", "aac", "-b:a", "160k",
					"-vn",
					str(aac_path),
				],
				check=True,
			)
			subprocess.run(
				[
					ffmpeg, "-y",
					"-i", str(no_audio_path),
					"-i", str(aac_path),
					"-c", "copy",
					str(target),
				],
				check=True,
			)
			merged = target.exists() and target.stat().st_size > 0

		if not merged:
			target.unlink(missing_ok=True)
			no_audio_path.replace(target)
			target_moved = False
			warnings.warn(
				"Audio transfer failed; the interpolated video has no audio.",
				RuntimeWarning,
				stacklevel=2,
			)
			return
	finally:
		if target_moved and not merged and no_audio_path.exists():
			target.unlink(missing_ok=True)
			no_audio_path.replace(target)
		elif no_audio_path.exists():
			no_audio_path.unlink(missing_ok=True)
		audio_path.unlink(missing_ok=True)
		aac_path.unlink(missing_ok=True)
		if created_directory:
			try:
				temp_directory.rmdir()
			except OSError:
				pass


class RifeVideoInterpolator:
	"""Load once, interpolate many videos.

	Parameters
	----------
	model_dir:
		Directory containing ``flownet.pkl``. Defaults to the ``train_log``
		directory bundled inside the installed package.
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
		return self.interpolate_in_tensor(video, exp, **kwargs)

	def interpolate_in_tensor(
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

	def interpolate_in_file(
			self,
			video: Path,
			temp_directory: Path,
			exp: int,
			output: Path,
			ffmpeg: Path,
			*,
			fps: float | None = None,
			progress_fn: Callable[[int, int], None] | None = None,
	) -> Path:
		"""Interpolate a video file and write the result to another video file.

		Parameters
		----------
		video:
			Input video path, decoded with FFmpeg through scikit-video.
		temp_directory:
			Scratch directory for audio extraction. Only touched when audio is
			merged, i.e. when ``fps`` is ``None``.
		exp:
			Non-negative integer; the output frame rate is multiplied by
			``2 ** exp``.
		output:
			Output video path. Must be a :class:`pathlib.Path`; the parent
			directory must already exist.
		ffmpeg:
			Path to the FFmpeg executable. ``ffprobe`` must be installed in
			the same directory, because scikit-video locates both binaries by
			directory.
		fps:
			Output frame rate. ``None`` (default) derives
			``source_fps * 2 ** exp`` and merges the source audio; an explicit
			value disables audio merging, matching ``inference_video.py
			--fps``.
		progress:
			Show a tqdm progress bar.

		Returns
		-------
		pathlib.Path
			The written video path.

		Raises
		------
		FileNotFoundError
			If ``video`` or ``ffmpeg`` does not exist.
		TypeError
			If ``output`` or ``ffmpeg`` is not a :class:`pathlib.Path`.
		ValueError
			If ``exp`` is invalid, the source FPS cannot be determined while
			``fps`` is ``None``, or ``output`` equals ``video``.
		RuntimeError
			If the video cannot be decoded or the writer cannot be opened.

		Notes
		-----
		The per-frame algorithm matches ``inference_video.py``: static frames
		are collapsed through a look-ahead frame (SSIM > 0.996) and scene cuts
		repeat the previous frame (SSIM < 0.2). As a result the output may
		contain fewer than ``(T - 1) * 2 ** exp + 1`` frames.
		"""
		import cv2

		if isinstance(exp, bool) or not isinstance(exp, int) or exp < 0:
			raise ValueError(f"exp must be a non-negative integer, got {exp!r}")
		if not isinstance(output, Path):
			raise TypeError(
				f"output must be a pathlib.Path, got {type(output).__name__}"
			)
		if not isinstance(ffmpeg, Path):
			raise TypeError(
				f"ffmpeg must be a pathlib.Path, got {type(ffmpeg).__name__}"
			)

		video = Path(video)
		if not video.is_file():
			raise FileNotFoundError(f"video not found: {video}")
		if not ffmpeg.is_file():
			raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")

		multi = 2 ** exp
		capture = cv2.VideoCapture(str(video))
		try:
			if not capture.isOpened():
				raise RuntimeError(f"could not open video: {video}")
			source_fps = float(capture.get(cv2.CAP_PROP_FPS))
			source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		finally:
			capture.release()

		merge_audio = fps is None
		if merge_audio:
			if not np.isfinite(source_fps) or source_fps <= 0:
				raise ValueError(
					f"could not determine a valid FPS for {video} "
					f"(got {source_fps!r}); pass fps= to skip audio merging"
				)
			output_fps = source_fps * multi
		else:
			output_fps = float(fps)
		if not np.isfinite(output_fps) or output_fps <= 0:
			raise ValueError(f"fps must be positive and finite, got {output_fps!r}")

		if output.resolve() == video.resolve():
			raise ValueError("output path must differ from the input video")

		working_path = output.with_suffix(f".part{output.suffix}")
		video_io = _load_skvideo(ffmpeg)
		reader = video_io.vreader(str(video))
		writer = None
		# pbar = None
		completed = False

		try:
			try:
				first_frame = next(reader, None)
			except AssertionError as exc:
				raise RuntimeError(
					"ffmpeg and ffprobe must be installed next to the given "
					f"ffmpeg executable: {ffmpeg}"
				) from exc
			if first_frame is None:
				raise RuntimeError(f"no frames could be decoded from: {video}")

			height, width = first_frame.shape[:2]
			padding = _compute_padding(height, width, self.scale)
			fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
			writer = cv2.VideoWriter(
				str(working_path), fourcc, output_fps, (width, height)
			)
			if not writer.isOpened():
				raise RuntimeError(
					f"could not open video writer for {output} "
					f"(fps={output_fps}, size={width}x{height})"
				)

			# if progress:
			#     from tqdm import tqdm
			#
			#     pbar = tqdm(
			#         total=source_frames if source_frames > 0 else None,
			#         unit="frame",
			#         desc=output.name,
			#     )

			def to_tensor(frame):
				return numpy_frame_to_tensor(frame, self.device, padding, self.fp16)

			def to_rgb(frame_tensor: torch.Tensor):
				return (
					frame_tensor[0]
					.mul(255.0)
					.byte()
					.cpu()
					.numpy()
					.transpose(1, 2, 0)[:height, :width]
				)

			def write_frame(frame_rgb):
				writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

			with torch.inference_mode():
				lastframe = first_frame
				I1 = to_tensor(first_frame)
				pending = None

				frame_cnt = 0
				while True:
					if pending is not None:
						frame, pending = pending, None
					else:
						frame = next(reader, None)
						if frame is None:
							break

					I0 = I1
					I1 = to_tensor(frame)
					ssim = ssim_score(I0, I1)

					break_flag = False
					if ssim > _STATIC_SSIM:
						lookahead = next(reader, None)
						if lookahead is None:
							break_flag = True
							frame = lastframe
						else:
							frame = lookahead
							pending = lookahead
						I1 = self.model.inference(
							I0, to_tensor(frame), scale=self.scale
						)
						ssim = ssim_score(I0, I1)
						frame = to_rgb(I1)

					if ssim < _CUT_SSIM:
						mids = [I0] * (multi - 1)
					else:
						mids = make_inference(
							self.model, I0, I1, multi - 1, self.scale
						)

					write_frame(lastframe)
					for mid in mids:
						write_frame(to_rgb(mid))
					# if pbar is not None:
					#     pbar.update(1)

					if progress_fn is not None:
						frame_cnt += 1
						progress_fn(frame_cnt, source_frames)

					lastframe = frame
					if break_flag:
						break

				write_frame(lastframe)
				# if pbar is not None:
				#     pbar.update(1)
				if progress_fn is not None:
					frame_cnt += 1
					progress_fn(frame_cnt, source_frames)

			completed = True
		finally:
			if writer is not None:
				writer.release()
			# if pbar is not None:
			# 	pbar.close()
			close = getattr(reader, "close", None)
			if close is not None:
				close()
			if not completed:
				working_path.unlink(missing_ok=True)

		working_path.replace(output)

		if merge_audio:
			try:
				_transfer_audio(video, output, temp_directory, ffmpeg)
			except Exception as exc:
				warnings.warn(
					f"audio transfer failed; {output} has no audio ({exc})",
					RuntimeWarning,
					stacklevel=2,
				)
		return output


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
	return engine.interpolate_in_tensor(
		video,
		exp,
		layout=layout,
		output_layout=output_layout,
		output_device=output_device,
	)
