from pathlib import Path

from tqdm import tqdm

from practical_rife_a6i import RifeVideoInterpolator

def test_interpolation_in_file():
	engine = RifeVideoInterpolator(fp16=True, device='cuda')

	output = Path('D:\\sc\\豆包_2x.mp4')

	pbar = tqdm(
		total=None,
		unit="frame",
		desc=output.name,
	)

	def progress(current: int, total: int):
		if pbar.total != total:
			pbar.total = total
			pbar.refresh()
		pbar.update(current - pbar.n)

	engine.interpolate_in_file(
		video=Path('D:\\sc\\豆包.mp4'),
		temp_directory=Path('D:\\temp'),
		exp=1,
		output=output,
		ffmpeg=Path('C:\\Program Files\\WinGet\\Packages\\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-9.0.2-full_build-shared\\bin\\ffmpeg.exe'),
		progress_fn=progress
	)