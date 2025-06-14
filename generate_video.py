from pathlib import Path
import imageio.v3 as iio
import re
import numpy as np

# 设置路径和参数
frames_dir = Path("/ssd10/jxc/smallpt/cu-smallpt_video/build/frames")
output_file = "render_progress.mp4"
fps = 10

# 匹配文件名中的数字，例如 frame_0040.ppm -> 40
def extract_frame_number(file: Path):
    match = re.search(r"frame_(\d+)\.ppm", file.name)
    return int(match.group(1)) if match else float('inf')

# 获取并排序所有 .ppm 文件
frame_files = sorted(frames_dir.glob("frame_*.ppm"), key=extract_frame_number)

# 读取所有图像帧
frames = [iio.imread(str(file)) for file in frame_files]

# 写入视频
iio.imwrite(output_file, np.array(frames), fps=fps)

print(f"✅ 视频已生成：{output_file}（{len(frames)}帧）")