#!/bin/bash

# 设置输出文件名和帧率
OUTPUT="render_progress.mp4"
FRAMERATE=10

# 创建临时输入列表
INPUT_LIST="input.txt"
rm -f $INPUT_LIST

# 遍历 frames 目录下的 ppm 文件，按文件名排序
for f in $(ls frames/frame_*.ppm | sort); do
    echo "file '$PWD/$f'" >> $INPUT_LIST
done

# 生成视频
ffmpeg -framerate $FRAMERATE -f concat -safe 0 -i $INPUT_LIST -c:v libx264 -pix_fmt yuv420p -y $OUTPUT

echo "✅ 视频已生成: $OUTPUT"