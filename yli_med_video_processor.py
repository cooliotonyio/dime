"""
This processes the YFCC100M dataset  to multiple frames at 24fps for later feature extraction using the pretrained
DeepMind i3D model.

usage:
    python3 yli_med_video_processor.py (video directory)

ex:
    python3 yli_med_video_processor.py ../mmcommons/

    where the directory structure of ./mmcommons/ is:
        ./mmcommons/
            (video-1).mp4
            (video-2).mp4
                ...
"""

import os
import io
import torch
import sys

base = './data/YLI-MED-25rgb/'
if not os.path.isdir(base):
    os.mkdir(os.fsencode(base))

video_directory_path = sys.argv[1]
video_files = [(f, video_directory_path + f) for f in os.listdir(video_directory_path) if
               os.path.isfile(os.path.join(video_directory_path, f))]

for f in video_files:
    name = f[0].split('.')[0]
    curr_work_pth = base + name
    os.mkdir(os.fsencode(curr_work_pth))
    cmd = "ffmpeg -i " + f[1] + " -vf fps=24 " + curr_work_pth + '/' + "%04d.jpg -hide_banner"
    os.system(cmd)
