"""
Given a directory of .mp4 files, this produces spatial features of the video through the method of mean pooling. For
each video, a single frame is taken per second and across all frames, each is passed through a pretrained resnet152
model and mean pooled into a single 2048 tensor. The features are then pickled.

usage:
    python3 spatial_feature_generator.py (video directory)

ex:
    python3 spatial_feature_generator.py ../mmcommons/

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
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

base = "/pickles/nuswide_features/"

scaler = transforms.Resize((224,224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

resnet152 = models.resnet152(pretrained=True)
modules = list(resnet152.children())[:-1]
resnet152 = torch.nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

def get_image_feature(im_path)
    img = Image.open(im_path)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    feature = resnet152(t_img).data
    return feature


base = "./video_features"
if not os.path.isdir(base):
    os.mkdir(os.fsencode(base))
    os.mkdir(os.fsencode(base + "/frame_level_features"))
    os.mkdir(os.fsencode(base + "/video_level_features"))

video_directory_path = sys.argv[1]
video_files = [(f, video_directory_path + f) for f in os.listdir(video_directory_path) if
               os.path.isfile(os.path.join(video_directory_path, f))]


frame_level_features = {}
for f in video_files:
    name = f[0].split('.')[0]
    curr_work_pth = base + "/frame_level_features/" + name
    os.mkdir(os.fsencode(curr_work_pth))
    cmd = "ffmpeg -i " + f[1] + " -vf fps=1 " + curr_work_pth + '/' + "%04d.jpg -hide_banner"
    os.system(cmd)

    frame_level_feat = []
    for img_pth in os.listdir(curr_work_pth):
        frame_level_feat.append(get_image_feature(curr_work_pth + '/' + img_pth))

    if len(frame_level_feat) == 0:
        continue;

    all_frames = torch.stack(frame_level_feat)
    avg_aggregation = torch.mean(all_frames, 0)
    frame_level_features[f] = avg_aggregation

pickle.dump(frame_level_features, open(base + 'frame_level_features.p', 'wb'))
