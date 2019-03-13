"""
This script passes through the entire nuswide dataset through a pretrained resnet152 model. The features
are stored in both dictionary and array formats, resnet152_nuswide_feats_dict.p & resnet152_nuswide_feats_arr.p
respectively. The dictionary object maps file path names to the feature which is stored as a FloatTensor object

Note:
    The file_path will be in respects to the root of the directory. The dictionary keys will be as such:

     ./data/Flickr/actor/0001_2124494179.jpg


The pickles containing the dictionary and array objects will be stored in ./pickles/nuswide_features/

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

from datasets import NUS_WIDE

base = "./pickles/nuswide_features/"
if not os.path.isdir(base):
    os.mkdir(os.fsencode(base))

scaler = transforms.Resize((224,224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

resnet152 = models.resnet152(pretrained=True)
modules = list(resnet152.children())[:-1]
resnet152 = torch.nn.Sequential(*modules)
resnet152.cuda()

for p in resnet152.parameters():
    p.requires_grad = False

def get_image_feature(im_path):
    img = Image.open(im_path).convert('RGB')
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()
    feature = resnet152(t_img).data
    return feature

dataset = NUS_WIDE('./data/Flickr', None)

feature_dict = {}
feature_array = []

for i in range(len(dataset)):
    print("file: ", i)
    file_path = dataset.imgs.samples[i][0]
    feature_i = get_image_feature(file_path)
    feature_dict[file_path] = feature_i
    feature_array.append(feature_i)

pickle.dump(feature_dict, open(base + 'resnet152_nuswide_feats_dict.p', 'wb'))
pickle.dump(feature_array, open(base + 'resnet152_nuswide_feats_arr.p', 'wb'))
