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

sys.path.append("{}/..".format(sys.path[0]))
from datasets import NUS_WIDE

base = "pickles/nuswide_features/"
if not os.path.isdir(base):
    raise RuntimeError("Base folder '{}' not found".format(base))
    

scaler = transforms.Resize((224,224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

resnet18 = models.resnet18(pretrained=True)
resnet18_layer = resnet18._modules.get('avgpool')
resnet18.eval()
resnet18.cuda()


def get_image_feature(im_path):

    embedding = torch.cuda.FloatTensor(1, 512, 1, 1).fill_(0)
    img = Image.open(im_path).convert('RGB')
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()

    def copy_data(m,i,o):
        embedding.copy_(o.data)

    h = resnet18_layer.register_forward_hook(copy_data)
    resnet18(t_img)
    h.remove()

    return embedding

dataset = NUS_WIDE('data/Flickr', None)

feature_dict = {}
feature_array = [None] * len(dataset)

for i in range(len(dataset)):
    file_path = dataset.image_paths[i]
    feature_i = get_image_feature(file_path)
    feature_dict[file_path] = feature_i.cpu().squeeze()
    feature_array[i] = feature_i.cpu().squeeze()

pickle.dump(feature_dict, open(base + 'resnet18_nuswide_feats_dict.p', 'wb'))
pickle.dump(feature_array, open(base + 'resnet18_nuswide_feats_arr.p', 'wb'))
