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

resnet152 = models.resnet152(pretrained=True)
resnet152_layer = resnet152._modules.get('avgpool')
resnet152.eval()
resnet152.cuda()


def get_image_feature(im_path):
    embedding = torch.cuda.FloatTensor(1, 2048, 1, 1).fill_(0)
    img = Image.open(im_path).convert('RGB')
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()

    def copy_data(m,i,o):
        embedding.copy_(o.data)

    h = resnet152_layer.register_forward_hook(copy_data)
    resnet152(t_img)
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

pickle.dump(feature_dict, open(base + 'resnet152_nuswide_feats_dict.p', 'wb'))
pickle.dump(feature_array, open(base + 'resnet152_nuswide_feats_arr.p', 'wb'))
