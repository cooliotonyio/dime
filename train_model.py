import csv
from datasets import NUS_WIDE
import pickle

import torchvision as tv
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from trainer import fit
import numpy as np

from networks import TextEmbeddingNet, Resnet152EmbeddingNet, IntermodalTripletNet, Resnet18EmbeddingNet
from losses import InterTripletLoss

### PARAMETERS ### 
batch_size = 128
margin = 5
lr = 1e-3
n_epochs = 10
output_embedding_size = 64
feature_mode = 'resnet152'
##################

cuda = torch.cuda.is_available()
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

print("Loading NUS_WIDE dataset...")
data_path = './data/Flickr'
dataset = NUS_WIDE(root=data_path,
    transform=transforms.Compose([tv.transforms.Resize((224,224)), transforms.ToTensor(),
                                 transforms.Normalize(mean,std)]), features=feature_mode)
print("Done\n")

# setting up dictionary
print("Loading in word vectors...")
text_dictionary = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
print("Done\n")

# creating indices for training data and validation data
print("Making training and validation indices...")
from torch.utils.data.sampler import SubsetRandomSampler

dataset_size = len(dataset)
validation_split = 0.3

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(21)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
print("Done.")

# making loaders
kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}
i_triplet_train_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, sampler=train_sampler, **kwargs)
i_triplet_val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, **kwargs)

# Set up the network and training parameters
text_embedding_net = TextEmbeddingNet(dim=output_embedding_size)
if feature_mode == 'resnet152':
    image_embedding_net = Resnet152EmbeddingNet(dim=output_embedding_size)
elif feature_mode == 'resnet18':
    image_embedding_net = Resnet18EmbeddingNet(dim=output_embedding_size)
    
model = IntermodalTripletNet(image_embedding_net, text_embedding_net)
if cuda:
    model.cuda()

loss_fn = InterTripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

log_interval = 100
fit(i_triplet_train_loader, i_triplet_val_loader, dataset, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, text_dictionary)

pickle.dump(model, open('pickles/models/entire_nuswide_model_10.p', 'wb'))