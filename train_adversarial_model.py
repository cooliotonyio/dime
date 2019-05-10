import csv
from datasets import NUS_WIDE
import pickle

import torchvision as tv
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from adversarial_trainer import fit
import numpy as np

from networks import TextEmbeddingNet, Resnet152EmbeddingNet, IntermodalTripletNet, Resnet18EmbeddingNet, ModalityDiscriminator
from losses import InterTripletLoss

### PARAMETERS ###
batch_size = 64
margin = 5
lr = 1e-3
n_epochs = 12
output_embedding_size = 200
feature_mode = 'resnet152'
##################

cuda = torch.cuda.is_available()

# Loading in word embedding dictionary
print("Loading in word vectors...")
text_dictionary = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))

print("Done\n")

# Loading in dataset
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
print("Loading NUS_WIDE dataset...")
data_path = './data/Flickr'
dataset = NUS_WIDE(root=data_path,
                    transform=transforms.Compose([tv.transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean,std)]),
                    feature_mode=feature_mode,
                    word_embeddings=text_dictionary,
                    train=True)

#val_dataset = NUS_WIDE(root=data_path,
#                    transform=transforms.Compose([tv.transforms.Resize((224,224)),
#                                                    transforms.ToTensor(),
#                                                    transforms.Normalize(mean,std)]),
#                    feature_mode=feature_mode,
#                    word_embeddings=text_dictionary,
#                    train=False)

print("Done\n")

# Setting up dataloaders

# creating indices for training data and validation data
print("Making training and validation indices...")
from torch.utils.data.sampler import SubsetRandomSampler

dataset_size = len(dataset)
validation_split = 0.2

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(21)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
print("Done.")

kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}
i_triplet_train_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, **kwargs)
i_triplet_val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)

# Set up the embedding/projection network
text_embedding_net = TextEmbeddingNet(dim=output_embedding_size)
if feature_mode == 'resnet152':
    image_embedding_net = Resnet152EmbeddingNet(dim=output_embedding_size)
elif feature_mode == 'resnet18':
    image_embedding_net = Resnet18EmbeddingNet(dim=output_embedding_size)

proj_model = IntermodalTripletNet(image_embedding_net, text_embedding_net)

# Setting up the discriminator network
d_model = ModalityDiscriminator(dim=output_embedding_size)

if cuda:
    d_model.cuda()
    proj_model.cuda()

# Init the projection loss
proj_loss_fn = InterTripletLoss(margin)

# Init the discriminator loss
d_loss_fn = torch.nn.BCELoss()

# Init the optimizer for projector
proj_optimizer = optim.Adam(proj_model.parameters(), lr=lr)

# Init the optimizer for discriminator
d_optimizer = optim.Adam(d_model.parameters(), lr=1e-3)

#scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

log_interval = 100
fit(i_triplet_train_loader, i_triplet_val_loader, dataset.intermodal_triplet_batch_sampler,
        proj_model, d_model,
        proj_loss_fn, d_loss_fn,
        proj_optimizer, d_optimizer,
        n_epochs, cuda, log_interval)

pickle.dump(model, open('pickles/models/baseline.p', 'wb'))
