from torchvision import transforms
import torch.optim as optim
import numpy as np
import csv
import pickle
import torch
import gc

from trainer import fit
from datasets import NUS_WIDE

from losses import InterTripletLoss

class IntermodalTripletNet(torch.nn.Module):
    def __init__(self, modalityOne_net, modalityTwo_net):
        super(IntermodalTripletNet, self).__init__()
        self.modalityOneNet = modalityOne_net
        self.modalityTwoNet = modalityTwo_net

    def forward(self, a_v, p_t, n_t, a_t, p_v, n_v):
        output_anch1 = self.modalityOneNet(a_v)
        output_pos2 = self.modalityTwoNet(p_t)
        output_neg2 = self.modalityTwoNet(n_t)

        output_anch2 = self.modalityTwoNet(a_t)
        output_pos1 = self.modalityOneNet(p_v)
        output_neg1 = self.modalityOneNet(n_v)

        return output_anch1, output_pos2, output_neg2, output_anch2, output_pos1, output_neg1

    def get_modOne_embedding(self, x):
        return self.modalityOneNet(x)

    def get_modTwo_embedding(self, x):
        return self.modalityTwoNet(x)

def main(n_epochs, feature_mode):
    ### PARAMETERS ###
    batch_size = 128
    margin = 5
    lr = 1e-3
    output_embedding_size = 200
    random_seed = 21
    ##################
    cuda = torch.cuda.is_available()
    print("CUDA:", cuda)
    # setting up dictionary
    print("Loading in word vectors...")
    text_dictionary = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
    print("Done\n")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    print("Loading NUS_WIDE dataset...")
    data_path = './data/Flickr'
    dataset = NUS_WIDE(root=data_path,
                        transform=transforms.Compose([transforms.Resize((224,224)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean,std)]),
                        feature_mode=feature_mode,
                        word_embeddings=text_dictionary,
                        train=True)
    print("Done\n")


    # creating indices for training data and validation data
    print("Making training and validation indices...")
    from torch.utils.data.sampler import SubsetRandomSampler

    dataset_size = len(dataset)
    validation_split = 0.2

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
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
    text_embedding_net = torch.nn.Sequential(
        torch.nn.Linear(300, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.PReLU(),
        torch.nn.Linear(128, output_embedding_size))
    if feature_mode == 'resnet152':
        image_embedding_net = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.PReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, output_embedding_size))
    elif feature_mode == 'resnet18':
        image_embedding_net = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, output_embedding_size))

    model = IntermodalTripletNet(image_embedding_net, text_embedding_net)
    if cuda:
        model.cuda()

    loss_fn = InterTripletLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    log_interval = 50
    fit(i_triplet_train_loader, i_triplet_val_loader, dataset.intermodal_triplet_batch_sampler, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    pickle.dump(model, open(f'pickles/models/{feature_mode}_{n_epochs}epochs.p', 'wb'))

if __name__ == "__main__":
    # main(5, "resnet152")
    # gc.collect()
    main(15, "resnet152")