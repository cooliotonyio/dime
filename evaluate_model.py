"""
Calculates MiAP score, per-class-recall, per-class-precision, per-class-f1
"""


import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datasets import NUS_WIDE_KNN
from torchvision import transforms
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import faiss

model = pickle.load(open("pickles/models/entire_nuswide_model_12.p", "rb"))
relevancy_matrix = pickle.load(open("pickles/nuswide_metadata/relevancy_matrix.p", "rb"))
features = pickle.load(open("pickles/nuswide_features/resnet152_nuswide_feats_arr.p", "rb"))
NUS_WIDE_classes = []

"""
args:
    tag_rankings: 2D matrix, 269648 x 81
        tag_rankings[i:j]: the jth relevant tag of the ith image

output:
    MiAP score
"""
def MiAP(tag_rankings, q_indices):
    iAPs = np.zeros(len(q_indices))

    nonzero = 0
    for idx, tag_ranking in enumerate(tag_rankings):
        R = np.sum(relevancy_matrix[idx, :])
        if R == 0:
            continue

        S = 0.
        rj = 0.
        for j, tag in enumerate(tag_ranking, start=1):
            if relevancy_matrix[idx, tag] == 1:
                rj += 1.
                S += rj / j
        iAPs[idx] = np.divide(1, R) * S
        nonzero += 1

    print(iAPs)
    return np.sum(iAPs) / nonzero

def f1_precision_recall(t_indices, tag_rankings, k=3):
    top_k_relevant = tag_rankings[:,:k]
    num_concepts = relevancy_matrix.shape[1]

    class_recalls = np.zeros(num_concepts)
    class_precisions = np.zeros(num_concepts)
    Np = np.zeros(num_concepts)
    Nc = np.zeros(num_concepts)
    Ng = np.zeros(num_concepts)


    for i in range(top_k_relevant.shape[0]):
        for j in range(top_k_relevant.shape[1]):
            Np[top_k_relevant[i,j]] += 1
            if relevancy_matrix[i, top_k_relevant[i,j]] == 1:
                Nc[top_k_relevant[i,j]] += 1

    for concept_idx in range(num_concepts):
        Ng_i = np.sum(relevancy_matrix[:,concept_idx])
        Ng[concept_idx] = Ng_i
        Np_i = Np[concept_idx]
        Nc_i = Nc[concept_idx]

        if Ng_i != 0:
            class_recalls[concept_idx] = Nc_i / Ng_i
        if Np_i != 0:
            class_precisions[concept_idx] = Nc_i / Np_i

    np.seterr(divide='ignore', invalid='ignore')
    per_class_f1 = np.divide(2 * np.multiply(class_precisions, class_recalls), class_precisions + class_recalls)
    return per_class_f1, np.sum(Nc) / np.sum(Np), np.sum(Nc) / np.sum(Ng), class_precisions, class_recalls

def make_loaders():
   # init dataset
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    data_path = 'NUS_WIDE'

    dataset = NUS_WIDE_KNN(data_path,
        transforms.Compose([tv.transforms.Resize((224,224)), transforms.ToTensor(),
                                     transforms.Normalize(mean,std)]), NUS_WIDE_classes)
    # splitting up train and test:
    dataset_size = len(dataset)
    validation_split = 0.3

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(21)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # init loaders
    batch_size = 512
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)

    return train_loader, test_loader

"""
    outputs:
        base_loader: nuswide 81 concept pairs of (concept, FastText[concept])
        query_loader: all nuswide images to query the faiss index
"""
def make_loaders_text():
   # init dataset
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    data_path = './data/Flickr'

    dataset = NUS_WIDE_KNN(data_path,
        transforms.Compose([tv.transforms.Resize((224,224)), transforms.ToTensor(),
                                     transforms.Normalize(mean,std)]), NUS_WIDE_classes, features=features)

    base_loader = pickle.load(open("./pickles/nuswide_metadata/base_loader.p", "rb"))

    dataset_size = len(dataset)
    validation_split = 0.3

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(21)
    np.random.shuffle(indices)
    test_indices = indices[:split]

    test_sampler = SubsetRandomSampler(test_indices)

    # init loaders
    batch_size = 512
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}
    query_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, **kwargs)

    return base_loader, query_loader

"""
output:
    - t_indices | list :
        - t_indices[i] = index of the ith test image in the original dataset
    - ranking | # of test set images x 81 x 2 :
        - ranking[i,j]: tuple corresponding to the jth ranked concept
            - tuple[0]: count of number of appearances of jth ranked concept in k nearest neighbors retrieval
            - tuple[1]: index of jth ranked concept
"""
def make_tag_ranking(model, train_loader, test_loader, relevancy_matrix):
    nearest_images, t_indices = faiss_similarity(model, train_loader, test_loader)

    # ranking[i,j,:]: tuple corresponding to the jth relevant tag of image i
    # ranking[i,j,0]: count of tag
    # ranking[i,j,1]: column index in relevancy matrix

    ranking = np.zeros((nearest_images.shape[0], 81, 2), dtype=int)

    for i in range(nearest_images.shape[0]):
        for j in range(nearest_images.shape[1]):
            relevant_tags = np.argwhere(relevancy_matrix[nearest_images[i,j],:])[:,0]
            for k in relevant_tags:
                ranking[i,k,0] += 1
        ranking[i,:,1] += np.argsort(ranking[i,:,0])[::-1]
        ranking[i,:,0].sort()
        ranking[i,:,0] = np.flip(ranking[i,:,0], axis=0)

    return t_indices, ranking


"""
input:
    - t_indices:
    - ranking | # of test set images x 81 x 2 :
        - ranking[i,j]: tuple corresponding to the jth ranked concept
            - tuple[0]: count of number of appearances of jth ranked concept in k nearest neighbors retrieval
            - tuple[1]: index of jth ranked concept
output:
    image_to_ranked_relevancy | 269645 x 81 matrix :
        - [i,j]: jth ranked concept to the ith image
"""
def make_image_ranked_relevancy_matrix(t_indices, ranking):
    largest_index = t_indices.max()
    image_to_ranked_relevancy = np.full((largest_index + 1, ranking.shape[1]), -1)

    for img_idx, tag_ranking in zip(t_indices, ranking):
        for idx, count_concept in enumerate(tag_ranking):
            image_to_ranked_relevancy[img_idx, idx] = count_concept[1]
    return image_to_ranked_relevancy


"""
output:
    indices | a matrix:
        - indices[i,j]: index of jth relevant image of image i
            (where image i is a query from the test loader)
            (where the index of jth relevant image is the index in the full dataset)
    t_indices | a list:
        - ith entry is an index in the original dataset that corresponds to the ith image of the test set
"""

def faiss_similarity(model, base_loader, query_loader, k=64):
    base_db, b_indices = make_db_text(model, base_loader)
    query_db, q_indices = make_db_images(model, query_loader)

    index = faiss.IndexFlatL2(k)
    index.add(base_db)
    _, indices = index.search(query_db, k)

    return indices, q_indices

"""
output:
    - from embeddings produced by model, creates matrix for faiss training/querying
    - returns a list: ith entry is an index in the original dataset that corresponds to the ith image in faiss_db
"""
def make_db_images(model, train_loader):
    cuda = True
    n = len(train_loader.sampler)
    d = 64 # size of the embeddings outputted by the model
    model.eval()

    faiss_db = np.empty((n,d), dtype='float32')
    fidx_to_idx = np.empty(n, dtype=int)

    n_idx = 0
    for _, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple,list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
        target = target.numpy()

        with torch.no_grad():
            embeddings = model.get_modOne_embedding(*data)

        for idx in range(len(embeddings)):
            faiss_db[n_idx + idx, :] = embeddings[idx].cpu().detach().numpy()
            fidx_to_idx[n_idx + idx] = target[idx]
        n_idx += len(embeddings)

    return faiss_db, fidx_to_idx

"""
output:
    - from embeddings produced by model, creates matrix for faiss training/querying
    - returns a list: ith entry is an index in the original dataset that corresponds to the ith image in faiss_db
"""
def make_db_text(model, loader):
    cuda = True
    n = len(loader)
    d = 64 # size of embeddings outputted by the model

    faiss_db = np.empty((n,d), dtype='float32')
    fidx_to_word = [None] * n
    model.eval()

    n_idx = 0
    for fidx, (key, value) in enumerate(loader):
        with torch.no_grad():
            embedding = model.get_modTwo_embedding(value.unsqueeze(0).cuda()).data.cpu(). numpy()
        faiss_db[fidx, :] = embedding
        fidx_to_word[fidx] = key

    return faiss_db, fidx_to_word

def display_metrics():
    base_loader, query_loader = make_loaders_text()
    tag_rankings, q_indices = faiss_similarity(model, base_loader, query_loader)
    miap = MiAP(tag_rankings, q_indices)
    f1, overall_precision, overall_recall, class_precisions, class_recalls = f1_precision_recall(q_indices, tag_rankings, k=7)

    print("MiAP:", miap)
    print("f1: ", np.nanmean(f1))
    print("overall Precision: ", overall_precision)
    print("overall Recall: ", overall_recall)
    print("per class Precision: ", class_precisions)
    print("per class Recall: ", class_recalls)
    print("mean class Precision: ", np.sum(class_precisions), len(np.nonzero(class_precisions)[0]))
    print("mean class Recall: ", np.sum(class_recalls), len(np.nonzero(class_recalls)[0]))

display_metrics()
