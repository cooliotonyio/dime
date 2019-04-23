import numpy as np
from PIL import Image
import os
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch
import pickle
import random
import numpy as np

class BaseCMRetrievalDataset(Dataset):
    """
    Description:
        Base dataset to build off of for other datasets for crossmodal retrieval tasks

    """
    def __init__(self, root, transform, primary_tags=None, secondary_tags=None):
        super(BaseCMRetrievalDataset, self).__init__()
        self._folder_names, self._folder_to_idx = self._find_folders(root)

        self.image_paths, self._folder_targets = self._make_base(root, self._folder_to_idx)

        self.transform = transform

        if primary_tags is None:
            self.primary_tags = self._folder_targets
        else:
            self.primary_tags = primary_tags

        self.secondary_tags = secondary_tags


    def _find_folders(self, dir):
        """
        Organizes and indexes the class folders

        Args:
            dir (string): Directory path

        Returns:
            tuple: (folder name, folder_to_idx) where folder name is the name of a folder in the directory and
            folder_to_idx is a dictionary that takes the folder name and gives its corresponding index
        """

        folder_names = [d.name for d in os.scandir(dir) if d.is_dir()]
        folder_names.sort()
        folder_to_idx = {folder_names[i] : i for i in range(len(folder_names))}
        return folder_names, folder_to_idx


    def _make_base(self, dir, folder_to_idx):
        """
        Produces the two most basic elements of the dataset, inputs (images) and targets

        Args:
            dir (string): Directory path

        Returns:
            tuple: (images, folder_targets).
                - images (list): images[i] = image path of i'th image
                - folder_targets (list): folder_targets[i] = the index of the folder that the i'th image came from
        """

        POS_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        images = []
        folder_targets = []

        dir = os.path.expanduser(dir)
        for target in sorted(folder_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(POS_EXT):
                        path = os.path.join(root, fname)
                        images.append(path)
                        folder_targets.append(folder_to_idx[target])

        return images, folder_targets

    def _get_image(self, index):
        path = self.image_paths[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        return img


    def __getitem__(self, index):
        img = self._get_image(index)

        if self.transform is not None:
            img = self.transform(img)

        target = self.primary_tags[index]

        return img, target


    def get_primary_tags(self, index):
        """
        Args:
            index of image

        Returns:
            primary_tags (list): list of strings that are related to the image of the given index
        """

        return self.primary_tags[index]


    def get_secondary_tags(self, index):
        """
        Args:
            index of image

        Returns:
            secondary_tags (list): list of strings that are related to the image of the given index
        """

        return self.secondary_tags[index]


    def __len__(self):
        return len(self.image_paths)


def idx_maker(fname):
    with open(fname) as f:
        idx_to_ = f.readlines()

    for idx, line in enumerate(idx_to_):
        idx_to_[idx] = line.split('\n')[0]

    return idx_to_

def label_matrices_maker(relevancy_matrix, idx_to_label, exclude_list=None):
    n = relevancy_matrix.shape[0]
    label_matrix = [None] * n
    neg_label_matrix = [None] * n

    for idx, line in enumerate(relevancy_matrix):
        labels = []
        neg_labels = []
        for count, indicator in enumerate(line):
            if exclude_list is not None and idx_to_label[count] in exclude_list:
                continue

            if indicator == 1:
                labels.append(idx_to_label[count])
            else:
                neg_labels.append(idx_to_label[count])
        label_matrix[idx] = labels
        neg_label_matrix[idx] = neg_labels

    return label_matrix, neg_label_matrix

class NUS_WIDE(BaseCMRetrievalDataset):
    """
    Description:
        Dataset class to manage the NUSWIDE dataset

    """

    def __init__(self, root, transform, train=True, feature_mode='resnet152', word_embeddings=None):
        primary_tags = pickle.load(open("pickles/nuswide_metadata/tag_matrix.p", "rb"))

        super(NUS_WIDE, self).__init__(root, transform, primary_tags=None)

        self.image_paths = self._make_image_paths(root, train)

        self._folder_names = pickle.load(open("pickles/nuswide_metadata/folder_labels.p", "rb"))
        self.secondary_tags = self._make_secondary_tags()

        self.feature_mode = feature_mode

        if feature_mode == 'resnet152':
            self.features = pickle.load(open("pickles/nuswide_features/resnet152_nuswide_feats_dict.p","rb"))
        elif feature_mode == 'resnet18':
            self.features = pickle.load(open("pickles/nuswide_features/resnet18_nuswide_feats_dict.p", "rb"))
        else:
            self.features, self.feature_mode = None, 'vanilla'

        self.word_embeddings = word_embeddings

        self.concept_relevancy_matrix = NUS_WIDE.make_concept_relevancy_matrix(train)

        self.idx_to_concept = NUS_WIDE._make_idx_to_concept()

        self.positive_concept_matrix, self.negative_concept_matrix = self._make_concept_matrices()

        self.tag_relevancy_matrix = NUS_WIDE.make_tag_relevancy_matrix(train)

        self.idx_to_tag = NUS_WIDE._make_idx_to_tag()

        self.positive_tag_matrix, self.negative_tag_matrix = self._make_tag_matrices()

        if word_embeddings is not None:
            self.primary_tags = NUS_WIDE._filter_unavailable(self.positive_tag_matrix, word_embeddings)

        if word_embeddings is not None:
            self.negative_tag_matrix = NUS_WIDE._filter_unavailable(self.negative_tag_matrix, word_embeddings)


    def _make_secondary_tags(self):
        """
        The secondary tags for NUS-WIDE are the associated folder names that the images reside in

        """

        return [self._folder_names[self._folder_targets[i]] for i in range(len(self))]


    def _make_image_paths(self, dir, train=True):
        if train:
            file_paths_fname = "./nuswide_metadata/Imagelist/TrainImagelist.txt"
        else:
            file_paths_fname = "./nuswide_metadata/Imagelist/TestImagelist.txt"

        image_paths = []

        with open(file_paths_fname) as fn:
            lines = fn.readlines()

        for line in lines:
            image_paths.append(os.path.join(dir, line.split('\n')[0].replace("\\","/")))

        return image_paths

    def _make_idx_to_concept():
        fname = "nuswide_metadata/Concepts81.txt"
        idx_to_concept = idx_maker(fname)
        return idx_to_concept


    def _make_idx_to_tag():
        fname = "./nuswide_metadata/TagList1k.txt"
        idx_to_tag = idx_maker(fname)
        return idx_to_tag


    def _make_concept_matrices(self):
        concept_relevancy_matrix = self.concept_relevancy_matrix
        idx_to_concept = self.idx_to_concept
        concept_matrix, neg_concept_matrix = label_matrices_maker(concept_relevancy_matrix, idx_to_concept)
        return concept_matrix, neg_concept_matrix


    def _make_tag_matrices(self):
        tag_relevancy_matrix = self.tag_relevancy_matrix
        idx_to_tag = self.idx_to_tag
        tag_matrix, neg_tag_matrix = label_matrices_maker(tag_relevancy_matrix, idx_to_tag, exclude_list=self.idx_to_concept)
        return tag_matrix, neg_tag_matrix


    def make_concept_relevancy_matrix(train=True):
        path = './nuswide_metadata/TrainTestLabels/'
        if train:
            suffix_indicator = "Train.txt"
            n = 161789
        else:
            suffix_indicator = "Test.txt"
            n = 107859

        relevancy_matrix = np.zeros((n,81), dtype=int)
        filenames = []

        for idx, filename in enumerate(os.listdir(path)):
            if filename.endswith(suffix_indicator):
                filenames.append(filename)

        filenames.sort()

        for idx, filename in enumerate(filenames):
            with open(path + filename) as f:
                content = f.readlines()
                curr_column = np.array([int(i[0]) for i in content], dtype=int)
            relevancy_matrix[:, idx] = curr_column

        return relevancy_matrix


    def make_tag_relevancy_matrix(train=True):
        if train:
            path = './nuswide_metadata/Train_Tags1k.dat'
            n = 161789
        else:
            path = './nuswide_metadata/Test_Tags1k.dat'
            n = 107859

        relevancy_matrix = np.zeros((n,1000), dtype=int)

        with open(path) as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            relevancy_matrix[idx,:] = np.array([int(i) for i in line.split('\t')[:-1]])

        return relevancy_matrix


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, data, target) where target is class_index of the target class.
        """

        sample = self._get_image(index)
        target = self._folder_targets[index]

        if self.feature_mode is not 'vanilla':
            return index, self.features[self.image_paths[index]], self._folder_targets[index]

        if self.transform is not None:
            return index, self.transform(sample), self._folder_targets[index]

        return index, sample, target



    def get_concepts(self, index):
        """
        Args:
            index (int): Index of image
        Returns:
            List of concepts (strings) for the image
        """

        return self.positive_concept_matrix[index]


    def get_random_primary_tag(self, index, dtype='embedding'):
        """
        Obtains a random associated primary tag for the given image index in
        either the format of the word embedding or a string

        Args:
            index: index of the desired image
            dtype: either 'embedding' or 'string'

        Returns:
            random primary tag
        """

        return self._get_random_embedding(index, mode='primary', dtype=dtype)


    def get_random_secondary_tag(self, index, dtype='embedding'):
        """
        Obtains a random associated secondary tag for the given image index in
        either the format of the word embedding or a string

        Args:
            index: index of the desired image
            dtype: either 'embedding' or 'string'

        Returns:
            random secondary tag
        """

        return self._get_random_embedding(index, mode='secondary', dtype=dtype)


    def get_negative_concepts(self, index):
        """
        Args:
            index (int): Index of image
        Returns:
            List of negative concepts (strings) for the image
        """

        return self.negative_concept_matrix[index]


    def _filter_unavailable(tags, embedding_dictionary):
        """
        Filters out the unavailable tags given the available embeddings

        Args:
            tags (list of lists): tags[i] = list of associated tags for i'th image
            embedding_dictionary (dictionary): (word:word embedding tensor)

        Returns:
            tags (list of lists): same as the argument except modified to remove the tags
                                    that are unavailable
        """

        for idx in range(len(tags)):
            iter_list = list(tags[idx])
            for tag in iter_list:
                if not tag in embedding_dictionary:
                    tags[idx].remove(tag)

        return tags

    def intermodal_triplet_batch_sampler(self, batch, cuda):
        """
        Batch sampling function for crossmodal triplet loss

        Args:
            Batch (tuple): tensors of length batch size


        """
        (indices, data, target) = batch
        target = target if len(target) > 0 else None


        labels_set = set()
        label_to_indices = dict()
        for idx in indices:
            sec_tag = self.secondary_tags[idx]
            if sec_tag not in label_to_indices:
                label_to_indices[sec_tag] = [idx]
            else:
                label_to_indices[sec_tag].append(idx)

            for tag in self.get_primary_tags(idx):
                labels_set.add(tag)
                if tag not in label_to_indices:
                    label_to_indices[tag] = [idx]
                else:
                    label_to_indices[tag].append(idx)


        intermod_triplet_data = [[]] * 6
        for i in range(len(intermod_triplet_data)):
            intermod_triplet_data[i] = [None] * len(target)

        for idx in range(len(target)):
            ds_idx = indices[idx] # index of image in dataset (dataset index)
            img = data[idx]       # image data (pre-extracted feature)
            label = target[idx]   # label (folder label in nuswide)
            b_idx = idx           # index of image in batch   (batch index)

            # --- setting image anchor ---
            a_img = img

            # --- setting text anchor ---
            a_txt = self.get_random_primary_tag(ds_idx, dtype='embedding')
            if a_txt is None:
                a_txt = self.get_random_secondary_tag(ds_idx, dtype='embedding')

            # ---setting the positive word vector---
            p_txt = self.get_random_primary_tag(ds_idx, dtype='embedding')
            if p_txt is None:
                p_txt = self.get_random_secondary_tag(ds_idx, dtype='embedding')

            # ---setting negative word vector---
            n_txt = self.word_embeddings[random.choice(self.negative_tag_matrix[ds_idx])]

            # ---setting positive image---
            positive_tag = self.get_random_primary_tag(ds_idx, dtype='string')
            if positive_tag is None:
                positive_tag = self.get_random_secondary_tag(ds_idx, dtype='string')

            positive_index = random.choice(label_to_indices[positive_tag])
            p_img = self[positive_index][1]

            # ---setting negative image---
            negative_label = random.choice(list(labels_set - set(self.get_primary_tags(ds_idx))))
            negative_index = random.choice(label_to_indices[negative_label])
            n_img = self[negative_index][1]

            intermod_triplet_data[0][b_idx] = a_img
            intermod_triplet_data[1][b_idx] = p_txt
            intermod_triplet_data[2][b_idx] = n_txt
            intermod_triplet_data[3][b_idx] = a_txt
            intermod_triplet_data[4][b_idx] = p_img
            intermod_triplet_data[5][b_idx] = n_img

        intermod_triplet_data = [torch.stack(seq) for seq in intermod_triplet_data]

        if cuda:
            intermod_triplet_data = tuple(d.cuda() for d in intermod_triplet_data)

        return intermod_triplet_data


    def _get_random_embedding(self, index, mode, dtype):
        if mode == 'primary':
            tag_set = self.primary_tags
        else:
            tag_set = self.secondary_tags

        tag_list = tag_set[index]

        if not tag_list:
            return None

        if not type(tag_list) is list:
            random_tag = tag_list
        else:
            random_tag = random.choice(tag_list)

        if dtype == 'embedding':
            return self.word_embeddings[random_tag]

        return random_tag


# Dataset used for nearest neighbors loading
class NUS_WIDE_KNN(NUS_WIDE):
    def __init__(self, root, transform, feature_mode='resnet152', train=True):
        super(NUS_WIDE_KNN, self).__init__(root, transform, train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self.features is not None:
            return self.features[self.image_paths[index]], index
        return self.transform(self.imgs[index][0]), index # TODO: FIX

    def get_text_label(self, index):
        return self.text_labels[self.imgs[index][1]]

    def get_raw_image(self, index):
        return self.imgs[index][0]
