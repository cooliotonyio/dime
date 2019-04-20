import numpy as np
from PIL import Image
import os
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch
import pickle

class BaseCMRetrievalDataset(Dataset):
    """
    Description:
        Base dataset to build off of for other datasets for crossmodal retrieval tasks

    """
    def __init__(self, root, transform, primary_tags=None, secondary_tags=None):
        super(BaseCMRetrievalDataset, self).__init__()
        self._folder_names, self._folder_to_idx = self._find_folders(root)

        self.image_paths, self._folder_targets = self._make_base(self, root, self._folder_to_idx)

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

            for root, _, fname in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if filename.lower().endswith(POS_EXT):
                        path = os.path.join(root, fname)
                        images.append(path)
                        folder_targets.append(folder_to_idx[target])

        return images, folder_targets


    def __getitem__(self, index):
        path = self.images[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = primary_tags[index]

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
        return len(self.images)


class NUS_WIDE(BaseCMRetrievalDataset):
    """
    Description:
        Dataset class to manage the NUSWIDE dataset

    """

    def __init__(self, root, transform, feature_mode='resnet152', word_embeddings='fastText'):
        primary_tags = pickle.load(open("pickles/nuswide_metadata/tag_matrix.p", "rb"))

        super(NUS_WIDE, self).__init__(root, transform, primary_tags=primary_tags)

        self.primary_tags = self._filter_unavailable(self.primary_tags, )
        self.secondary_tags = self._make_secondary_tags()

        self.feature_mode = feature_mode

        if feature_mode == 'resnet152':
            self.features = pickle.load(open("pickles/nuswide_features/resnet152_nuswide_feats_arr.p","rb"))
        elif feature_mode == 'resnet18':
            self.features = pickle.load(open("pickles/nuswide_features/resnet18_nuswide_feats_arr.p", "rb"))
        else:
            self.features, self.feature_mode = None, 'vanilla'

        if word_embeddings == 'fastText':
            self.word_embeddings = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))

        self.positive_concept_matrix = pickle.load(open("pickles/nuswide_metadata/concept_matrix.p", "rb"))
        self.negative_concept_matrix = pickle.load(open("pickles/nuswide_metadata/neg_concept_matrix.p", "rb"))
        self.relevancy_matrix = pickle.load(open("pickles/nuswide_metadata/relevancy_matrix.p", "rb"))


    def _make_secondary_tags(self):
        """
        The secondary tags for NUS-WIDE are the associated folder names that the images reside in

        """

        return [self._folder_names[self._folder_targets[i]] for i in range(len(self))]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, data, target) where target is class_index of the target class.
        """

        sample, target = super().__getitem__(index)

        if self.feature_mode is not 'vanilla':
            return index, self.features[index], target

        if self.transform is not None:
            return index, self.transform(sample), target

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

        for tag_list in tags:
            for tag in tag_list:
                if tag not in embedding_dictionary:
                    tag_list.remove(tag)

        return tags


    def _get_random_embedding(self, index, mode, dtype):
        if mode == 'primary':
            tag_set = self.primary_tags
        else:
            tag_set = self.secondary_tags

        tag_list = tag_set[index]
        random_tag = random.choice(tag_list)
        if mode == 'embedding':
            return self.word_embeddings[random_tag]

        return random_tag


# Dataset used for nearest neighbors loading
class NUS_WIDE_KNN(Dataset):
    def __init__(self, root, transform, text_labels, features=None):
        self.imgs = tv.datasets.ImageFolder(root=root)
        self.transform = transform
        self.text_labels = text_labels
        self.features = features

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self.features is not None:
            return self.features[index], index
        return self.transform(self.imgs[index][0]), index

    def get_text_label(self, index):
        return self.text_labels[self.imgs[index][1]]

    def get_raw_image(self, index):
        return self.imgs[index][0]

    def __len__(self):
        return len(self.imgs)
