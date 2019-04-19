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
    def __init__(self, root, transform, target_transform=None primary_tags=None, secondary_tags=None, feature_mode='vanilla'):
        super(BaseCMRetrievalDataset, self)
        self._foldernames, self._folder_to_idx = self._find_folders(root)

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

    def __len__(self):
        return len(self.images)


class NUS_WIDE(Dataset):
    def __init__(self, root, transform, features='resnet152'):
        self.imgs = tv.datasets.ImageFolder(root=root)
        self.transform = transform

        if features == 'resnet152':
            self.features, self.feature_mode = pickle.load(open("pickles/nuswide_features/resnet152_nuswide_feats_arr.p","rb")), 'resnet152'
        elif features == 'resnet18':
            self.features, self.feature_mode = pickle.load(open("pickles/nuswide_features/resnet18_nuswide_feats_arr.p", "rb")), 'resnet18'
        else:
            self.features, self.feature_mode = None, 'vanilla'

        self.positive_concept_matrix = pickle.load(open("pickles/nuswide_metadata/concept_matrix.p", "rb"))
        self.negative_concept_matrix = pickle.load(open("pickles/nuswide_metadata/neg_concept_matrix.p", "rb"))
        self.relevancy_matrix = pickle.load(open("pickles/nuswide_metadata/relevancy_matrix.p", "rb"))
        self.tag_matrix = pickle.load(open("pickles/nuswide_metadata/tag_matrix.p", "rb"))
        self.folder_labels = pickle.load(open("pickles/nuswide_metadata/folder_labels.p", "rb"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, data, target) where target is class_index of the target class.
        """
        if self.feature_mode is not 'vanilla':
            return index, self.features[index], self.imgs[index][1]

        if self.transform is not None:
            return index, self.transform(self.imgs[index][0]), self.imgs[index][1]

        return index, self.data[index], self.labels[index]

    def get_concepts(self, index):
        """
        Args:
            index (int): Index of image
        Returns:
            List of concepts (strings) for the image
        """
        return self.positive_concept_matrix[index]

    def get_negative_concepts(self, index):
        """
        Args:
            index (int): Index of image
        Returns:
            List of negative concepts (strings) for the image
        """
        return self.negative_concept_matrix[index]

    def get_folder_label(self, index):
        """
        Args:
            index (int): Index of image
        Returns:
            label of folder (string)
        """
        return self.folder_labels[index]

    def __len__(self):
        return len(self.imgs)


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
