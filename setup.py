"""
Running this script will fetch and cache all necessary datasets

If datasets have already been downloaded to data_zipped, 
this script will only extract/process/pickle the downloaded dataset

If a re-download is desired, simple change FORCE_DOWNLOAD to True
"""


import os
import io
import torch
import pickle
import tarfile
from zipfile import ZipFile
from util import fetch_and_cache

FORCE_DOWNLOAD = False

print("Starting setup... This might take a while.")
print("Making directories...", end=" ")
if not os.path.isdir("./data_zipped"):
    os.mkdir(os.fsencode('./data_zipped'))
if not os.path.isdir("./data"):
    os.mkdir(os.fsencode('./data'))
if not os.path.isdir("./pickles/word_embeddings"):
    os.mkdir(os.fsencode("./pickles/word_embeddings"))
if not os.path.isdir("./pickles/models"):
    os.mkdir(os.fsencode("./pickles/models"))
if not os.path.isdir("./pickles/nuswide_metadata"):
    os.mkdir(os.fsencode("./pickles/nuswide_metadata"))
print("Done!")

print("Downloading NUSWIDE_metadata...")
fetch_and_cache(data_url = 'http://dl.nextcenter.org/public/nuswide/NUS_WID_Tags.zip',
                file = 'tags.zip',
                data_dir = './data_zipped',
                force = FORCE_DOWNLOAD)
fetch_and_cache(data_url = 'http://dl.nextcenter.org/public/nuswide/Groundtruth.zip',
                file = 'groundtruth.zip',
                data_dir = './data_zipped'.
                force = FORCE_DOWNLOAD)
fetch_and_cache(data_url = 'http://dl.nextcenter.org/public/nuswide/ConceptsList.zip',
                file = 'concepts.zip',
                data_dir = './data_zipped',
                force = FORCE_DOWNLOAD)

print("Extracting NUSWIDE metadata...")
with ZipFile('data_zipped/tags.zip', 'r') as data_zipped:
    data_zipped.extractall(path = "data/nuswide_metadata/")
with ZipFile('data_zipped/groundtruth.zip', 'r') as data_zipped:
    data_zipped.extractall(path = "data/nuswide_metadata/")
with ZipFile('data_zipped/concepts.zip', 'r') as data_zipped:
    data_zipped.extractall(path = "data/nuswide_metadata/")
print("Done extracting NUSWIDE metadata!")


print("Processing and pickling NUSWIDE metadata...")
os.system("python3 nuswide_processing_scripts/make_relevancy_matrix.py")
os.system("python3 nuswide_processing_scripts/make_tag_matrix.py")
os.system("python3 nuswide_processing_scripts/make_concepts.py")
print("Done processing and pickling NUSWIDE metadata!")

print("Downloading the FastText word embeddings... (this might take some time)")
fetch_and_cache(data_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                file = 'word_vecs.zip',
                data_dir = './data_zipped',
                force = FORCE_DOWNLOAD)

print("Extracting FastText word embeddings...")
with ZipFile('data_zipped/word_vecs.zip', 'r') as data_zipped:
    data_zipped.extractall(path = "data/")
print("Done extracting FastText word embeddings!")


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = torch.FloatTensor(list(map(float, tokens[1:])))
    return data

print("Processing the word vectors and pickling them... (this might take some time)")
text_dictionary = load_vectors('data/wiki-news-300d-1M.vec')
pickle.dump(text_dictionary, open('pickles/word_embeddings/word_embeddings_tensors.p', 'wb'))
print("Done processing word vectors!")

print("Downloading NUSWIDE...(this will take a lot of time)")
fetch_and_cache(data_url = 'https://s3-us-west-2.amazonaws.com/multimedia-berkeley/Flickr.tar.gz',  
                file = "flickr.tar.gz", 
                data_dir = "./data_zipped",
                force = FORCE_DOWNLOAD)
      
print("Extracting NUSWIDE... (this might take some time)")
image_data = tarfile.open("./data_zipped/flickr.tar.gz")
image_data.extractall(path='./data')
print("Done extracting NUSWIDE!")

print("Finished setup!")
