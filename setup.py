import os
import io
import sys
import torch
import pickle
import tarfile
from zipfile import ZipFile

print("Making directories...", end="\t")
if not os.path.isdir("./data_zipped"):
    os.mkdir(os.fsencode('./data_zipped'))
if not os.path.isdir("./data"):
    os.mkdir(os.fsencode('./data'))
if not os.path.isdir("./pickles"):
    init = "./pickles"
    os.mkdir(os.fsencode(init))
    os.mkdir(os.fsencode(init + '/word_embeddings'))
    os.mkdir(os.fsencode(init + '/models'))
    os.mkdir(os.fsencode(init + '/nuswide_metadata'))
print("Done!")

print("Downloading NUSWIDE_metadata...")
if not os.path.isdir("./data/nuswide_metadata"):
    os.mkdir(os.fsencode("./data/nuswide_metadata"))
    os.system("wget -O data_zipped/nuswide_metadata/NUS_WID_Tags.zip http://dl.nextcenter.org/public/nuswide/NUS_WID_Tags.zip")
    os.system("wget -O data_zipped/nuswide_metadata/Groundtruth.zip http://dl.nextcenter.org/public/nuswide/Groundtruth.zip")
    os.system("wget -O data_zipped/nuswide_metadata/Concepts.zip http://dl.nextcenter.org/public/nuswide/ConceptsList.zip")
    with ZipFile('data_zipped/nuswide_metadata/NUS_WID_Tags.zip', 'r') as data_zipped:
        data_zipped.extractall(path = "data/nuswide_metadata/")
    with ZipFile('data_zipped/nuswide_metadata/Groudtruth.zip', 'r') as data_zipped:
        data_zipped.extractall(path = "data/nuswide_metadata/")
    with ZipFile('data_zipped/nuswide_metadata/Concepts.zip', 'r') as data_zipped:
        data_zipped.extractall(path = "data/nuswide_metadata/")
print("Done")

print("Running nuswide processing scripts to make pickles..")
os.system("python3 nuswide_processing_scripts/make_relevancy_matrix.py")
os.system("python3 nuswide_processing_scripts/make_tag_matrix.py")
os.system("python3 nuswide_processing_scripts/make_concepts.py")
print("Done")


print("Downloading the FastText word embeddings")
os.system("wget -O data_zipped/wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip")
os.system("unzip data_zipped/wiki-news-300d-1M.vec.zip -d data/")
print("Done")


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = torch.FloatTensor(list(map(float, tokens[1:])))
    return data

print("Processing the word vectors and pickling them...")
text_dictionary = load_vectors('data/wiki-news-300d-1M.vec')
pickle.dump(text_dictionary, open('pickles/word_embeddings/word_embeddings_tensors.p', 'wb'))
os.system("mv entire_nuswide_model.p pickles/models/")
print("Done")

print("Downloading and extracting NUSWIDE...")
if not os.path.isdir("./data/Flickr"):
    os.system("wget -O data_zipped/flickr.tar.gz https://s3-us-west-2.amazonaws.com/multimedia-berkeley/Flickr.tar.gz")
    print("Extracting...")
    image_data = tarfile.open("Flickr.tar.gz")
    image_data.extractall(path='./data')
print("Done")