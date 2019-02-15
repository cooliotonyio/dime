import os
import io
import sys
import torch
import pickle
import tarfile
from util import fetch_and_cache

print("Making pickles directory tree...")
if not os.path.isdir("./pickles"):
    init = "./pickles"
    os.mkdir(os.fsencode(init))
    os.mkdir(os.fsencode(init + '/word_embeddings'))
    os.mkdir(os.fsencode(init + '/models'))
    os.mkdir(os.fsencode(init + '/nuswide_metadata'))
print("Done")

print("Downloading and extracting NUSWIDE...")
if not os.path.isdir("./Flickr"):
    image_data_url = 'https://s3-us-west-2.amazonaws.com/multimedia-berkeley/Flickr.tar.gz'
    image_data_filename = 'nus-wide.tar.gz'
    fetch_and_cache(data_url = image_data_url, data_dir = ".", file = image_data_filename, force = False)
    image_data = tarfile.open("./" + image_data_filename)
    image_data.extractall()
    os.system("rm nus-wide.tar.gz")
print("Done")


print("Downloading NUSWIDE_metadata...")
if not os.path.isdir("./nuswide_metadata"):
    os.mkdir(os.fsencode("./nuswide_metadata"))
    os.system("wget -O nuswide_metadata/NUS_WID_Tags.zip http://dl.nextcenter.org/public/nuswide/NUS_WID_Tags.zip")
    os.system("unzip nuswide_metadata/NUS_WID_Tags.zip -d nuswide_metadata/")
    os.system("rm nuswide_metadata/NUS_WID_Tags.zip")

    os.system("wget -O nuswide_metadata/Groundtruth.zip http://dl.nextcenter.org/public/nuswide/Groundtruth.zip")
    os.system("unzip nuswide_metadata/Groundtruth.zip -d nuswide_metadata/")
    os.system("rm nuswide_metadata/Groundtruth.zip")

    os.system("wget -O nuswide_metadata/Concepts.zip http://dl.nextcenter.org/public/nuswide/ConceptsList.zip")
    os.system("unzip nuswide_metadata/Concepts.zip -d nuswide_metadata/")
    os.system("rm nuswide_metadata/Concepts.zip")
print("Done")

print("Running nuswide processing scripts to make pickles..")
os.system("python3 nuswide_processing_scripts/make_relevancy_matrix.py")
os.system("python3 nuswide_processing_scripts/make_tag_matrix.py")
os.system("python3 nuswide_processing_scripts/make_concepts.py")
print("Done")


print("Downloading the FastText word embeddings")
os.system("wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip")
print("Done")

os.system("unzip wiki-news-300d-1M.vec.zip")

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = torch.FloatTensor(list(map(float, tokens[1:])))
    return data

print("Processing the word vectors and pickling them...")
text_dictionary = load_vectors('wiki-news-300d-1M.vec')

pickle.dump(text_dictionary, open('pickles/word_embeddings/word_embeddings_tensors.p', 'wb'))
os.system("rm wiki-news-300d-1M.vec")
os.system("rm wiki-news-300d-1M.vec.zip")

os.system("mv entire_nuswide_model.p pickles/models/")
print("Done")
