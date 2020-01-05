"""
Running this script will fetch and cache all necessary datasets

If datasets have already been downloaded to data_zipped, 
this script will only extract/process/pickle the downloaded dataset

If a re-download is desired, simple change force to True
"""
import os
import io
import torch
import pickle
import tarfile
import time
from zipfile import ZipFile
import requests
from pathlib import Path
from dime.model import load_model

##### PARAMS #####
FORCE_DOWNLOAD = False
BATCH_SIZE = 32
##################
def data():
    def fetch_and_cache(data_url, file, data_dir="data", force=False):
        """
        Download and cache a url and return the object file path.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok = True)
        file_path = data_dir / Path(file)
        if force and file_path.exists():
            file_path.unlink()
        if force or not file_path.exists():
            print('Downloading...', end=' ')
            resp = requests.get(data_url)
            with file_path.open('wb') as f:
                f.write(resp.content)
            print('Done!')
            last_modified_time = time.ctime(file_path.stat().st_mtime)
        else:
            last_modified_time = time.ctime(file_path.stat().st_mtime)
            print("Using cached version that was downloaded (UTC):", last_modified_time)
        return file_path

    start_time = time.time()
    print("Starting setup... This might take a while.")
    print("Making directories...", end=" ")
    if not os.path.isdir("./data_zipped"):
        os.mkdir(os.fsencode('./data_zipped'))
    if not os.path.isdir("./data"):
        os.mkdir(os.fsencode('./data'))
    print("Done!")

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
    pickle.dump(text_dictionary, open('data/wiki_word2vec.data.pkl', 'wb'))
    print("Done processing word vectors!")

    print("Downloading NUSWIDE...(this will take a lot of time)")
    fetch_and_cache(
        data_url = 'https://s3-us-west-2.amazonaws.com/multimedia-berkeley/Flickr.tar.gz',  
        file = "flickr.tar.gz", 
        data_dir = "./data_zipped",
        force = FORCE_DOWNLOAD)
        
    print("Extracting NUSWIDE... (this might take some time)")
    image_data = tarfile.open("./data_zipped/flickr.tar.gz")
    image_data.extractall(path='./data')
    print("Done extracting NUSWIDE!")

    print("Finished downloading data in {} seconds!".format(time.time() - start_time))

def setup():
    print("Starting setup...")
    engine_params = {
        "name": "demo_engine",
        "cuda": True,
        "verbose": True,
        "dataset_dir": "data/",
        "index_dir": "indexes/",
        "model_dir": "models/",
        "embedding_dir": "embeddings/",
        "modalities": ["text", "image", "audio", "video"]   
    }
    engine = SearchEngine(engine_params)
    print("Engine created")

    r152_params = {
        "name": "resnet152",
        "output_dim": (2048,),
        "modalities": ["image"],
        "embedding_nets": [torch.nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-1])],
        "input_dim": [(3, 224, 224)],
        "desc": "Resnet152 with the last layer removed for feature extraction"
    } 
    engine.add_model(r152_params)

    r18_params = {
        "name": "resnet18",
        "output_dim": (512,),
        "modalities": ["image"],
        "embedding_nets": [torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])],
        "input_dim": [(3, 224, 224)],
        "desc": "Resnet18 with the last layer removed for feature extraction"
    } 
    engine.add_model(r18_params)

    for model_name in ["cm-r18-15epochs", "cm-r152-5epochs", "cm-r152-15epochs"]:
        engine.vprint(f"Loading model '{model_name}'... ", end = "")
        start_time = time.time()
        model = load_model(engine, model_name)
        engine.models[model.name] = model
        for model_modality in model.modalities:
            engine.modalities[model_modality]["model_names"].append(model.name)
            engine.vprint(f"done in {round(time.time() - start_time, 4)} seconds!")

    engine.add_preprocessor("cm-r152-5epochs", "image", "resnet152")
    engine.add_preprocessor("cm-r152-15epochs", "image", "resnet152")
    engine.add_preprocessor("cm-r18-15epochs", "image", "resnet18")

    nuswide_params = {
        "name": "nuswide",
        "data_dir": "Flickr/",
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "dim": (3, 224, 224),
        "modality": "image",
        "desc": "The nuswide dataset"
    }
    engine.add_dataset(nuswide_params)

    with open("data/wiki_word2vec.data.pkl", "rb") as f:
        wiki = pickle.load(f)

    wiki_word2vec_params = {
        "name": "wiki_word2vec",
        "data": wiki,
        "modality": "text",
        "dim": (300,),
        "desc": "one million word2vec entries trained on English Wikipedia"
    }
    engine.add_dataset(wiki_word2vec_params)


    ind1= {
        "name": "NUSWIDE (cm-r152-5epochs)",
        "model_name": "cm-r152-5epochs",
        "dataset_name": "nuswide",
        "desc": "The index corresponding to cm-r152-5epochs model and nuswide"
    }
    ind2 = {
        "name": "WIKI WORD2VEC (cm-r152-5epochs)",
        "model_name": "cm-r152-5epochs",
        "dataset_name": "wiki_word2vec",
        "desc": "The index corresponding to cm-r152-5epochs model and wiki_word2vec"
    }
    ind3 = {
        "name": "NUSWIDE (cm-r152-15epochs)",
        "model_name": "cm-r152-15epochs",
        "dataset_name": "nuswide",
        "desc": "The index corresponding to cm-r152-15epochs model and nuswide"
    }
    ind4 = {
        "name": "WIKI WORD2VEC (cm-r152-15epochs)",
        "model_name": "cm-r152-15epochs",
        "dataset_name": "wiki_word2vec",
        "desc": "The index corresponding to cm-r152-15epochs model and wiki_word2vec"
    }
    ind5 = {
        "name": "NUSWIDE (cm-r18-15epochs)",
        "model_name": "cm-r18-15epochs",
        "dataset_name": "nuswide",
        "desc": "The index corresponding to cm-r18-15epochs model and nuswide"
    }
    ind6 = {
        "name": "WIKI WORD2VEC (cm-r18-15epochs)",
        "model_name": "cm-r18-15epochs",
        "dataset_name": "wiki_word2vec",
        "desc": "The index corresponding to cm-r18-15epochs model and wiki_word2vec"
    }
    ind7 = {
        "name": "NUSWIDE (resnet152)",
        "model_name": "resnet152",
        "dataset_name": "nuswide",
        "desc": "The index corresponding to resnet152 and nuswide"
    }
    ind8 = {
        "name": "NUSWIDE (resnet18)",
        "model_name": "resnet18",
        "dataset_name": "nuswide",
        "desc": "The index corresponding to resnet18 and nuswide"
    }

    inds = [ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8]

    for ind in inds:
        engine.build_index(ind1, batch_size = BATCH_SIZE)

    print("Saving engine")
    engine.save(save_data=True)

if __name__ == "__main__":
    data()
    setup()