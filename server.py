from flask import Flask, request, redirect, url_for, send_from_directory, jsonify

from werkzeug.utils import secure_filename
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import PIL
import pickle
import os
import traceback
import json
import logging

from search import SearchEngine
from networks import FeatureExtractor

EMBEDDING_DIR = ""
UPLOAD_DIR = "./uploads"
DATA_DIR = "./data"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
FAST_TEXT = None

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_engine(app):

    #Building SearchEngine
    print("Building search engine")
    save_directory = './embeddings'
    search_engine = SearchEngine(["text", "image"], save_directory = save_directory, verbose = True)

    # Build Networks
    print("Building networks...")
    resnet152_15 = pickle.load(open("pickles/models/entire_nuswide_model_15.p", "rb")).cpu().eval()
    search_engine.add_model(
        resnet152_15, "ResNet152-15", ["image","text"], [(2048,), (300,)], 64, 
        desc = "ResNet152 trained with 15 epochs")
    search_engine.models["ResNet152-15"].add_preprocessing("image", FeatureExtractor("resnet152").get_embedding)

    resnet152_5 = pickle.load(open("pickles/models/entire_nuswide_model_5.p", "rb")).cpu().eval()
    search_engine.add_model(
        resnet152_5, "ResNet152-5", ["image","text"], [(2048,), (300,)], 64, 
        desc = "ResNet152 trained with 5 epochs")
    search_engine.models["ResNet152-5"].add_preprocessing("image", FeatureExtractor("resnet152").get_embedding)


    resnet18_5 = pickle.load(open("pickles/models/entire_nuswide_model_5-18.p", "rb")).cpu().eval()
    search_engine.add_model(
        resnet152_15, "ResNet18-5", ["image","text"], [(512,), (300,)], 64, 
        desc = "ResNet18 trained with 5 epochs")
    search_engine.models["ResNet18-5"].add_preprocessing("image", FeatureExtractor("resnet18").get_embedding)
    

    # Build Datasets
    print("Building datasets...")
    image_directory = 'data/Flickr'
    image_from_idx = [i[0] for i in ImageFolder(image_directory).samples]

    image_data18 = np.array([])
    directory = "data/nuswide_features/resnet18/"
    filenames = sorted(["{}/{}".format(directory, filename) for filename in os.listdir(directory) if filename[-3:] == "npy"])
    for filename in filenames:
        image_data18 = np.append(image_data18, np.load(filename))

    image_data152 = np.array([])
    directory = "data/nuswide_features/resnet152/"
    filenames = sorted(["{}/{}".format(directory, filename) for filename in os.listdir(directory) if filename[-3:] == "npy"])
    for filename in filenames:
        image_data152 = np.append(image_data152, np.load(filename))

    global FAST_TEXT
    FAST_TEXT = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
    fast_text = FAST_TEXT
    text_from_idx = [None] * len(fast_text)
    text_data = [None] * len(fast_text)
    for idx, (key, value) in enumerate(fast_text.items()):
        text_from_idx[idx] = key
        text_data[idx] = (value, idx)

    # Build DataLoaders
    print("Building dataloaders...")
    batch_size = 128
    image18_dataloader = DataLoader(image_data18, batch_size = batch_size)
    image152_dataloader = DataLoader(image_data152, batch_size = batch_size)
    text_dataloader = DataLoader(text_data, batch_size = batch_size)
    search_engine.add_dataset("fast_text", text_dataloader, text_from_idx, "text", (300,))
    search_engine.add_dataset("nus-wide_18", image18_dataloader, image_from_idx, "image", (512,))
    search_engine.add_dataset("nus-wide_152", image152_dataloader, image_from_idx, "image", (2048,))

    #Build Indexes
    print("Building Indexes")
    search_engine.build_index("fast_text")
    search_engine.build_index("nus-wide")

    #Finished
    app.search_engine = search_engine

def target_to_tensor(target, modality):
    search_engine = app.search_engine
    if "text" == modality:
        if target in FAST_TEXT:
            tensor = FAST_TEXT[target]
        else:
            raise KeyError("No tensor representation of '{}' in text dataset".format(str(target)))
    elif "image" == modality:
        image = PIL.Image.open(target)
        image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tensor = image_transform(image).detach().to("cpu")
    elif "dataset" == modality:
        dataset_name, idx = target
        assert dataset_name in search_engine.datasets, "Target dataset '{}' not a valid dataset".format(dataset_name)
        new_target = search_engine.data_from_idx(dataset_name, idx)[0]
        new_modality = search_engine.datasets[dataset_name].modality
        tensor, modality = target_to_tensor(new_target, new_modality)
    else:
        raise ValueError("Modality '{}' not supported".format(modality))
    return tensor, modality

def search(target, modality, n=5):
    print("NEW SEARCH")
    search_engine = app.search_engine
    tensor, modality = target_to_tensor(target, modality)
    model_names = search_engine.valid_models(tensor, modality)
    models = [search_engine.models[model_name] for model_name in model_names]
    print(" -> Modality:    \t", modality)
    print(" -> Tensor Shape:\t", tensor.shape)
    print(" -> Target:      \t '{}'".format(target))
    print(" -> models:      \t {}".format(str(model_names)))
    if modality == "image":
        # TODO: Make this less dumb
        tensor = tensor[None,:,:,:]
        embeddings = [model.get_embedding(tensor)[0] for model in models]
    else:
        embeddings = [model.get_embedding(tensor) for model in models]
    all_results = []
    '''      
       Everything below this is not updated   
    '''
    for index_key in search_engine.valid_indexes(embedding):
        dis, idx = search_engine.search(embedding, index_key, n = n)
        result = {
            "dataset": index_key[0],
            "model": index_key[1],
            "is_binarized": index_key[2],
            "dis": [float(d) for d in dis],
            "idx": [int(i) for i in idx],
            "data": [str(x) for x in search_engine.data_from_idx(index_key, idx)],
            "modality": search_engine.datasets[index_key[0]].modality,
            "num_results": n
        }
        all_results.append(result)
    print("SEARCH FINISHED, RETURNING {} RESULTS FOR {} DATASETS".format(n, len(all_results)))
    return all_results, modality
            
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

@app.route('/query/<modality>', methods=["POST"])
def query(modality):
    try:
        # Determine modality and target
        if "text" == modality:
            target = request.values["text"]
        elif "image" == modality:
            if "file" in request.files:
                f = request.files["file"]
                if f.filename and allowed_file(f.filename):
                    target = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                    f.save(target)
                else:
                    raise RuntimeError("Filename '{}' not allowed".format(str(f.filename)))
            else:
                raise RuntimeError("No file attached to request")
        elif "dataset" == modality:
            target = [request.values["dataset"], int(request.values["target"])]
        else:
            raise RuntimeError("Modality '{}' not supported".format(modality))
        
        # Figure out how many results
        if "num_results" in request.values:
            num_results = int(request.values["num_results"])
        else:
            num_results = 10
        
        # Search and return results
        results, modality = search(target, modality, n=num_results)
        response = {
            "input_target": target,
            "input_modality": modality,
            "num_sets": len(results),
            "num_results": num_results,
            "results": results
        }
        return jsonify(response)
    
    except Exception as err:
        traceback.print_tb(err.__traceback__)
        print(str(err))
        return "Query Error: {}".format(str(err))
    
if __name__ == "__main__":
    init_engine(app)
    app.run(
        host=os.getenv('LISTEN', '0.0.0.0'),
        port=int(os.getenv('PORT', '80'))
    )
