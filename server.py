from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import PIL
import pickle
import os
import traceback
import json
import logging

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from search import SearchEngine
from networks import FeatureExtractor

EMBEDDING_DIR = ""
UPLOAD_DIR = "uploads/"
DATA_DIR = "data/"
IMAGE_DIR = 'data/Flickr'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
FAST_TEXT = None
EMBED_DIR = 'embeddings/'
BATCH_SIZE = 128
LOAD_EMBED = True

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_engine(app):
    # Instantiate SearchEngine Object
    search_engine = SearchEngine(["text", "image"], save_directory = EMBED_DIR, verbose = True)

    # Add models
    # hard-coded numerical values are specific to each model
    print("Loading models")
    resnet152_15 = pickle.load(open("pickles/models/entire_nuswide_model_15.p", "rb"))
    resnet152_5 = pickle.load(open("pickles/models/entire_nuswide_model_5.p", "rb"))
    resnet18_5 = pickle.load(open("pickles/models/entire_nuswide_model_5-18.p", "rb"))
    
    print("Adding models")
    search_engine.add_model(
        name = "ResNet152_15", 
        modalities = ["image","text"], 
        embedding_nets = [resnet152_15.modalityOneNet, resnet152_15.modalityTwoNet],
        input_dimensions= [(2048,), (300,)], 
        output_dimension = 64, 
        desc = "ResNet152 model trained with 15 epochs")
    search_engine.add_model(
        name = "ResNet152_5", 
        modalities = ["image","text"], 
        embedding_nets = [resnet152_5.modalityOneNet, resnet152_5.modalityTwoNet],
        input_dimensions= [(2048,), (300,)], 
        output_dimension = 64, 
        desc = "ResNet152 model trained with 5 epochs")
    search_engine.add_model(
        name = "ResNet18_5", 
        modalities = ["image","text"], 
        embedding_nets = [resnet18_5.modalityOneNet, resnet18_5.modalityTwoNet],
        input_dimensions= [(512,), (300,)], 
        output_dimension = 64, 
        desc = "ResNet18 model trained with 5 epochs")
    
    # Since our datasets will use nus-wide feature vectors as the dataset instead of PIL.Image,
    # we need to add preprocessing feature extractor to deal with images received as a search input
    print("Adding preprecessors")
    search_engine.models["ResNet152_15"].add_preprocessing("image", FeatureExtractor("resnet152").get_embedding)
    search_engine.models["ResNet152_5"].add_preprocessing("image", FeatureExtractor("resnet152").get_embedding)
    search_engine.models["ResNet18_5"].add_preprocessing("image", FeatureExtractor("resnet18").get_embedding)
    
    
    # Importing datasets...
    print("Loading datasets")
    image_from_idx = [i[0] for i in ImageFolder(IMAGE_DIR).samples] # Gets all filenames (targets)
    
    # Feature vectors from ResNet18
    image_data18 = np.array([])
    directory_18 = "data/nuswide_features/resnet18/"
    filenames = sorted(["{}/{}".format(directory_18, filename) 
                        for filename in os.listdir(directory_18) if filename[-3:] == "npy"])
    for filename in filenames:
        image_data18 = np.append(image_data18, np.load(filename,))
    image_data18.resize(len(image_data18) // 512, 512) # resizing one long 1D vector into appropriate size
    image_data18 = torch.from_numpy(image_data18).cuda().float()
    
    # Feature vectors from ResNet152
    image_data152 = np.array([])
    directory_152 = "data/nuswide_features/resnet152/"
    filenames = sorted(["{}/{}".format(directory_152, filename) 
                        for filename in os.listdir(directory_152) if filename[-3:] == "npy"])
    for filename in filenames:
        image_data152 = np.append(image_data152, np.load(filename).astype('float32'))
    image_data152.resize(len(image_data152) // 2048, 2048)
    image_data152 = torch.from_numpy(image_data152).cuda().float()
    
    # Fasttext
    global FAST_TEXT
    FAST_TEXT = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
    fast_text = FAST_TEXT
    text_from_idx = [None] * len(fast_text)
    text_data = [None] * len(fast_text)
    for idx, (key, value) in enumerate(fast_text.items()):
        text_from_idx[idx] = key
        text_data[idx] = value.cuda()
        
    # Dataloaders (not really necessary if we've already extracted embeddings)
    image18_dataloader = DataLoader(image_data18, batch_size = BATCH_SIZE)
    image152_dataloader = DataLoader(image_data152, batch_size = BATCH_SIZE)
    text_dataloader = DataLoader(text_data, batch_size = BATCH_SIZE)
    
    # Finally adding the datasets to SearchEngine
    print("Adding datasets")
    search_engine.add_dataset(
        name = "nus-wide_18", 
        data = image18_dataloader, 
        targets = image_from_idx, 
        modality = "image", 
        dimension = (512,))
    search_engine.add_dataset(
        name = "nus-wide_152", 
        data = image152_dataloader, 
        targets = image_from_idx, 
        modality = "image", 
        dimension = (2048,))
    search_engine.add_dataset(
        name = "fast_text", 
        data = text_dataloader, 
        targets = text_from_idx, 
        modality = "text", 
        dimension = (300,))
    
    # Build Indexes
    print("Building Indexes")
    search_engine.build_index(dataset_name = "nus-wide_18", model_name = "ResNet18_5", load_embeddings = LOAD_EMBED)
    search_engine.build_index(dataset_name = "nus-wide_152", model_name = "ResNet152_5", load_embeddings = LOAD_EMBED)
    search_engine.build_index(dataset_name = "nus-wide_152", model_name = "ResNet152_15", load_embeddings = LOAD_EMBED)
    search_engine.build_index(dataset_name = "fast_text", model_name = "ResNet18_5", load_embeddings = LOAD_EMBED)
    search_engine.build_index(dataset_name = "fast_text", model_name = "ResNet152_5", load_embeddings = LOAD_EMBED)
    search_engine.build_index(dataset_name = "fast_text", model_name = "ResNet152_15", load_embeddings = LOAD_EMBED)
    
    # Done
    print("Finished initializing SearchEngine")
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
