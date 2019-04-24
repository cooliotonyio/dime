from flask import Flask, request, redirect, url_for, send_from_directory, jsonify

from werkzeug.utils import secure_filename
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.cuda import is_available
from torch.utils.data import DataLoader

import PIL
import pickle
import os
import json

from search import SearchEngine

# FAST_TAG = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
FAST_TAG = {'hello':123123}
EMBEDDING_DIR = ""
UPLOAD_DIR = "./uploads"
DATA_DIR = "./data"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_engine(app):
    # Build Networks
    print("Building networks...")
    text_net = pickle.load(open("pickles/models/entire_nuswide_model.p", "rb"))
    def get_text_embedding(*data):
        return text_net.text_embedding_net(data[0])
    text_net.get_embedding = get_text_embedding
    image_net = pickle.load(open("pickles/models/entire_nuswide_model.p", "rb"))

    # Build Datasets
    print("Building datasets...")
    image_directory = 'data/Flickr'
    image_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image_data = ImageFolder('data/Flickr', transform = image_transform)
    image_from_idx = [i[0] for i in image_data.samples]
    fast_tag = FAST_TAG
    text_from_idx = [None] * len(fast_tag)
    text_data = [None] * len(fast_tag)
    for idx, (key, value) in enumerate(fast_tag.items()):
        text_from_idx[idx] = key
        text_data[idx] = (value, idx)

    # Build DataLoaders
    print("Building dataloaders...")
    batch_size = 128
    cuda = is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    from torch.utils.data import DataLoader
    image_dataloader = DataLoader(image_data, batch_size = batch_size, **kwargs)
    text_dataloader = DataLoader(text_data, batch_size = batch_size, **kwargs)

    #Building SearchEngine
    print("Building search engine")
    save_directory = './embeddings'
    search_engine = SearchEngine(["text", "image"], cuda = cuda, save_directory = save_directory, verbose = True)
    search_engine.add_model(text_net, "text_net", "text", (300,) , 30)
    search_engine.add_model(image_net, "image_net", "image", (3, 224, 224), 30)
    search_engine.add_dataset("wiki_word2vec", text_dataloader, text_from_idx, "text", (300,))
    search_engine.add_dataset("nus-wide", image_dataloader, image_from_idx, "image", (3, 224, 224))

    #Build Indexes
    print("Building Indexes")
    search_engine.build_index("wiki_word2vec")
    search_engine.build_index("nus-wide")

    #Finished
    app.search_engine = search_engine

def target_to_tensor(target, modality):
    search_engine = app.search_engine
    if "text" == modality:
        if target in FAST_TAG:
            tensor = FAST_TAG[target]
        else:
            raise KeyError("No tensor representation of '{}'".format(str(target)))
    elif "image" == modality:
        image = PIL.Image.open(target)
        image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tensor = image_transform(image).detach().to("cuda")
    elif "index" == modality:
        index_key, idx = json.dump(target)
        assert len(index_key) == 3, "Target {} not a valid index_key".format(target)
        assert index_key in search_engine.indexes, "Target {} not a valid index_key".format(target)
        new_target = search_engine.data_from_idx(index_key, [idx])[0]
        new_modality = search_engine.dataset[index_key[0]].modality
        tensor, modality = target_to_tensor(new_target, new_modality)
    return tensor, modality

def search(target, modality, n=5):
    search_engine = app.search_engine
    tensor, modality = target_to_tensor(target, modality)
    print("SHAPE:   \t", tensor.shape)
    print("MODALITY:\t", modality)
    model_name = search_engine.valid_models(tensor, modality)[0]
    if modality == "image":
        #TODO: Make this less dumb
        tensor = tensor[None,:,:,:]
    embedding = search_engine.models[model_name].get_embedding(tensor)
    if modality == "image":
        #TODO: Make this less dumb
        embedding = embedding[0]
    results = []
    for index_key in search_engine.valid_indexes(embedding):
        dis, idx = search_engine.search(embedding, index_key, n = n)
        result = {
            "index_key": index_key,
            "dataset": index_key[0],
            "dis": [float(d) for d in dis],
            "idx": [int(i) for i in idx],
            "data": search_engine.data_from_idx(index_key, idx),
            "modality": search_engine.datasets[index_key[0]].modality,
            "num_results": n
        }
        results.append(result)
    return results
            
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

@app.route('/query/<modality>', methods=['POST','GET'])
def query(modality):
    try:
        if modality == "text":
            target = request.form["text"]
        elif modality == "image":
            if 'file' in request.files:
                f = request.files['file']
                if f.filename and allowed_file(f.filename):
                    target = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                    f.save(target)
                else:
                    raise RuntimeError("Filename '{}' not allowed".format(str(f.filename)))
            else:
                raise RuntimeError("No file attached to request")
        else:
            raise RuntimeError("Modality '{}' not supported".format(modality))
        results = search(target, modality, n=20)
        return jsonify(results)
    except Exception as e:
        return e.message
    
if __name__ == "__main__":
    init_engine(app)
    app.run(
        host=os.getenv('LISTEN', '0.0.0.0'),
        port=int(os.getenv('PORT', '80'))
    )
