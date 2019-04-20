from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import pickle
import os

app = Flask(__name__)

def init_engine():
    # Build Networks
    print("Building networks...")
    import pickle
    text_net = pickle.load(open("pickles/models/entire_nuswide_model.p", "rb"))
    def get_text_embedding(*data):
        return text_net.text_embedding_net(data[0])
    text_net.get_embedding = get_text_embedding
    image_net = pickle.load(open("pickles/models/entire_nuswide_model.p", "rb"))

    # Build Datasets
    print("Building datasets...")
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    image_directory = 'data/Flickr'
    image_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image_data = ImageFolder('data/Flickr', transform = image_transform)
    image_from_idx = [i[0] for i in image_data.samples]
    word2vec_dict = WORD2VEC
    text_from_idx = [None] * len(word2vec_dict)
    text_data = [None] * len(word2vec_dict)
    for idx, (key, value) in enumerate(word2vec_dict.items()):
        text_from_idx[idx] = key
        text_data[idx] = (value, idx)

    # Build DataLoaders
    print("Building dataloaders...")
    from torch.cuda import is_available
    batch_size = 128
    cuda = is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    from torch.utils.data import DataLoader
    image_dataloader = DataLoader(image_data, batch_size = batch_size, **kwargs)
    text_dataloader = DataLoader(text_data, batch_size = batch_size, **kwargs)

    #Building SearchEngine
    print("Building search engine")
    from search import SearchEngine
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
    return search_engine

def target_to_tensor(target, modality):
    if modality == "image":
        from torchvision import transforms
        import PIL
        image = PIL.Image.open(target)
        image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tensor = image_transform(image)
    elif modality == "text":
        tensor = WORD2VEC[target]
    return tensor

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search(target, modality, n=5):
    tensor = target_to_tensor(target, modality).detach().to("cuda")
    print("SHAPE:\t", tensor.shape)
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
        data = search_engine.data_from_idx(index_key, idx)
        dis = [float(d) for d in dis]
        dataset = search_engine.datasets[index_key[0]]
        results.append([index_key, list(dis), list(data), dataset.modality])
    return results
            
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

@app.route('/query/<modality>', methods=['POST','GET'])
def query(modality):
    if modality == "text":
        text = request.form["text"]
        if text in WORD2VEC:
            results = search(text, "text")
            return jsonify(results)
        else:
            print("WORD NOT IN DICTIONARY")
    elif modality == "image":
        if 'file' in request.files:
            f = request.files['file']
            print("FILE FOUND")
            if f.filename and allowed_file(f.filename):
                filename = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                f.save(filename)
                results = search(filename, "image")
                return jsonify(results)
        else:
            print("NO FILE IN REQUEST")
    elif modality == "audio":
        return "audio not supported"
    return "NOT SUPPORTED"
    


EMBEDDING_DIR = ""
UPLOAD_DIR = "./uploads"
DATA_DIR = "./data"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
WORD2VEC = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
search_engine = init_engine()
if __name__ == "__main__":
    app.run(
        host=os.getenv('LISTEN', '0.0.0.0'),
        port=int(os.getenv('PORT', '80'))
    )
