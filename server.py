from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import os

app = Flask(__name__)

EMBEDDING_DIR = ""
UPLOAD_DIR = "./uploads"
DATA_DIR = "./data"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
WORD2VEC = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))

def init_engine(app):
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
    search_engine.add_model(image_net, "image_net", "image", (3, 244, 244), 30)
    search_engine.add_dataset("wiki_word2vec", text_dataloader, text_from_idx, "text", (300,))
    search_engine.add_dataset("nus-wide", image_dataloader, image_from_idx, "image", (3, 244, 244))

    #Build Indexes
    print("Building Indexes")
    search_engine.build_index("wiki_word2vec")
    search_engine.build_index("nus-wide")

    #Finished
    app.config['ENGINE'] = search_engine

def image_to_tensor(filename):
    image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = PIL.Image.open(filename).convert('RGB')
    tensor = image_transform(image)[None,:,:,:]
    return tensor

def text_to_tensor(text):
    return WORD2VEC[text]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search(target, modality):
    results = {}
    if modality == "text":
        text = target
        tensor = text_to_tensor(text)
        model_name = search_engine.valid_models(tensor, "text")[0]
        embedding = search_engine.get_embedding(tensor, model_name)
        indexes_keys = search_engine.valid_indexes(embedding)
        for key in indexes_keys:
            d, i = search_engine.search(embedding, key)
            resultes[key] = {
                distances: d,
                indicies: i
            }
            

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

@app.route('/query/<modality>', methods=['GET', 'POST'])
def query(modality):
    search_engine = app.config["ENGINE"]
    if request.method == "POST":
        if modality == "text":
            results = search(request.args["text"], "text")
            print(result)
            return render_template('results.html', modality = modality, input = text)
        elif modality == "image":
            if 'file' in request.files:
                f = request.files['file']
                if f.filename and allowed_file(f.filename):
                    filename = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                    f.save(filename)
                    return render_template('results.html', modality = modality, input = filename)
        elif modality == "audio":
            return "audio"
    return redirect(url_for('home'))

# if __name__ == "__main__":
#     app.run()


init_engine(app)