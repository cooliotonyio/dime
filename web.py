from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg',])
ENGINE_URL = "http://ec2-3-212-166-121.compute-1.amazonaws.com"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS   

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/query/<modality>', methods=['GET', 'POST'])
def query(modality):
    if request.method == "POST":
        engine_url = ENGINE_URL + "/query/" + modality
        if modality == "text":
            r = requests.post(engine_url, data = {modality : request.form[modality]})
            print(r.json())
            results = r.json()
            return render_template('results.html', 
                modality=modality, 
                input=request.form[modality], 
                results = results, 
                num_datasets = len(results), 
                num_results = 5,
                engine_url = ENGINE_URL)
        elif modality == "image":
            if 'file' in request.files:
                f = request.files['file']
                if f.filename and allowed_file(f.filename):
                    r = requests.post(engine_url, files = {'file' : (f.filename, f)})
                    print(r.json())
                    input_url = ENGINE_URL+"/uploads/"+f.filename
                    results = r.json()
                    return render_template('results.html', 
                        modality=modality, 
                        input=input_url, 
                        results = results, 
                        num_datasets = len(results), 
                        num_results = 5,
                        engine_url = ENGINE_URL)
        elif modality == "audio":
            return "audio not supported"
        elif modality == "video":
            return "video not supported"
    return "EH"

if __name__ == "__main__":
    app.run(
        host=os.getenv('LISTEN', '0.0.0.0'),
        port=int(os.getenv('PORT', '81'))
    )
