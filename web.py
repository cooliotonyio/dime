from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import os
import traceback

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg',])
ENGINE_URL = "http://3.212.166.121"

def make_request(request, modality):
    data = {}
    if "num_results" in request.values:
        data["num_results"] = request.values["num_results"]
    else:
        data["num_results"] = 30

    query_url = ENGINE_URL + "/query/" + modality

    if "text" == modality:
        query_input = request.values["text"]
        data["text"] = request.values["text"]
        r = requests.post(query_url, data = data)
    elif "image" == modality:
        if "file" in request.files:
            f = request.files["file"]
            if f.filename and allowed_file(f.filename):
                query_input = ENGINE_URL + "/uploads/" + f.filename
                files = {"file": (f.filename, f)}
                r = requests.post(query_url, files = files, data = data)
            else:
                return "Filename '{}' not supported".format(f.filename)
        else:
            return "File not found"
    elif "dataset" == modality:
        query_input = request.values["query_input"]
        data["dataset"] = request.values["dataset"]
        data["target"] = request.values["target"]
        r = requests.post(query_url, data = data)
    else:
        return "Modality '{}' not supported".format(modality)
    return r.json(), query_input

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS   

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/query/<modality>", methods=["GET", "POST"])
def query(modality):
    if request.method == "POST":
        try:
            response, query_input = make_request(request, modality)
            results = response["results"]
            for result in results:
                if result["modality"] == "text":
                    tags = result["data"]
            results = [r for r in results if r["modality"] != "text"]
            return render_template("results.html", 
                modality = response["input_modality"], 
                query_input = query_input, 
                results = results, 
                tags = tags,
                num_datasets = len(results), 
                num_results = response["num_results"],
                engine_url = ENGINE_URL)
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            print(str(err))
            return str(err)
    elif request.method == "GET":
        return home()
    else:
        return "Method '{}' not supported".format(request.method)

if __name__ == "__main__":
    print("ALLOWED_EXTENSIONS: ", ALLOWED_EXTENSIONS)
    print("ENGINE_URL: ",ENGINE_URL)
    app.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "80"))
    )
