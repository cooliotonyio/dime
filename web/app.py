from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import os
import json

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
            raise ValueError("File not found")
    elif "dataset" == modality:
        query_input = request.values["query_input"]
        data = {
            "dataset": request.values["dataset"],
            "target": request.values["target"],
            "model": request.values["model"],
            "binarized": request.values["binarized"],
            "num_results": request.values["num_results"]
        }
        r = requests.post(query_url, data = data)
    else:
        raise ValueError("Modality '{}' not supported".format(modality))

    try:
        return r.json(), query_input
    except:
        raise RuntimeError(r.content.decode())

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
            data = {
                "engine_url": ENGINE_URL,
                "input_modality": response["input_modality"],
                "num_datasets": response["num_sets"],
                "num_results": response["num_results"],
                "query_input": query_input,
                "results": response["results"]
            }
            return render_template("results.html", data = json.dumps(data))
        except Exception as err:
            print(err)
            return render_template("error.html", error = str(err))
    elif request.method == "GET":
        return home()
    else:
        error = "Method '{}' not supported".format(request.method)
        return render_template("error.html", error = error)

if __name__ == "__main__":
    print("ALLOWED_EXTENSIONS: ", ALLOWED_EXTENSIONS)
    print("ENGINE_URL: ",ENGINE_URL)
    app.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "80"))
    )
