from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import requests
import os
import json

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    "image": set(['jpg', 'jpeg',])
}
SERVER_URL = "http://0.0.0.0:5000"

def make_request(request, modality):
    data = {}

    if "num_results" in request.values and request.values["num_results"]:
        data["num_results"] = int(request.values["num_results"])
    else:
        data["num_results"] = 30

    query_url = f"{SERVER_URL}/query/"

    if "text" == modality:
        query_input = request.values["text"]
        data["text"] = request.values["text"]
        r = requests.post(query_url, data = data)
    elif "image" == modality:
        if "file" in request.files:
            f = request.files["file"]
            if f.filename: #and allowed_file(f.filename.strip(), ALLOWED_EXTENSIONS["image"]):
                filename = secure_filename(f.filename.strip())
                query_input = SERVER_URL + "/uploads/" + filename
                files = {"file": (filename, f)}
                r = requests.post(query_url, files = files, data = data)
            else:
                raise ValueError("Filename '{}' not supported".format(f.filename))
        else:
            raise ValueError("File not found")
    elif "dataset" == modality:
        query_input = request.values["query_input"]
        data = {
            "dataset": request.values["dataset"],
            "target": request.values["target"],
            "model": request.values["model"],
            "post_processing": request.values["post_processing"],
            "num_results": request.values["num_results"]
        }
        r = requests.post(query_url, data = data)
    else:
        raise ValueError("Modality '{}' not supported".format(modality))

    try:
        return r.json(), query_input
    except:
        raise RuntimeError("Error decoding response from server: {}".format(r.content.decode()))

@app.route("/")
def home():
    print("\n\nHome page request")
    info = requests.get(f"{SERVER_URL}/info", data = {"supported_modalities": True}).json()
    data = {"supported_modalities" : info["supported_modalities"]}
    return render_template("home.html", data = data)

@app.route("/query/<modality>", methods=["GET", "POST"])
def query(modality):
    if request.method == "POST":
        try:
            response, query_input = make_request(request, modality)
            data = {
                "engine_url": SERVER_URL,
                "input_modality": response["input_modality"],
                "valid_indexes": response["valid_indexes"],
                "num_results": response["num_results"],
                "query_input": query_input,
                "results": response["results"],
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
    print("SERVER_URL: ", SERVER_URL)
    code = requests.get(f"{SERVER_URL}/info", data={"alive":True}).status_code
    print(f"Server responded with status code: {code}")
    app.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "5001"))
    )
