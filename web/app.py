from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import requests
import os
import json

SERVER_URL = "http://0.0.0.0:5000"

app = Flask(__name__)

@app.route("/")
def index():
    info = requests.get(f"{SERVER_URL}/info", data = {"supported_modalities": True}).json()
    data = {
        "supported_modalities": info["supported_modalities"],
        "server_url": SERVER_URL
        }
    return render_template("index.html", data = json.dumps(data))

@app.route("/query",)
def query():
    if "modality" in request.values:
        try:
            data = {
                "modality": request.values["modality"],
                "target": request.values["target"],
                "index_name": request.values["index_name"],
                "num_results": request.values["num_results"],
                "server_url": SERVER_URL
            }
            if "dataset_name" in request.values:
                data["dataset_name"] = request.values["dataset_name"]
            return render_template("results.html", data=json.dumps(data))
        except Exception as err:
            print(err)
            return render_template("error.html", error = str(err))
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    print("SERVER_URL: ", SERVER_URL)
    code = requests.get(f"{SERVER_URL}/info", data={"alive":True}).status_code
    print(f"Server responded with status code: {code}")
    app.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "5001"))
    )
