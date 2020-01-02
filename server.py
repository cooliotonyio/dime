from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging

#TODO fix these
from dime.engine import load_engine
from dime.utils import allowed_file, in_and_true, sanitize_dict

logging.getLogger('werkzeug').setLevel(logging.ERROR)
ENGINE_NAME = "demo_engine.engine"
UPLOAD_DIR = "uploads/"
ALLOWED_EXTENSIONS = {"image": set(["png", "jpg", "jpeg"])}

server = Flask(__name__)
CORS(server)
engine = load_engine(ENGINE_NAME)

def handle_search(request, engine):
    """
    Search through a given index and return results
    
    Parameters:
    target (str): The target to use as query
    modality (str): Modality of the target
    index_name (str): Name of the index to search in
    num_results (int): Number of results to return
    """
    print("Handling search...")

    target = request.values["target"]
    modality = request.values["modality"]

    print("Target:", target)
    print("Modality:", modality)

    if "num_results" in request.values:
        num_results = int(request.values["num_results"])
    else:
        num_results = 30

    if "index_name" in request.values:
        index = engine.indexes[request.values["index_name"]]
    else:
        #TODO: get a random compatible index
        pass

    print("Index:", index.name)

    # Process target and convert to tensor
    if "text" == modality:
        tensor = engine.target_to_tensor(target, modality = modality)
    elif "image" == modality:
        if "file" in request.files:
            f = request.files["file"]
            if f.filename and allowed_file(f.filename, ALLOWED_EXTENSIONS[modality]):
                target = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                tensor = engine.target_to_tensor(target, modality = modality)
                f.save(target)
        else:
            raise RuntimeError("No file attached to request")
    elif "dataset" == modality:
        dataset = engine.datasets[request.values["dataset_name"]]
        modality = dataset.modality
        target = engine.idx_to_target(request.values["target"], dataset.name)
        tensor = engine.target_to_tensor(target, dataset_name = dataset.name)
    else:
        raise RuntimeError(f"Modality '{modality} not supported")

    dis, idx = engine.search(tensor, modality, index.name, n = num_results, preprocessing=True)

    results = {
        "target": target,
        "dataset_name": index.dataset_name,
        "model_name": index.model_name,
        "index_name": index.name,
        "post_processing": index.post_processing,
        "dis": [float(d) for d in dis],
        "idx": [int(i) for i in idx],
        "results": [str(x) for x in engine.idx_to_target(idx, index.name)],
        "modality": modality,
        "num_results": num_results,
    }
    print("Search handled successfully.")
    return results
            
@server.route("/uploads/<path:filename>")
def handle_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@server.route("/data/<path:filename>")
def handle_data(filename):
    return send_from_directory(engine.dataset_dir, filename, as_attachment=True)

@server.route("/info")
def handle_info():
    info = {}
    # Available
    if in_and_true("available_indexes", request.values):
        modality = request.values["available_indexes"]
        info["available_indexes"] = engine.valid_index_names(modality)
    if in_and_true("available_models", request.values):
        info["available_models"] = [m for m in engine.models.values() if modality in m.modalities]

    # Listing all 
    if in_and_true("all_datasets", request.values):
        info["all_datasets"] = list(engine.datasets.keys())
    if in_and_true("all_indexes", request.values):
        info["all_indexes"] = list(engine.indexes.keys())
    if in_and_true("all_models", request.values):
        info["all_models"] = list(engine.models.keys())

    # Params info
    if in_and_true("dataset_params", request.values):
        dataset = engine.datasets[request.values["dataset_params"]]
        info["dataset_params"] = sanitize_dict(dataset.params)
    if in_and_true("index_params", request.values):
        index = engine.indexes[request.values["index_params"]]
        info["index_params"] = sanitize_dict(index.params)
    if in_and_true("model_params", request.values):
        model = engine.models[request.values["model_params"]]
        info["model_params"] = sanitize_dict(model.params) 
    if in_and_true("engine_params", request.values):
        info["engine_params"] = sanitize_dict(engine.params)

    # Misc
    if in_and_true("supported_modalities", request.values):
        info["supported_modalities"] = list(engine.modalities.keys())
    if in_and_true("alive", request.values):
        info["alive"] = True
    
    return jsonify(sanitize_dict(info))

@server.route("/query", methods=["POST"])
def handle_query():
    """
    Returns page of results based on request

    request.values = {
        "modality",
        "target",
        "num_results",
    }
    """
    print("\n\nRECEIVED QUERY")
    if in_and_true("target", request.values) and in_and_true("modality", request.values):
        response = {
            "initial_target": request.values["target"],
            "initial_modality": request.values["modality"],
        }
        try:
            response["results"] = handle_search(request, engine)
        except Exception as e:
            response["error"] = str(e.__repr__())
        return jsonify(response)
    else:
        return "Request missing either 'target' or 'modality'", 400

if __name__ == "__main__":
    server.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000"))
    )
