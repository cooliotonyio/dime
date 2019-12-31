from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os

from dime.engine import load_engine
from dime.utils import allowed_file, in_and_true, sanitize_dict

ENGINE_NAME = "demo_engine.engine"
UPLOAD_DIR = "uploads/"
BATCH_SIZE = 32
ALLOWED_EXTENSIONS = {"image": set(["png", "jpg", "jpeg"])}

server = Flask(__name__)
engine = load_engine(ENGINE_NAME)

def search(target, modality, n=5, model = None):
    #TODO:
    results = []

    dis, idx = engine.search(tensor, tensor_modality, index_name, n = n, preprocessing=True)
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "post_processing": post_processing,
            "dis": [float(d) for d in dis],
            "idx": [int(i) for i in idx],
            "data": [str(x) for x in engine.idx.to_target(idx, index_key)],
            "modality": engine.datasets[dataset_name].modality,
            "num_results": n,
            "model_info": engine.models[model_name].get_info()
        })

    return results

def target_to_tensor(target, modality, dataset_name=None):
            # Determine modality and target
        if "text" == modality:
            target = request.values["text"]
        elif "image" == modality:
            if "file" in request.files:
                f = request.files["file"]
                if f.filename and allowed_file(f.filename, ALLOWED_EXTENSIONS[modality]):
                    target = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
                    f.save(target)
            else:
                raise RuntimeError("No file attached to request")
        else:
            raise RuntimeError("Modality '{}' not supported in query".format(modality))

            
@server.route("/uploads/<path:filename>")
def handle_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@server.route("/data/<path:filename>")
def handle_data(filename):
    return send_from_directory(server.engine.dataset_dir, filename, as_attachment=True)

@server.route("/info/")
def handle_info():
    info = {}

    # Available
    if in_and_true("available_indexes", request.values):
        modality = request.values["available_indexes"]
        info["available_index_names"] = engine.valid_index_names(modality)
    if in_and_true("available_models", request.values):
        info["available_model_names"] = [m for m in engine.models.values() if modality in m.modalities]

    # Listing all 
    if in_and_true("all_dataset", request.values):
        info["all_dataset_names"] = list(engine.datasets.keys())
    if in_and_true("all_indexes", request.values):
        info["all_index_names"] = list(engine.indexes.keys())
    if in_and_true("all_models", request.values):
        info["all_model_names"] = list(engine.models.keys())

    # Params info
    if in_and_true("dataset_params", request.values):
        dataset = engine.datasets[request.values["dataset_params"]]
        info["dataset_params"] = sanitize_dict(dataset.params)
    if in_and_true("index_params". request_values):
        index = engine.indexes[request.values["index_params"]]
        info["index_params"] = sanitize_dict(index.params)
    if in_and_true("model_params". request_values):
        model = engine.models[request.values["model_params"]]
        info["model_params"] = sanitize_dict(model.params) 
    if in_and_true("engine_params", request.values):
        info["engine_params"] = sanitize_dict(engine.params)


@server.route("/query/", methods=["POST"])
def handle_query():
    """
    Returns page of results based on request

    request.values = {
        "modality",
        "index_name",
        "target",
        "n",
        "dataset_name" (optional)
    }
    """
    modality = request.values["modality"]
    try:

        tensor = target_to_tensor
        
        # Figure out how many results
        n = int(request.values["n"])
        
        # Search and return results
        results, valid_indexes, input_modality = search(target, modality, n = num_results, model = model)
        response = {
            "input_target": target,
            "input_modality": input_modality,
            "valid_indexes": valid_indexes,
            "n": n,
            "results": results
        }
        return jsonify(response)
    except Exception as err:
        return "Query Error: {}".format(str(err))
    
    
if __name__ == "__main__":
    app.engine = load_engine(ENGINE_NAME)
    app.run(
        host=os.getenv("LISTEN", "0.0.0.0"),
        port=int(os.getenv("PORT", "80"))
    )
