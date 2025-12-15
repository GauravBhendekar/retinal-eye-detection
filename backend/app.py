
from flask import Flask, jsonify, send_file
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib
matplotlib.use("Agg")  # <-- avoid Tkinter errors in Flask threads
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import io
import os
import seaborn as sns
import json
import gc
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
from flask_cors import CORS


# ---------------- GPU memory growth ----------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# ======================================================
# Flask App
# ======================================================
app = Flask(__name__)
CORS(app)


class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Load histories for plotting
with open('training_history.pkl', 'rb') as f:
    hist1 = pd.DataFrame(pickle.load(f))
with open('training_history_resnet.pkl', 'rb') as f:
    hist2 = pd.DataFrame(pickle.load(f))
with open('training_history_eff.pkl', 'rb') as f:
    hist3 = pd.DataFrame(pickle.load(f))
with open('training_history_CNN_acc.pkl', 'rb') as f:
    hist4 = pd.DataFrame(pickle.load(f))

# ------------------------------------------------------
# utility: load model only when needed
# ------------------------------------------------------
def load_model_by_name(name):
    name = name.lower()
    if name == "mobilenet":
        model = tf.keras.models.load_model('trained_eye_model_mobnet.h5', compile=False)
    elif name == "resnet":
        model = tf.keras.models.load_model('trained_eye_model_resnet.h5', compile=False)
    elif name == "cnn":
        model = tf.keras.models.load_model('trained_eye_model_CNN_acc.h5', compile=False)
    elif name == "efficientnet":
        INPUT_SHAPE = (240, 240, 3)
        efficient = tf.keras.applications.EfficientNetB1(
            include_top=True, weights="imagenet",
            input_shape=INPUT_SHAPE, classifier_activation="softmax"
        )
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=INPUT_SHAPE),
            efficient,
            tf.keras.layers.Dense(4, activation="softmax")
        ])
        model.load_weights('trained_eye_model_eff_weights.h5')
    else:
        raise ValueError("Invalid model name")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------------------------------
# Batched evaluator (safe memory usage)
# ------------------------------------------------------
def evaluate_and_save(model, model_name, test_csv="test/test_labels.csv", batch_size=4):
    """
    Evaluate model in safe batches, save confusion matrix png and JSON report.
    batch_size should be small for big models like ResNet (e.g. 2-4).
    """
    if not os.path.exists(test_csv):
        return {"error": "test_labels.csv not found"}

    df = pd.read_csv(test_csv)  # columns: filename,label
    y_true, y_pred = [], []

    print(f"\n🔍 Evaluating {model_name.upper()} ...")

    if model_name.lower() in ["mobilenet"]:
        target_size = (224, 224)
    elif model_name.lower() in ["resnet"]:
        target_size = (224, 224)
    elif model_name.lower() in ["cnn"]:
        target_size = (224, 224)
    else:
        target_size = (240, 240)

    images, labels, filenames = [], [], []

    for idx, row in df.iterrows():
        path = os.path.join("test", row['label'], row['filename'])
        if not os.path.exists(path):
            continue

        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        if model_name.lower() == "cnn":
            img_array = img_array / 255.0
        images.append(img_array)
        labels.append(row['label'])
        filenames.append(row['filename'])

        if len(images) >= batch_size:
            batch = np.array(images)
            preds = model.predict(batch, verbose=0)
            for j, pred in enumerate(preds):
                pred_class = class_names[np.argmax(pred)]
                y_pred.append(pred_class)
                y_true.append(labels[j])
                print(f"Processing {filenames[j]} - True: {labels[j]} | Pred: {pred_class}")
            images, labels, filenames = [], [], []

    # process remaining
    if len(images) > 0:
        batch = np.array(images)
        preds = model.predict(batch, verbose=0)
        for j, pred in enumerate(preds):
            pred_class = class_names[np.argmax(pred)]
            y_pred.append(pred_class)
            y_true.append(labels[j])
            print(f"Processing {filenames[j]} - True: {labels[j]} | Pred: {pred_class}")

    # confusion matrix + save
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    os.makedirs("confusion_matrices", exist_ok=True)
    save_path = f"confusion_matrices/{model_name}_confusion.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    result = {"confusion_matrix": cm.tolist(), "report": report}

    with open(f"confusion_matrices/{model_name}_report.json", "w") as f:
        json.dump(result, f, indent=4)

    return result




# ------------------------------------------------------
# Route: TP, TN, FP, FN calculation and visualization
@app.route("/metrics/tpfpfn/<model_name>", methods=["GET"])
def get_tp_tn_fp_fn(model_name):
    """
    Returns TP, TN, FP, FN values for each class of the given model
    based on its saved confusion matrix JSON.
    Example: http://127.0.0.1:5000/metrics/tpfpfn/mobilenet
    """
    model_name = model_name.lower()
    json_path = f"confusion_matrices/{model_name}_report.json"

    if not os.path.exists(json_path):
        return jsonify({"error": f"No report found for {model_name}. Please run /evaluate/{model_name} first."}), 404

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract confusion matrix
    if "confusion_matrix" not in data:
        return jsonify({"error": "Confusion matrix not found in report"}), 400

    cm = np.array(data["confusion_matrix"])
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    # Compute TP, TN, FP, FN
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    results = []
    for i, cls in enumerate(classes):
        results.append({
            "class": cls,
            "TP": int(tp[i]),
            "TN": int(tn[i]),
            "FP": int(fp[i]),
            "FN": int(fn[i])
        })

    # Optional: also generate a heatmap visualization
    plt.figure(figsize=(6, 4))
    df = pd.DataFrame(results).set_index("class")[["TP", "TN", "FP", "FN"]]
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{model_name.upper()} - TP/TN/FP/FN per Class")
    plt.tight_layout()
    save_path = f"confusion_matrices/{model_name}_tp_tn_fp_fn.png"
    plt.savefig(save_path)
    plt.close()

    return jsonify({
        "message": f"✅ TP/TN/FP/FN calculated for {model_name}",
        "results": results,
        "heatmap_path": save_path
    })


def predict_with_all_models(image_path):
    """Return predictions from all three models for a given image path."""
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    results = {}
    models_info = [
        ("mobilenet", (224, 224)),
        ("resnet", (224, 224)),
        ("efficientnet", (240, 240)),
        ("cnn", (224, 224))
    ]

    for model_name, target_size in models_info:
        try:
            print(f"🔍 Loading {model_name.upper()} for single prediction...")
            model = load_model_by_name(model_name)

            # Load & preprocess image
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # normalize
            if model_name.lower() == "cnn":
                img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(preds)
            top_label = class_names[top_idx]
            confidence = float(preds[top_idx])

            results[model_name] = {
                "predicted_class": top_label,
                "confidence": round(confidence * 100, 2),
                "all_confidences": {class_names[i]: float(preds[i]) for i in range(len(class_names))}
            }

        except Exception as e:
            results[model_name] = {"error": str(e)}
        finally:
            # Clear GPU memory
            tf.keras.backend.clear_session()
            gc.collect()

    return results

# ------------------------------------------------------
# Routes: evaluate single model (loads model, evaluates, frees memory)
# ------------------------------------------------------
@app.route("/evaluate/mobilenet", methods=["GET"])
def evaluate_mobilenet():
    model = None
    try:
        model = load_model_by_name("mobilenet")
        result = evaluate_and_save(model, "mobilenet", batch_size=16)  
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if model is not None:
            tf.keras.backend.clear_session()
            del model
            gc.collect()
    if "error" in result:
        return jsonify(result), 404
    return jsonify({"message": "✅ MobileNet evaluation done!", "classes": class_names, "report": result["report"]})

@app.route("/evaluate/resnet", methods=["GET"])
def evaluate_resnet():
    model = None
    try:
        model = load_model_by_name("resnet")
        result = evaluate_and_save(model, "resnet", batch_size=2)  # small batch for ResNet (2)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if model is not None:
            tf.keras.backend.clear_session()
            del model
            gc.collect()
    if "error" in result:
        return jsonify(result), 404
    return jsonify({"message": "✅ ResNet evaluation done!", "classes": class_names, "report": result["report"]})

@app.route("/evaluate/efficientnet", methods=["GET"])
def evaluate_efficientnet():
    model = None
    try:
        model = load_model_by_name("efficientnet")
        result = evaluate_and_save(model, "efficientnet", batch_size=2)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if model is not None:
            tf.keras.backend.clear_session()
            del model
            gc.collect()
    if "error" in result:
        return jsonify(result), 404
    return jsonify({"message": "✅ EfficientNet evaluation done!", "classes": class_names, "report": result["report"]})


@app.route("/evaluate/cnn", methods=["GET"])
def evaluate_cnn():
    model = None
    try:
        model = load_model_by_name("cnn")
        result = evaluate_and_save(model, "cnn", batch_size=16)  # CNN can handle bigger batches
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if model is not None:
            tf.keras.backend.clear_session()
            del model
            gc.collect()
    if "error" in result:
        return jsonify(result), 404
    return jsonify({"message": "✅ CNN evaluation done!", "classes": class_names, "report": result["report"]})


# ------------------------------------------------------
# Evaluate all models sequentially (loads each one at a time)
# ------------------------------------------------------
@app.route("/evaluate/all", methods=["GET"])
def evaluate_all_models():
    results = {}
    for mname, bsize in [("mobilenet", 16), ("resnet", 2), ("efficientnet", 2), ("cnn", 16)]:
        model = None
        try:
            model = load_model_by_name(mname)
            results[mname] = evaluate_and_save(model, mname, batch_size=bsize)
        except Exception as e:
            results[mname] = {"error": str(e)}
        finally:
            if model is not None:
                tf.keras.backend.clear_session()
                del model
                gc.collect()

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return jsonify({"message": "✅ All models evaluated successfully!", "models": list(results.keys()), "results": results})

# ------------------------------------------------------
# Confusion image route and metrics route
# ------------------------------------------------------
@app.route("/confusion/<model_name>")
def get_confusion(model_name):
    model_name = model_name.lower()
    path = f"confusion_matrices/{model_name}_confusion.png"
    if not os.path.exists(path):
        return jsonify({"error": f"No confusion matrix found for {model_name}. Please run /evaluate/{model_name} first."}), 404
    return send_file(path, mimetype='image/png')

@app.route("/metrics/<model_name>", methods=["GET"])
def get_metrics_for_model(model_name):
    model_name = model_name.lower()
    path = f"confusion_matrices/{model_name}_report.json"
    if not os.path.exists(path):
        return jsonify({"error": "Run evaluation first"}), 404
    with open(path) as f:
        data = json.load(f)
    return jsonify(data)

# ------------------------------------------------------
# Graph plotting (unchanged)
# ------------------------------------------------------
@app.route("/graph/<metric>")
def graph(metric):
    metric_map = {"f1": ("val_f1_score", "F1-Score"), "accuracy": ("val_accuracy", "Accuracy"), "loss": ("val_loss", "Loss")}
    if metric not in metric_map:
        return jsonify({"error": "Invalid metric"}), 400
    key, label = metric_map[metric]
    plt.figure(figsize=(8, 6))
    plt.plot(hist1[key], label=f"MobileNet {label}")
    plt.plot(hist2[key], label=f"ResNet {label}")
    plt.plot(hist3[key], label=f"EfficientNet {label}")
    if key in hist4:
        plt.plot(hist4[key], label=f"CNN {label}")
    else:
    # CNN does not have this metric → use zeros or skip
        plt.plot([0] * len(hist4), label=f"CNN {label} (N/A)")

    plt.title(f"Validation {label} Comparison") 
    plt.xlabel("Epochs")    
    plt.ylabel(label)   
    plt.legend()    
    plt.grid(True)  
    buf = io.BytesIO()  
    plt.savefig(buf, format='png')      
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')



@app.route("/heatmap/<model_name>")
def get_heatmap(model_name):
    model_name = model_name.lower()
    path = f"confusion_matrices/{model_name}_tp_tn_fp_fn.png"
    if not os.path.exists(path):
        return jsonify({"error": "No heatmap found. Run /metrics/tpfpfn/<model_name> first."}), 404
    return send_file(path, mimetype='image/png')


# ------------------------------------------------------
# Run
# ------------------------------------------------------
from flask import request       
import tempfile 





@app.route("/predict/upload", methods=["POST"]) 
def predict_uploaded_image():   
    """
    Upload an image and get predictions from all models.
    Example (cURL):
      curl -X POST -F "image=@test/NORMAL/sample1.jpeg" http://127.0.0.1:5000/predict/upload
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    img_file = request.files["image"]

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:   
        img_path = tmp.name     
        img_file.save(img_path) 

    # Get predictions from all models
    results = predict_with_all_models(img_path) 

    # Find best model by confidence
    best_model = max(results.items(), key=lambda x: x[1].get("confidence", 0))[0]   

    # Cleanup temporary file
    os.remove(img_path) 

    return jsonify({
        "message": "Predictions for uploaded image",
        "best_model": best_model,   
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# if __name__ == "__main__":
#     app.run(debug=False, port=5000, use_reloader=False)

