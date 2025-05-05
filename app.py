## app.py
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import onnxruntime as ort
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__, static_folder="web", template_folder="web")
app.config['UPLOAD_FOLDER'] = 'web/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB Verbindung
client = MongoClient("mongodb+srv://albanese11:EvianWasser1*@mdm-aca-db.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
db = client["image_logs"]
collection = db["predictions"]

session = ort.InferenceSession("model.onnx")

with open("labels_map.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.repeat(img, 10, axis=0)
    return img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = preprocess(filepath)
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    prediction = labels[idx]
    confidence = float(probs[idx])

    # Logging in MongoDB
    collection.insert_one({
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    })

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=round(confidence, 2), 
                           image_url=f"/uploads/{filename}")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/logs", methods=["GET"])
def logs():
    logs = list(collection.find().sort("timestamp", -1).limit(5))
    return render_template("logs.html", logs=logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)