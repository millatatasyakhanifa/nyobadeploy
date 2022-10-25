# Import Library
import requests
from keras.preprocessing import image
import urllib.request
from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print(os.getcwd())
model = keras.models.load_model("model_A2_.h5")
label = ["Hourse", "Human"]

app = Flask(__name__)


def predict_label(img):
    i = np.asarray(img) / 255.0
    i = i.reshape(150, 150, 3)
    p = model.predict(i)
    result = label[np.argmax(p)]
    return result


@app.route("/predict", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})

    if request.method == "POST":
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((150, 150), Image.NEAREST)
        p = predict_label(img)
    return p


if __name__ == "__main__":
    app.run(debug=True)
